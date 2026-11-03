#!/usr/bin/env python3
"""
DeepStream 6.3 pipeline for Jetson Xavier NX -- DUAL INDEPENDENT INFERENCE ARCHITECTURE
(v2: OSD + metadata extraction moved PER-BRANCH, before the merge stage)

  Boson 640 thermal (NV12, native 640x512, no resize before inference)
    -> nvstreammux_thermal (batch=1)
    -> nvinfer_thermal   (thermal.engine, thermal-specific config)
    -> tracker_thermal
    -> nvdsosd_thermal        [draws boxes at NATIVE 640x512 -- see "WHY OSD MOVED" below]
    -> RGBA convert -> probe_thermal (metadata extraction + save enqueue) -> back to NV12

  Logitech C270 (MJPEG, native 1280x720 @ 30fps, hardware-decoded)
    -> nvv4l2decoder (hardware JPEG decode -> I420 NVMM -> converted to NV12, no resize)
    -> nvstreammux_rgb (batch=1)
    -> nvinfer_rgb       (rgb.engine, rgb-specific config)
    -> tracker_rgb
    -> nvdsosd_rgb             [draws boxes at NATIVE 1280x720]
    -> RGBA convert -> probe_rgb (metadata extraction + save enqueue) -> back to NV12

  Both ALREADY-ANNOTATED streams
    -> nvstreammux_merge (batch=2)   [pure video compositing now -- see below]
    -> nvmultistreamtiler (1x2 grid)
    -> tee:
         branch A -> RTSP server (intranet)          [highest priority]
         branch B -> (optional) local display via nvvideosink [second priority]


Run:
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  export USE_NEW_NVSTREAMMUX=no
  python3 pipeline_app.py \
      --boson /dev/v4l/by-id/usb-FLIR_Boson_66959-video-index0 \
      --webcam /dev/v4l/by-id/usb-046d_0825_A7767C50-video-index0 \
      --thermal-engine thermal.engine --thermal-config config_infer_thermal.txt \
      --rgb-engine rgb.engine --rgb-config config_infer_rgb.txt \
      --rtsp-port 8554 --rtsp-mount /ds --out-dir ./output --save-every 150
"""
import argparse
import os
import sys
import json
import queue
import threading
import datetime

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

DEEPSTREAM_VERSION = "6.3"
DEEPSTREAM_ROOT = f"/opt/nvidia/deepstream/deepstream-{DEEPSTREAM_VERSION}"

sys.path.append(f"{DEEPSTREAM_ROOT}/lib")
import pyds

TRACKER_LIB = f"{DEEPSTREAM_ROOT}/lib/libnvds_nvmultiobjecttracker.so"

CLASS_NAMES = {}  # populated from labels.txt at startup
frame_counter = {"n": 0}

# Tracking visualization only
TRACKING_ENABLED = False
TRAIL_POINTS = 30
TRAIL_LINE_WIDTH = 6
TRAIL_ALPHA = 0.90
track_history = {"boson_thermal": {}, "logitech_c270": {}}


save_queue = queue.Queue(maxsize=20)
save_stats = {"enqueued": 0, "dropped": 0, "saved": 0, "errors": 0}


def load_labels(path="labels.txt"):
    if os.path.exists(path):
        with open(path) as f:
            for i, line in enumerate(f):
                CLASS_NAMES[i] = line.strip()


# =============================================================================
# SOURCE HELPERS -- each returns a Gst.Bin with a single ghost "src" pad emitting
# NV12 in NVMM memory at the camera's NATIVE resolution. No resizing happens here.
# =============================================================================
def make_boson_source(device_path, width=640, height=512, framerate=30, index=0):
    """
    FLIR Boson 640, thermal, USB/UVC.
    Natively NV12 @ 640x512 -> v4l2src delivers this directly, nvvideoconvert is
    effectively a passthrough/NVMM-wrap, no scaling occurs.
    """
    bin_name = f"source-bin-boson-{index:02d}"
    nbin = Gst.Bin.new(bin_name)

    src = Gst.ElementFactory.make("v4l2src", f"v4l2src-boson-{index}")
    src.set_property("device", device_path)

    caps_raw = Gst.ElementFactory.make("capsfilter", f"boson-raw-caps-{index}")
    caps_raw.set_property(
        "caps",
        Gst.Caps.from_string(f"video/x-raw,format=NV12,width={width},height={height},framerate={framerate}/1")
    )

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"boson-nvvidconv-{index}")

    caps_nvmm = Gst.ElementFactory.make("capsfilter", f"boson-nvmm-caps-{index}")
    caps_nvmm.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))

    for e in [src, caps_raw, nvvidconv, caps_nvmm]:
        nbin.add(e)

    src.link(caps_raw)
    caps_raw.link(nvvidconv)
    nvvidconv.link(caps_nvmm)

    ghost_pad = Gst.GhostPad.new("src", caps_nvmm.get_static_pad("src"))
    nbin.add_pad(ghost_pad)
    return nbin


def make_logitech_source(device_path, width=1280, height=720, framerate=30, index=1):
    """
    Logitech C270, RGB, USB/UVC.
    Captured at its native MJPEG 1280x720@30 -- the highest mode this sensor supports --
    and hardware-decoded via nvv4l2decoder. No software JPEG decode (no jpegdec/
    avdec_mjpeg) and no OpenCV touches this frame. jpegparse is inserted only to
    guarantee clean JPEG frame boundaries for the hardware decoder; it does not
    decode or resize anything itself.

    NOTE: on this DS 6.3 / L4T 35.5 build, nvv4l2decoder negotiates its output as
    I420, not NV12, for this MJPEG source (confirmed via gst-launch-1.0 caps
    trace -- forcing NV12 directly on the decoder's output caused
    "streaming stopped, reason not-negotiated (-4)"). We accept I420 from the
    decoder, then convert to NV12 via nvvideoconvert so the rest of the pipeline
    (which expects NV12 into nvstreammux_rgb) is unaffected.
    """
    bin_name = f"source-bin-logitech-{index:02d}"
    nbin = Gst.Bin.new(bin_name)

    src = Gst.ElementFactory.make("v4l2src", f"v4l2src-logitech-{index}")
    src.set_property("device", device_path)
    src.set_property("io-mode", 2)  # mmap io-mode -- required for clean MJPG negotiation
                                     # with this UVC driver; confirmed via gst-launch-1.0 test

    caps_mjpeg = Gst.ElementFactory.make("capsfilter", f"logitech-mjpeg-caps-{index}")
    caps_mjpeg.set_property(
        "caps",
        Gst.Caps.from_string(f"image/jpeg,width={width},height={height},framerate={framerate}/1")
    )

    jpegparse = Gst.ElementFactory.make("jpegparse", f"logitech-jpegparse-{index}")

    decoder = Gst.ElementFactory.make("nvv4l2decoder", f"logitech-nvv4l2dec-{index}")
    decoder.set_property("mjpeg", 1)  # hardware MJPEG decode path on Jetson

    caps_nvmm = Gst.ElementFactory.make("capsfilter", f"logitech-nvmm-caps-{index}")
    caps_nvmm.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=I420"))

    nvvidconv_to_nv12 = Gst.ElementFactory.make("nvvideoconvert", f"logitech-to-nv12-{index}")
    caps_nv12 = Gst.ElementFactory.make("capsfilter", f"logitech-nv12-caps-{index}")
    caps_nv12.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))

    for e in [src, caps_mjpeg, jpegparse, decoder, caps_nvmm, nvvidconv_to_nv12, caps_nv12]:
        nbin.add(e)

    src.link(caps_mjpeg)
    caps_mjpeg.link(jpegparse)
    jpegparse.link(decoder)

    decoder_src_pad = decoder.get_static_pad("src")
    caps_nvmm_sink_pad = caps_nvmm.get_static_pad("sink")
    link_result_decoder_to_nvmm = decoder_src_pad.link(caps_nvmm_sink_pad)
    print(f"[link-debug] nvv4l2decoder -> caps_nvmm link result: {link_result_decoder_to_nvmm}")

    caps_nvmm.link(nvvidconv_to_nv12)
    nvvidconv_to_nv12.link(caps_nv12)

    ghost_pad = Gst.GhostPad.new("src", caps_nv12.get_static_pad("src"))
    nbin.add_pad(ghost_pad)
    return nbin


# =============================================================================
# STREAMMUX / PGIE / TRACKER HELPERS
# =============================================================================
def make_streammux(name, batch_size, width, height, live_source=1, batched_push_timeout=4000, enable_padding=0):
    streammux = Gst.ElementFactory.make("nvstreammux", name)
    streammux.set_property("batch-size", batch_size)
    streammux.set_property("width", width)
    streammux.set_property("height", height)
    streammux.set_property("batched-push-timeout", batched_push_timeout)
    streammux.set_property("live-source", live_source)
    streammux.set_property("enable-padding", enable_padding)
    return streammux


def make_pgie(name, config_path, engine_path, batch_size, enable_dla=False, dla_core=0):
    pgie = Gst.ElementFactory.make("nvinfer", name)
    pgie.set_property("qos", False)
    pgie.set_property("config-file-path", config_path)
    pgie.set_property("batch-size", batch_size)
    if engine_path:
        pgie.set_property("model-engine-file", engine_path)
    if enable_dla:
        pass
    return pgie


def make_tracker(name, config_path, tracker_width=640, tracker_height=384, gpu_id=0):
    tracker = Gst.ElementFactory.make("nvtracker", name)
    tracker.set_property("qos", False)
    tracker.set_property("tracker-width", tracker_width)
    tracker.set_property("tracker-height", tracker_height)
    tracker.set_property("ll-lib-file", TRACKER_LIB)
    tracker.set_property("ll-config-file", config_path)
    tracker.set_property("gpu-id", gpu_id)
    tracker.set_property("display-tracking-id", 0)
    return tracker


def make_tracking_overlay_probe(cam_name):
    def probe(pad, info, u_data):
        if not TRACKING_ENABLED:
            return Gst.PadProbeReturn.OK

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            history = track_history[cam_name]
            active_ids = set()
            segments = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                track_id = int(obj_meta.object_id)

                if track_id != 18446744073709551615:
                    active_ids.add(track_id)

                    rect = obj_meta.rect_params
                    cx = float(rect.left) + float(rect.width) / 2.0
                    cy = float(rect.top) + float(rect.height) / 2.0

                    points = history.setdefault(track_id, [])
                    points.append((cx, cy))

                    if len(points) > TRAIL_POINTS:
                        del points[:-TRAIL_POINTS]

                    # Exactly one ID label: [class_id]:track_id
                    obj_meta.text_params.display_text = f"[{int(obj_meta.class_id)}]:{track_id}"

                    if len(points) >= 2:
                        for i in range(len(points) - 1):
                            x1, y1 = points[i]
                            x2, y2 = points[i + 1]
                            segments.append((x1, y1, x2, y2))

                l_obj = l_obj.next

            for tid in list(history.keys()):
                if tid not in active_ids:
                    del history[tid]

            if segments:
                pos = 0
                while pos < len(segments):
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    count = min(16, len(segments) - pos)
                    display_meta.num_lines = count

                    for i in range(count):
                        x1, y1, x2, y2 = segments[pos + i]
                        line = display_meta.line_params[i]
                        line.x1 = int(x1)
                        line.y1 = int(y1)
                        line.x2 = int(x2)
                        line.y2 = int(y2)
                        line.line_width = TRAIL_LINE_WIDTH
                        line.line_color.set(0.10, 0.85, 1.0, TRAIL_ALPHA)

                    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                    pos += count

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    return probe


def make_osd(name):
    nvosd = Gst.ElementFactory.make("nvdsosd", name)
    nvosd.set_property("qos", False)
    nvosd.set_property("display-text", 1)
    nvosd.set_property("process-mode", 1)  # GPU draw mode
    return nvosd


# =============================================================================
# BACKGROUND SAVE WORKER -- unchanged. All OpenCV/numpy/disk/JSON work happens
# here, on its own thread -- never on the GStreamer pipeline thread.
# =============================================================================
def save_worker(out_dir, stop_event):
    import numpy as np
    import cv2

    while not stop_event.is_set():
        try:
            item = save_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:  # sentinel for shutdown
            break

        try:
            frame_bgr, detections, cam_name, ts = item

            raw_dir = os.path.join(out_dir, "raw", cam_name)
            ann_dir = os.path.join(out_dir, "annotated", cam_name)
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(ann_dir, exist_ok=True)

            raw_path = os.path.join(raw_dir, f"{ts}.jpg")
            cv2.imwrite(raw_path, frame_bgr)

            # NOTE: since nvdsosd already burned the boxes into this frame upstream
            # (per-branch, at native resolution), "annotated" here is effectively
            # the same frame nvdsosd drew -- we still redraw with cv2 for a
            # consistent, independently-verifiable JPEG output and for the
            # per-class crop extraction below, which needs raw pixel access.
            annotated = frame_bgr.copy()
            for det in detections:
                b = det["bbox"]
                x1, y1 = int(b["left"]), int(b["top"])
                x2, y2 = int(b["left"] + b["width"]), int(b["top"] + b["height"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{det["class_name"]} ID:{det["track_id"]}'
                cv2.putText(annotated, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cls_dir = os.path.join(out_dir, "crops", cam_name, det["class_name"])
                os.makedirs(cls_dir, exist_ok=True)
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
                if x2c > x1c and y2c > y1c:
                    crop = frame_bgr[y1c:y2c, x1c:x2c]
                    crop_path = os.path.join(cls_dir, f'{ts}_id{det["track_id"]}.jpg')
                    cv2.imwrite(crop_path, crop)

            ann_path = os.path.join(ann_dir, f"{ts}.jpg")
            cv2.imwrite(ann_path, annotated)

            log_path = os.path.join(out_dir, "detections.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": ts,
                    "camera": cam_name,
                    "raw_frame": raw_path,
                    "annotated_frame": ann_path,
                    "detections": detections,
                }) + "\n")

            save_stats["saved"] += 1
        except Exception as e:
            save_stats["errors"] += 1
            print(f"[save-worker] error: {e}", file=sys.stderr)
        finally:
            save_queue.task_done()


# =============================================================================
# GSTREAMER PAD PROBE -- now attached PER-BRANCH, immediately after that
# branch's own nvdsosd + RGBA conversion, while metadata is still guaranteed to
# be in the correct native-resolution coordinate space. cam_name is fixed per
# branch (not looked up via pad_index) since each branch's streammux is
# batch-size=1, single source, always pad_index 0.
# =============================================================================
def make_branch_probe(cam_name, save_every, out_dir):
    def probe(pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_counter["n"] += 1
            do_save = (frame_meta.frame_num % save_every == 0)

            if do_save:
                detections = []
                l_obj = frame_meta.obj_meta_list
                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break
                    rect = obj_meta.rect_params
                    detections.append({
                        "track_id": obj_meta.object_id,
                        "class_id": obj_meta.class_id,
                        "class_name": CLASS_NAMES.get(obj_meta.class_id, obj_meta.obj_label),
                        "confidence": float(obj_meta.confidence),
                        "bbox": {
                            "left": float(rect.left), "top": float(rect.top),
                            "width": float(rect.width), "height": float(rect.height),
                        },
                        "camera": cam_name,
                    })
                    l_obj = l_obj.next

                if detections:
                    try:
                        import numpy as np
                        import cv2
                        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                        rgba = np.array(n_frame, copy=True, order='C')
                        frame_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        save_stats["enqueued"] += 1
                        save_queue.put_nowait((frame_bgr, detections, cam_name, ts))
                    except queue.Full:
                        save_stats["dropped"] += 1
                    except Exception as e:
                        save_stats["errors"] += 1
                        print(f"[probe:{cam_name}] enqueue error: {e}", file=sys.stderr)

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK
    return probe


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}: {debug}")
        loop.quit()
    return True


def start_rtsp_server(rtsp_port, rtsp_mount, udp_port, codec):
    encoding_name = "H264" if codec == "h264" else "H265"
    server = GstRtspServer.RTSPServer()
    server.set_service(str(rtsp_port))
    mounts = server.get_mount_points()
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_latency(50)
    factory.set_launch(
        f'( udpsrc name=pay0 port={udp_port} buffer-size=524288 '
        f'caps="application/x-rtp,media=video,clock-rate=90000,encoding-name={encoding_name},payload=96" )'
    )
    factory.set_shared(True)
    mounts.add_factory(rtsp_mount, factory)
    server.attach(None)
    print(f"RTSP stream ready at rtsp://<jetson-ip>:{rtsp_port}{rtsp_mount}  (codec={encoding_name})")
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boson", default="/dev/video1")
    parser.add_argument("--webcam", default="/dev/video0")
    parser.add_argument("--boson-w", type=int, default=640)
    parser.add_argument("--boson-h", type=int, default=512)
    parser.add_argument("--webcam-w", type=int, default=1280)
    parser.add_argument("--webcam-h", type=int, default=720)
    parser.add_argument("--webcam-fps", type=int, default=30)

    parser.add_argument("--thermal-engine", default="thermal.engine")
    parser.add_argument("--thermal-config", default="config_infer_thermal.txt")
    parser.add_argument("--rgb-engine", default="rgb.engine")
    parser.add_argument("--rgb-config", default="config_infer_rgb.txt")

    parser.add_argument("--thermal-tracker-config", default="tracker_config_nvsort.yml")
    parser.add_argument("--rgb-tracker-config", default="tracker_config_nvsort.yml")

    parser.add_argument("--thermal-timeout", type=int, default=1000,
                         help="streammux_thermal batched-push-timeout in microseconds.")
    parser.add_argument("--rgb-timeout", type=int, default=1000,
                         help="streammux_rgb batched-push-timeout in microseconds.")
    parser.add_argument("--merge-timeout", type=int, default=1000,
                         help="streammux_merge batched-push-timeout in microseconds.")

    parser.add_argument("--merge-w", type=int, default=1280,
                         help="Common width nvstreammux_merge scales both ALREADY-"
                              "ANNOTATED streams to before tiling. Boxes are already "
                              "burned into pixels at this point, so this resize is a "
                              "normal uniform image scale -- it cannot cause box "
                              "misalignment the way the old post-merge OSD design could.")
    parser.add_argument("--merge-h", type=int, default=720)

    parser.add_argument("--rtsp-port", type=int, default=8554)
    parser.add_argument("--rtsp-mount", default="/ds")
    parser.add_argument("--udp-port", type=int, default=5400)
    parser.add_argument("--out-dir", default="./output")
    parser.add_argument("--save-every", type=int, default=150)
    parser.add_argument("--enable-display", action="store_true")
    parser.add_argument("--enable-tracking", action="store_true",
                        help="Enable tracking IDs and trajectory lines")

    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    parser.add_argument("--enable-dla", action="store_true")
    args = parser.parse_args()
    global TRACKING_ENABLED
    TRACKING_ENABLED = args.enable_tracking


    load_labels("labels.txt")
    os.makedirs(args.out_dir, exist_ok=True)

    stop_event = threading.Event()
    worker = threading.Thread(target=save_worker, args=(args.out_dir, stop_event), daemon=True)
    worker.start()

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # =========================================================================
    # THERMAL BRANCH: source -> streammux_thermal(batch=1) -> pgie_thermal ->
    # tracker_thermal -> nvdsosd_thermal (draws at native 640x512) -> RGBA
    # convert -> probe_thermal -> back to NV12 -> ready for the merge stage.
    # =========================================================================
    boson_source = make_boson_source(args.boson, args.boson_w, args.boson_h, index=0)
    streammux_thermal = make_streammux("stream-muxer-thermal", batch_size=1,
                                        width=args.boson_w, height=args.boson_h,
                                        batched_push_timeout=args.thermal_timeout)
    pgie_thermal = make_pgie("primary-inference-thermal", args.thermal_config,
                              args.thermal_engine, batch_size=1,
                              enable_dla=args.enable_dla)
    tracker_thermal = make_tracker("tracker-thermal", args.thermal_tracker_config,
                                    tracker_width=min(640, args.boson_w),
                                    tracker_height=min(384, args.boson_h))
    osd_thermal = make_osd("osd-thermal")

    nvvidconv_rgba_thermal = Gst.ElementFactory.make("nvvideoconvert", "convertor-rgba-thermal")
    caps_rgba_thermal = Gst.ElementFactory.make("capsfilter", "caps-rgba-thermal")
    caps_rgba_thermal.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

    nvvidconv_back_thermal = Gst.ElementFactory.make("nvvideoconvert", "convertor-back-nv12-thermal")
    caps_nv12_thermal_out = Gst.ElementFactory.make("capsfilter", "caps-nv12-thermal-out")
    caps_nv12_thermal_out.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))

    for e in [boson_source, streammux_thermal, pgie_thermal, tracker_thermal, osd_thermal,
              nvvidconv_rgba_thermal, caps_rgba_thermal, nvvidconv_back_thermal, caps_nv12_thermal_out]:
        pipeline.add(e)

    thermal_sinkpad = streammux_thermal.get_request_pad("sink_0")
    boson_source.get_static_pad("src").link(thermal_sinkpad)
    streammux_thermal.link(pgie_thermal)
    pgie_thermal.link(tracker_thermal)
    tracker_thermal.link(osd_thermal)
    tracker_thermal.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        make_tracking_overlay_probe("boson_thermal"),
        None
    )

    osd_thermal.link(nvvidconv_rgba_thermal)
    nvvidconv_rgba_thermal.link(caps_rgba_thermal)
    caps_rgba_thermal.link(nvvidconv_back_thermal)
    nvvidconv_back_thermal.link(caps_nv12_thermal_out)

    caps_rgba_thermal.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        make_branch_probe("boson_thermal", args.save_every, args.out_dir),
        None
    )

    # =========================================================================
    # RGB BRANCH: source (MJPEG 1280x720 hw-decoded) -> streammux_rgb(batch=1)
    # -> pgie_rgb -> tracker_rgb -> nvdsosd_rgb (draws at native 1280x720) ->
    # RGBA convert -> probe_rgb -> back to NV12 -> ready for the merge stage.
    # =========================================================================
    logitech_source = make_logitech_source(args.webcam, args.webcam_w, args.webcam_h,
                                            args.webcam_fps, index=1)
    streammux_rgb = make_streammux("stream-muxer-rgb", batch_size=1,
                                    width=args.webcam_w, height=args.webcam_h,
                                    batched_push_timeout=args.rgb_timeout)
    pgie_rgb = make_pgie("primary-inference-rgb", args.rgb_config,
                          args.rgb_engine, batch_size=1,
                          enable_dla=args.enable_dla)
    tracker_rgb = make_tracker("tracker-rgb", args.rgb_tracker_config,
                                tracker_width=640, tracker_height=384)
    osd_rgb = make_osd("osd-rgb")

    nvvidconv_rgba_rgb = Gst.ElementFactory.make("nvvideoconvert", "convertor-rgba-rgb")
    caps_rgba_rgb = Gst.ElementFactory.make("capsfilter", "caps-rgba-rgb")
    caps_rgba_rgb.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

    nvvidconv_back_rgb = Gst.ElementFactory.make("nvvideoconvert", "convertor-back-nv12-rgb")
    caps_nv12_rgb_out = Gst.ElementFactory.make("capsfilter", "caps-nv12-rgb-out")
    caps_nv12_rgb_out.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))

    for e in [logitech_source, streammux_rgb, pgie_rgb, tracker_rgb, osd_rgb,
              nvvidconv_rgba_rgb, caps_rgba_rgb, nvvidconv_back_rgb, caps_nv12_rgb_out]:
        pipeline.add(e)

    rgb_srcpad = logitech_source.get_static_pad("src")
    rgb_sinkpad = streammux_rgb.get_request_pad("sink_0")
    link_result_rgb_to_streammux = rgb_srcpad.link(rgb_sinkpad)
    print(f"[link-debug] logitech_source -> streammux_rgb link result: {link_result_rgb_to_streammux}")

    streammux_rgb.link(pgie_rgb)
    pgie_rgb.link(tracker_rgb)
    tracker_rgb.link(osd_rgb)
    tracker_rgb.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        make_tracking_overlay_probe("logitech_c270"),
        None
    )

    osd_rgb.link(nvvidconv_rgba_rgb)
    nvvidconv_rgba_rgb.link(caps_rgba_rgb)
    caps_rgba_rgb.link(nvvidconv_back_rgb)
    nvvidconv_back_rgb.link(caps_nv12_rgb_out)

    caps_rgba_rgb.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        make_branch_probe("logitech_c270", args.save_every, args.out_dir),
        None
    )

    # =========================================================================
    # MERGE STAGE: pure video compositing now. Both streams already have boxes
    # burned into their pixels at native resolution -- this resize/rebatch is a
    # normal uniform image scale and cannot desync box position from the frame.
    # =========================================================================
    streammux_merge = make_streammux("stream-muxer-merge", batch_size=2,
                                      width=args.merge_w, height=args.merge_h,
                                      batched_push_timeout=args.merge_timeout, enable_padding=1)
    pipeline.add(streammux_merge)

    merge_sink_thermal = streammux_merge.get_request_pad("sink_0")
    merge_sink_rgb = streammux_merge.get_request_pad("sink_1")
    caps_nv12_thermal_out.get_static_pad("src").link(merge_sink_thermal)
    caps_nv12_rgb_out.get_static_pad("src").link(merge_sink_rgb)

    # =========================================================================
    # DOWNSTREAM: tiler -> tee -> RTSP branch, display branch. No metadata
    # dependency past this point, so no probe/save branch needed here anymore.
    # =========================================================================
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", 1)
    tiler.set_property("columns", 2)
    tiler.set_property("width", args.merge_w * 2)
    tiler.set_property("height", args.merge_h)

    tee = Gst.ElementFactory.make("tee", "tee")

    # --- Branch A: RTSP out [HIGHEST PRIORITY] ---
    queue_rtsp = Gst.ElementFactory.make("queue", "queue-rtsp")
    queue_rtsp.set_property("leaky", 2)
    queue_rtsp.set_property("max-size-buffers", 2)
    queue_rtsp.set_property("max-size-bytes", 0)
    queue_rtsp.set_property("max-size-time", 100000000)

    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    caps_i420 = Gst.ElementFactory.make("capsfilter", "caps-i420")
    caps_i420.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=I420"))

    if args.codec == "h264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    else:
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
    encoder.set_property("bitrate", 6000000)
    encoder.set_property("insert-sps-pps", 1)
    encoder.set_property("idrinterval", 15)
    encoder.set_property("preset-level", 1)
    encoder.set_property("maxperf-enable", True)
    encoder.set_property("iframeinterval", 15)
    encoder.set_property("control-rate", 1)
    encoder.set_property("profile", 2)
    encoder.set_property("insert-vui", 1)

    if args.codec == "h264":
        parse = Gst.ElementFactory.make("h264parse", "parse")
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    else:
        parse = Gst.ElementFactory.make("h265parse", "parse")
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
    rtppay.set_property("pt", 96)
    rtppay.set_property("config-interval", 1)

    udpsink = Gst.ElementFactory.make("udpsink", "udpsink")
    udpsink.set_property("host", "127.0.0.1")
    udpsink.set_property("port", args.udp_port)
    udpsink.set_property("sync", 0)
    udpsink.set_property("async", 0)

    # --- Branch B: local display [SECOND PRIORITY] ---
    display_elems = []
    if args.enable_display:
        queue_display = Gst.ElementFactory.make("queue", "queue-display")
        queue_display.set_property("max-size-buffers", 1)
        queue_display.set_property("max-size-bytes", 0)
        queue_display.set_property("max-size-time", 0)
        queue_display.set_property("leaky", 2)

        nvvidconv_disp = Gst.ElementFactory.make(
            "nvvideoconvert", "convertor-display"
        )

        # nv3dsink is the confirmed-working hardware-accelerated Jetson display
        # sink (EGL) on this DS 6.3 build -- it accepts NVMM buffers directly,
        # no CPU conversion or scale needed. nveglglessink was tried first but
        # failed ("cannot handle NVRM surface array") since it doesn't support
        # this build's default nvvideoconvert output memory type; nvvideosink
        # failed to start at all; ximagesink worked but required CPU-side
        # videoconvert/videoscale, which was too slow. nv3dsink avoids all
        # three problems.
        caps_display = Gst.ElementFactory.make(
            "capsfilter", "caps-display"
        )
        caps_display.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM), format=NV12"
            )
        )

        overlaysink = Gst.ElementFactory.make(
            "nv3dsink", "overlaysink"
        )
        overlaysink.set_property("sync", False)
        overlaysink.set_property("async", False)
        overlaysink.set_property("qos", False)

        display_elems = [
            queue_display,
            nvvidconv_disp,
            caps_display,
            overlaysink
        ]

    for e in [tiler, tee, queue_rtsp, nvvidconv2, caps_i420, encoder, parse, rtppay, udpsink] + display_elems:
        pipeline.add(e)

    streammux_merge.link(tiler)
    tiler.link(tee)

    tee.get_request_pad("src_%u").link(queue_rtsp.get_static_pad("sink"))
    queue_rtsp.link(nvvidconv2)
    nvvidconv2.link(caps_i420)
    caps_i420.link(encoder)
    encoder.link(parse)
    parse.link(rtppay)
    rtppay.link(udpsink)

    if args.enable_display:
        display_tee_pad = tee.get_request_pad("src_%u")
        display_sink_pad = queue_display.get_static_pad("sink")
        r1 = display_tee_pad.link(display_sink_pad)
        print(f"[display-debug] tee -> queue_display link result: {r1}")

        r2 = queue_display.link(nvvidconv_disp)
        print(f"[display-debug] queue_display -> nvvidconv_disp link result: {r2}")

        r3 = nvvidconv_disp.link(caps_display)
        print(f"[display-debug] nvvidconv_disp -> caps_display link result: {r3}")

        r4 = caps_display.link(overlaysink)
        print(f"[display-debug] caps_display -> overlaysink link result: {r4}")

        def on_display_sink_pad_probe(pad, info, u_data):
            caps = pad.get_current_caps()
            print(f"[display-debug] overlaysink sink pad negotiated caps: {caps}")
            return Gst.PadProbeReturn.REMOVE
        overlaysink.get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, on_display_sink_pad_probe, None
        )

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    rtsp_server = start_rtsp_server(args.rtsp_port, args.rtsp_mount, args.udp_port, args.codec)

    def print_stats():
        print(f"[save-stats] enqueued={save_stats['enqueued']} "
              f"dropped={save_stats['dropped']} saved={save_stats['saved']} "
              f"errors={save_stats['errors']} queue_depth={save_queue.qsize()}")
        return True
    GLib.timeout_add_seconds(10, print_stats)

    print("Starting dual-pipeline v2 (per-branch OSD, thermal + RGB, independent inference)...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        stop_event.set()
        save_queue.put(None)
        worker.join(timeout=5)


if __name__ == "__main__":
    main()
