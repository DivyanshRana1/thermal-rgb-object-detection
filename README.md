# Thermal-Rgb-Object-Detection
Real-Time Thermal and RGB Object Detection and Tracking Using NVIDIA DeepStream on Jetson Xavier NX, Jetson Nano, Jetson AGX Xavier
# Multi-Camera Object Detection and Tracking with NVIDIA DeepStream

A reusable starting point for building real-time object-detection and
tracking pipelines with multiple cameras on NVIDIA Jetson or NVIDIA GPU
systems.

> **Important:** This README describes a general multi-camera
> architecture. It is not a plug-and-play universal camera driver.
> Camera formats, source creation, batching, inference configuration,
> tracker settings, and output handling must be adapted to the hardware
> and DeepStream version in use.

## 1. Project overview

This project provides a base for processing multiple camera streams in
one DeepStream application. It is intended to be adapted for:

-   USB cameras (V4L2)
-   RTSP/IP cameras
-   CSI cameras supported by the platform
-   Mixed camera types, provided their source and format handling are
    configured

Typical processing flow:

``` text
Camera 1 ─┐
Camera 2 ─┤
Camera N ─┘
    │
    ▼
Source elements / camera configuration
    │
    ▼
Stream muxer (batching)
    │
    ▼
Object detector (one or more inference configurations)
    │
    ▼
Object tracker (optional)
    │
    ▼
Analytics / metadata / overlays
    │
    ├── Display
    ├── RTSP output
    ├── Recording
    └── Detection or event snapshots
```

The diagram is conceptual. The actual graph and supported outputs depend
on the application implementation.

## 2. Features to implement or configure

-   Add, remove, and identify camera sources
-   Configure source type, device path, or RTSP URL per camera
-   Configure resolution, frame rate, and input format
-   Batch streams using `nvstreammux` (or the muxer appropriate to the
    installed DeepStream release)
-   Configure detector model and inference precision
-   Enable or disable tracking
-   Preserve camera/source identity in metadata and output
-   Choose display, streaming, recording, and snapshot outputs
-   Log per-source errors and performance information

## 3. Hardware and software requirements

Record the versions tested by your deployment before publishing or
reproducing the project.

  Component          Deployment-specific value
  ------------------ ---------------------------------------------------
  NVIDIA platform    Jetson model or NVIDIA GPU system
  JetPack / CUDA     Specify installed versions
  DeepStream         Specify installed version
  TensorRT           Specify installed version
  Operating system   Specify distribution and release
  Python             Specify version
  Camera models      List camera model and connection type
  Detector           Model architecture, classes, and input dimensions

Use NVIDIA's official DeepStream documentation for installation and
version compatibility: https://docs.nvidia.com/metropolis/deepstream/

## 4. Suggested repository structure

Adapt this layout to the files actually used by your application. Do not
commit generated outputs or machine-specific credentials.

``` text
multi-camera-deepstream/
├── README.md
├── .gitignore
├── requirements.txt                 # Python dependencies, if applicable
├── pipeline_app.py                 # Main application
├── configs/
│   ├── camera_config.example.yaml  # Example source settings
│   ├── infer_camera_1.txt          # Inference config, if used
│   ├── infer_camera_2.txt
│   └── tracker_config.yml          # Tracker config, if used
├── models/
│   ├── model.onnx                 # Optional; check model licensing
│   └── labels.txt
├── sources/
│   └── camera_sources.py           # Optional source helpers
├── scripts/
│   ├── build_engine.sh             # Optional engine-generation script
│   └── run_pipeline.sh              # Optional launch script
└── docs/
    └── architecture.md               # Optional architecture notes
```

A project may use one shared detector, separate detectors per source, or
a detector selected by source type. Document which design your code
implements.

## 5. Camera configuration

Create a configuration entry for each camera. Keep credentials out of
committed files. For RTSP cameras, use environment variables or a local
ignored configuration file for usernames and passwords.

Example YAML (illustrative; your application must parse this schema):

``` yaml
cameras:
  - id: camera_1
    type: v4l2
    device: /dev/video0
    width: 640
    height: 480
    fps: 30

  - id: camera_2
    type: rtsp
    url: "${CAMERA_2_RTSP_URL}"
    width: 1920
    height: 1080
    fps: 25
```

This example is a configuration design, not a built-in DeepStream
format. Update the application to read it, validate it, and create the
correct source elements.

### USB / V4L2 cameras

1.  Identify the device:

    ``` bash
    v4l2-ctl --list-devices
    ```

2.  Inspect supported formats:

    ``` bash
    v4l2-ctl --device=/dev/video0 --list-formats-ext
    ```

3.  Prefer stable device paths when available:

    ``` bash
    ls -l /dev/v4l/by-id/
    ```

4.  Confirm that each camera can run at its selected resolution and
    frame rate simultaneously.

### RTSP / IP cameras

-   Confirm network reachability and the camera's stream URL.
-   Use the camera's supported codec, resolution, and frame rate.
-   Keep credentials out of source code and public configuration.
-   Test each stream independently before adding it to the multi-camera
    pipeline.

## 6. Adding another camera

Adding a camera is not necessarily just increasing a camera-count
variable. Verify each of these parts in your application:

1.  **Source creation:** instantiate and configure the source element
    for the new camera.
2.  **Unique identity:** assign a stable source ID and preserve it in
    metadata.
3.  **Muxer request pad:** connect the source to the correct muxer sink
    pad.
4.  **Batch size:** configure the muxer for the number of active sources
    and understand timeout behavior when sources have different frame
    rates.
5.  **Inference:** decide whether all cameras use one shared inference
    configuration or source-specific models.
6.  **Tracker:** decide whether tracking is global or independent per
    camera. IDs from independent views should not be assumed to
    represent the same physical object.
7.  **Output:** ensure overlays, display tiles, RTSP branches,
    recordings, and snapshots retain source identity.
8.  **Lifecycle:** handle camera disconnect, reconnect, EOS, and
    pipeline shutdown.
9.  **Performance:** measure throughput, latency, memory, GPU/DLA
    utilization, and dropped frames after adding the source.

Do not claim arbitrary camera-count support until the code handles
source creation, muxing, inference, tracking, and output dynamically or
has been tested for the stated fixed number of cameras.

## 7. Inference engine and model

DeepStream deployments may use ONNX models or TensorRT engine files.
Engine compatibility depends on the target platform, TensorRT/CUDA
versions, model, precision, and build settings.

Document:

-   Model source and license
-   Input dimensions and preprocessing
-   Class labels
-   ONNX export command or procedure
-   TensorRT engine build procedure
-   Inference configuration file
-   Precision (FP32, FP16, INT8) and any calibration requirements

Example placeholder:

``` bash
# Replace with the export/build command and arguments validated for your model.
python3 export_model.py --weights path/to/model.pt
```

Do not assume a prebuilt `.engine` file is portable between Jetson
devices, software versions, or GPU architectures. If engines are omitted
from Git, explain how users generate them locally.

## 8. Installation

Install the NVIDIA platform software and DeepStream using the official
instructions for your device and release. Then install
application-specific Python packages.

``` bash
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>

# If the project provides requirements.txt:
python3 -m pip install -r requirements.txt
```

DeepStream Python bindings (`pyds`) are tied to the DeepStream
installation and may not be available through ordinary pip. Follow the
installation method for your specific DeepStream release.

## 9. Configure and run

1.  Copy the example camera configuration:

    ``` bash
    cp configs/camera_config.example.yaml configs/camera_config.yaml
    ```

2.  Edit camera paths, RTSP URLs, formats, model paths, and output
    settings.

3.  Set camera credentials in environment variables or a local ignored
    file.

4.  Build or generate any required inference engines.

5.  Run the application's actual entry point.

Example command (replace with the flags supported by your application):

``` bash
python3 pipeline_app.py --config configs/camera_config.yaml
```

If the application does not support a YAML configuration or this
command-line argument, use its implemented arguments and document them
here. Do not copy this command unchanged without checking `--help`.

## 10. Outputs

Document only outputs implemented by the application.

  -----------------------------------------------------------------------
  Output                              Description
  ----------------------------------- -----------------------------------
  Live display                        Optional local display / tiled view

  RTSP                                Optional encoded stream and mount
                                      path

  Recording                           Optional video file output

  Snapshots                           Optional detection/event images

  Metadata                            Optional source ID, class,
                                      confidence, bounding box, tracker
                                      ID
  -----------------------------------------------------------------------

Example output paths:

``` text
output/
├── recordings/
├── snapshots/
└── logs/
```

Generated outputs should normally remain local and be excluded from Git.

## 11. Performance and multi-camera limitations

The maximum usable camera count depends on more than the number of
source elements:

-   USB controller bandwidth and camera format
-   RTSP network bandwidth and decoder capacity
-   Resolution and frame rate of each source
-   Muxer batch size and timeout
-   Model complexity and inference precision
-   GPU/DLA resources and memory
-   Tracker and display/encoding load
-   Thermal and power limits on embedded systems

Add cameras incrementally and benchmark. Record the tested camera count,
camera models, resolution/FPS, inference model, and measured
performance. Do not promise a universal maximum.

## 12. Troubleshooting

### Camera is not detected

-   Check `v4l2-ctl --list-devices` for V4L2 devices.
-   Inspect `/dev/v4l/by-id/` and permissions.
-   Verify the camera format and whether another process is using it.
-   For RTSP, verify the URL, credentials, network route, and stream
    codec.

### Pipeline fails to link

-   Confirm each source's output caps match the muxer requirements.
-   Check request-pad indices and source IDs.
-   Review the DeepStream/GStreamer logs for the failing element.

### One camera stalls or has a different frame rate

-   Check source-specific timeout and reconnect behavior.
-   Review muxer batch timeout and live-source settings.
-   Test each camera alone, then together.

### Low FPS or dropped frames

-   Reduce resolution or frame rate for testing.
-   Measure inference, decode, tracking, and display/encode load
    separately.
-   Check memory, GPU/DLA utilization, power, and thermal status.
-   Verify the selected model engine is actually loaded.

### Tracker IDs are confusing

-   Track IDs are generally local to a tracker instance or stream.
-   Do not treat matching IDs from different cameras as cross-camera
    identity.
-   Configure per-source tracking or implement explicit cross-camera
    association if needed.

## 13. Security and privacy

-   Never commit passwords, tokens, private keys, or RTSP credentials.
-   Use `.env` or local configuration files and add them to
    `.gitignore`.
-   Sanitize IP addresses and internal network details before
    publishing.
-   Confirm that you have permission to publish source code, model
    weights, datasets, and camera footage.
-   Review model and third-party licenses before redistribution.

## 14. Git ignore

Recommended starting point:

``` gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.env
configs/camera_config.yaml
*.engine
output/
logs/
*.log
*.bak
*.before_*
.DS_Store
Thumbs.db
```

Adjust this file to avoid ignoring required source files or
configuration examples.

## 15. Reproducibility checklist

Before calling the project reusable, verify:

-   [ ] A clean clone can follow the documented setup
-   [ ] All required dependencies and version numbers are listed
-   [ ] Camera configuration is documented
-   [ ] Each camera works individually
-   [ ] Multiple cameras run together
-   [ ] Source identity is retained in output
-   [ ] Inference engine generation or download is documented
-   [ ] Tracker behavior is explained
-   [ ] Shutdown and camera-disconnect behavior are tested
-   [ ] Measured performance and tested hardware are recorded
-   [ ] Secrets, generated files, and private data are excluded

## 16. References

-   NVIDIA DeepStream documentation:
    https://docs.nvidia.com/metropolis/deepstream/
-   NVIDIA DeepStream samples: https://github.com/NVIDIA deepstream
    reference applications (locate the sample repository for your
    installed release)
-   GStreamer documentation:
    https://gstreamer.freedesktop.org/documentation/
-   V4L2 documentation:
    https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/


