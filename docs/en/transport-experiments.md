# Optional transport experiments

These experiments compare weight codecs, NIXL/UCX, local camera shared
memory, GStreamer and Zenoh using A100, Thor and two SO101 camera hosts. Actions
terminate in memory; the camera-only publisher exposes no motion API.

## Shared-memory camera frames

`services.rollout.camera_capture` accepts `--shm-path` pointing to a new owned
file under `/dev/shm`. A same-host `CameraObservationSource` can then read
`shm:///dev/shm/<file>` using identical recorded frame IDs and hashes. The mmap
store uses a process lock, immutable recorded slots and a bounded latest slot.
Readers close their mappings but never unlink the owner file. Snapshot reads
still copy bytes. The default JPEG path retains JPEG decode in the observation
process. Metadata and payload capacities are validated before publishing a new
generation; an oversized status cannot partially replace the latest frame.

The camera-only publisher also accepts `--frame-format raw-bgr8`. This calls
`V4L2CameraSource.capture_raw()` and publishes immutable, tightly packed BGR uint8
bytes with width, height, row stride and checksum. It skips application JPEG
encoding; OpenCV may still decode the camera's negotiated MJPEG format. The
observation reader validates the layout, converts BGR to RGB, and uses the same
Pillow resize as the JPEG path. Existing robot/inference `CameraFrame` contracts
remain encoded-image contracts; raw local frames use a separate `RawCameraFrame`.

Use raw mode with a local `shm:///...` URL. HTTP can also expose the same raw
frames for a matched transport experiment, but base64 expands the larger raw
payload. Raw recording files use `.bgr8` and retain shape/format/hash metadata in
`observations.jsonl`. SHM slots grow to fit all configured cameras and mapped
capacity is reported as `shm_allocated_bytes`; check memory and tmpfs space on the
actual Orin before starting a recording.

Bypassing a fresh JPEG encoding can alter pixels relative to the old lossy
pipeline. For transport isolation, RLinf's `benchmark_camera_transport.py` uses
previously recorded JPEG files and prepares equivalent raw pixels once before
timing. A separate publisher process serves HTTP JPEG, SHM JPEG, HTTP raw and
SHM raw. All model-input fingerprints must match, and every fetched recording
is checked against its initial decoded state. The benchmark records consumer
wall/CPU time, publisher CPU deltas and image payload bytes, alternates order,
and bootstraps paired block means.

## GStreamer capture

The camera-only publisher additionally accepts `--capture-backend
gstreamer-cpu` or `gstreamer-jetson`, with `--frame-format raw-bgr8`. Input format
is explicit (`--input-format mjpeg` or `yuy2`), using resolved V4L2 devices only.
Both optional `--output-width` and `--output-height` move resizing into the
capture pipeline. Otherwise the original capture dimensions are retained.
OpenCV/V4L2 remains the default. There is no arbitrary pipeline argument and
missing plugins or unsupported negotiated formats cause an error.

`--input-format` also controls OpenCV/V4L2. When explicitly set, OpenCV reads
back FOURCC, dimensions and frame rate after warmup and rejects a silent driver
fallback; `negotiated_cameras` records these values. Omitting the option preserves
OpenCV's automatic selection and GStreamer's MJPEG default. For the two SO101
camera sets inspected on 2026-09-12, the wrist camera supports only YUYV at
640x480; the front camera supports both MJPEG and YUYV. Use `--input-format yuy2`
for a matched uncompressed acquisition comparison. Combine it with
`--frame-format raw-bgr8` to avoid both camera MJPEG and application JPEG encoding.
YUV-to-BGR conversion still occurs; transport checks compare the resulting BGR bytes.

Use the camera host's matching PyGObject and GStreamer/GstVideo typelibs, version
1.20 or newer. The optional `gstreamer` extra declares the Python binding;
native dependencies are supplied by the platform (for example `python3-gi`,
`gir1.2-gstreamer-1.0`, `gir1.2-gst-plugins-base-1.0`, and the required plugin
packages). A system Python or dedicated camera environment with those packages
can publish SHM to the separate RLinf observation environment. Importing EmbodiRun
or the camera module does not load GI; construction of this optional source does.
Local validation used GStreamer 1.28.2 and PyGObject 3.56.2. The lock also resolves
PyGObject 3.58.0. On Jetson, use bindings compatible with the installed JetPack
multimedia stack.

CPU MJPEG uses `jpegdec`, `videoconvert` and `videoscale`. Jetson MJPEG requests
`jpegparse`, `nvv4l2decoder mjpeg=true`, then `nvvidconv` to host BGRx and CPU
`videoconvert` to packed BGR. YUY2 omits JPEG decoding. Probe the actual Orin
plugins and camera modes first. Bounded 15-second camera-only pilots on both
Orin NX and AGX Orin (L4T R36.4.7, GStreamer 1.20.3, PyGObject 3.42.1) passed
with two YUYV cameras through CPU and Jetson conversion, at original dimensions
and Jetson resize to 224x224.
See NVIDIA's [accelerated GStreamer guide](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)
for the platform-specific elements.

Each pipeline has a one-buffer leaky queue and a one-buffer appsink that drops
old frames. The reader maps the negotiated sample and honours plane offsets and
row strides before copying owned BGR bytes. It checks shape, layout, increasing
PTS and maximum sample age. Timeout or invalid data closes the source; no
implicit restart occurs. [Appsink queue behaviour](https://gstreamer.freedesktop.org/documentation/app/appsink.html)
and [GStreamer clocks](https://gstreamer.freedesktop.org/documentation/application-development/advanced/clocks.html)
define the underlying semantics. The recorded time is pipeline presentation
time, not camera exposure time. Local live observation readers also check this
age, so a just-returned old sample cannot appear fresh merely because the
capture call finished recently. Separate cameras are not hardware synchronized.

`--measure-stages` writes per-observation `capture-stages.jsonl`: source call,
process CPU during the call, packet hash/encoding, recording I/O, SHM publication
and per-camera sample map/pack costs. For GStreamer it also enables bounded
source/appsink pad timing keyed by PTS (`native_pipeline_s`), covering queue,
decode, conversion, resize and the probes themselves. Process CPU includes
native streaming threads. Stage logging
has overhead and is excluded from its own work timer. Use a separate diagnostic
run, or enable it equally for every compared candidate.

RLinf's `benchmark_gstreamer_capture.py` reuses the production conversion/sink
tail with `appsrc` and existing JPEG files. It compares OpenCV raw decode,
GStreamer raw decode, and GStreamer resize before raw SHM publication and normal
observation conversion. Pipelines persist across frames. It verifies repeatable
pixels within each profile and reports pixel differences across profiles, source
hashes, paired block intervals and resource snapshots. The file replay runs
producer and observer in one Python process with native streaming threads;
camera-driver costs, process scheduling, inference and GPU upload require
separate hardware measurements.

## Zenoh

`services.rollout.zenoh_endpoint.ZenohEndpoint` accepts caller-provided payload
encoders and decoders. Install `embodirun[zenoh]` for the pinned Zenoh 1.10.1
runtime. RLinf's comparison also installs the `wireless` extra and uses
`SegmentedCodec` to preserve the existing dataclass/tensor representation.
Flattening and rebuilding segments add copies that must be included in RTT.

Zenoh uses explicit TCP listeners and optional `connect_to` peer IDs, disables
discovery and automatic local SHM, and retries configured connections in the
background. QoS priority and congestion policy, volatile recovery cache depth,
heartbeat period and per-route receive limits are configurable. Sending uses a
separate executor per peer. Liveliness and missed-message events are recorded.

Application-level `StreamSession` checks remain mandatory. A queue overflow or
sequence gap requires trajectory reset. Local publish completion does not mean
the remote application consumed the message. Native puts can finish after a
caller timeout; such a route is failed. History is off by default; explicitly
replaying cache history after consumer restart may return already consumed
messages, which the stream validator rejects.

Local testing uncovered a mapping failure in Zenoh's automatically selected SHM
path. LAN experiments use TCP. GStreamer successfully decoded existing camera
JPEGs locally, but its scaled pixels differed from Pillow. Jetson multimedia
performance and preprocessing equivalence require separate hardware tests.

The reproducible RLinf runner and configuration live outside this repository:
`RLinf/examples/embodiment/TRANSPORT_OPTIMIZATION.md`. These opt-in candidates
have not yet demonstrated a speedup or better recovery on A100/Thor/Orin.

## NIXL/UCX tensor transport

`services.rollout.nixl_tensors.NixlTensorTransport` adds a generic CPU/CUDA
tensor path through NIXL/UCX. Install the optional `nixl` extra (1.4.1) in a
compatible isolated environment. Torch/CUDA/UCX and Jetson wheel compatibility
must be checked on the actual hosts. The local GPU test uses TCP and CUDA copying.

Callers supply ordered asynchronous metadata send/receive functions and
serialize operations on the half-duplex connection. `connect(...,
initiator=True)` sends its hello first; the other side uses `initiator=False`.
Tensor bytes use a registered bounded arena and NIXL WRITE; metadata and credits
use the caller's control path. After a native completion notification the
receiver copies into owned tensors, waits for CUDA copies, and acknowledges
that chunk before reuse. Previously returned tensors remain independent of the
arena. Descriptor validation bounds tensor count and total payload size.

Timeout, cancellation, invalid sequence or native error fails the connection.
Failed arenas/registrations/handles remain retained until process exit because
an outstanding remote WRITE can still target them. `close()` only deregisters
a healthy idle connection. Callers must restart failed workers and resynchronize;
there is no automatic reconnect or device/transport fallback. A successful send
confirms byte ownership, not application of weights or execution of an action.

RLinf's optional adapter uses its existing Worker control path, preserves
key/shape/dtype negotiation, and sends initial checkpoint buckets and later
patches through this transport. The current topology is one actor rank and one
rollout rank with any supported number of camera bodies. Local tests cover
CPU/GPU chunk boundaries, scalar/empty/noncontiguous tensors, byte ownership,
reverse direction, foreign sequences and withheld credits. Actual Ray Worker
verification additionally checks full initialization, updates, empty patches,
version commit and unchanged rollout weights/version after a credit stall.
