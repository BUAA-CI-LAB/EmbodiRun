# 传输与相机采集优化实验

这些实验在 A100、Thor 和两台 SO101 相机主机上比较权重编解码、NIXL/UCX、
相机共享内存、GStreamer 和 Zenoh。模型动作只在内存中处理，相机采集服务不提供运动接口。

## 通过共享内存传递相机帧

为 `services.rollout.camera_capture` 设置 `--shm-path`，在 `/dev/shm` 下创建由采集进程管理的文件。
同机的 `CameraObservationSource` 可通过 `shm:///dev/shm/<file>` 读取，帧 ID 和哈希与录制数据一致。
mmap 存储通过进程锁同步访问；已录制帧的槽位不再修改，最新帧使用有容量上限的槽位。
读取方只关闭自己的映射，不删除采集进程的文件。读取快照仍需复制字节，默认 JPEG 格式也仍需在观测进程中解码。
发布新数据前会校验元数据和容量，超限数据不会覆盖部分最新帧。

采集服务也支持 `--frame-format raw-bgr8`，通过
`V4L2CameraSource.capture_raw()` 并发布不可变、紧凑打包的 BGR uint8
字节，附带宽度、高度、行步长和校验和。它跳过应用层 JPEG
编码；OpenCV 仍可能解码摄像头协商的 MJPEG 格式。观测
读取方会校验布局，将 BGR 转换为 RGB，并使用与 JPEG 路径相同的
Pillow 缩放。机器人和推理接口中的 `CameraFrame` 仍使用编码图像；
本地原始帧使用独立的 `RawCameraFrame` 类型。

将原始模式与本地 `shm:///...` URL 搭配使用。HTTP 也可以为匹配的传输实验暴露相同的
原始帧，但 base64 会膨胀更大的原始载荷。
原始录制文件使用 `.bgr8`，并在
`observations.jsonl` 中保留 shape/format/hash 元数据。SHM 槽位会增长以容纳所有已配置的摄像头，
映射容量通过 `shm_allocated_bytes` 报告；请在开始录制前
检查实际 Orin 上的内存和 tmpfs 空间。

跳过 JPEG 编码会改变有损压缩后的像素。为只比较传输差异，RLinf 的
`benchmark_camera_transport.py` 使用录制好的 JPEG，在计时前解码出对应的原始像素。
独立发布进程提供 HTTP JPEG、SHM JPEG、HTTP raw 和 SHM raw 四种方式；每次读取都校验原始解码结果，
确保模型输入指纹一致。测试交替运行各方案，记录读取方的墙钟时间和 CPU 时间、发布方 CPU 增量及传输字节数，
并对成对数据块的均值进行 bootstrap 重采样。

## GStreamer 采集

采集服务支持 `--capture-backend gstreamer-cpu` 或 `gstreamer-jetson`，
配合 `--frame-format raw-bgr8` 使用。通过 `--input-format mjpeg` 或 `yuy2` 指定输入格式，设备需解析为 V4L2 路径。
两个可选参数 `--output-width` 和 `--output-height` 会将缩放移入
采集流水线。否则保留原始采集尺寸。
默认后端仍为 OpenCV/V4L2。接口不接受自定义流水线字符串；缺少插件或协商格式不受支持时会报错。

`--input-format` 也控制 OpenCV/V4L2。显式设置时，OpenCV 会在预热后回读
FOURCC、尺寸和帧率，并拒绝驱动静默
回退；`negotiated_cameras` 会记录这些值。省略该选项会保留
OpenCV 的自动选择和 GStreamer 的 MJPEG 默认值。对于 2026-09-12 检查的两个 SO101
摄像头组，腕部摄像头在 640x480 下仅支持 YUYV；
前置摄像头同时支持 MJPEG 和 YUYV。使用 `--input-format yuy2`
进行匹配的未压缩采集比较。将其与
`--frame-format raw-bgr8` 结合使用，以同时避免摄像头 MJPEG 和应用层 JPEG 编码。
YUV 到 BGR 的转换仍然发生；传输检查比较所得的 BGR 字节。

使用摄像头主机匹配的 PyGObject 和 GStreamer/GstVideo typelib，版本
为 1.20 或更新。可选的 `gstreamer` extra 声明 Python 绑定；
原生依赖由平台提供（例如 `python3-gi`、
`gir1.2-gstreamer-1.0`、`gir1.2-gst-plugins-base-1.0`，以及所需的插件
包）。带有这些包的系统 Python 或专用摄像头环境
可以将 SHM 发布到单独的 RLinf 观测环境。导入 EmbodiRun
或摄像头模块不会加载 GI；构造这个可选源时会加载。
本地校验使用 GStreamer 1.28.2 和 PyGObject 3.56.2。锁定文件也解析
PyGObject 3.58.0。在 Jetson 上，使用与已安装 JetPack
多媒体栈兼容的绑定。

CPU MJPEG 使用 `jpegdec`、`videoconvert` 和 `videoscale`。Jetson MJPEG 请求
`jpegparse`、`nvv4l2decoder mjpeg=true`，然后用 `nvvidconv` 转换到主机 BGRx，并由 CPU
`videoconvert` 转换为紧凑 BGR。YUY2 省略 JPEG 解码。请先探测实际的 Orin
插件和摄像头模式。在 Orin NX 和 AGX Orin（L4T R36.4.7、GStreamer 1.20.3、PyGObject 3.42.1）上进行的
15 秒相机采集测试均通过：两个 YUYV 相机分别测试了 CPU 和 Jetson 原尺寸转换，以及 Jetson 缩放至 224x224。
平台特定的元素参见 NVIDIA 的
[加速 GStreamer 指南](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)
一文。

每条流水线配置容量为一帧的 leaky 队列和 appsink，丢弃旧帧以减少积压。
读取方按协商格式映射样本，根据平面偏移和行步长复制 BGR 数据，并检查形状、布局、
PTS 是否递增及样本是否过期。超时或数据无效时关闭数据源，不自动重启。
底层行为见 [Appsink 队列](https://gstreamer.freedesktop.org/documentation/app/appsink.html)
和 [GStreamer 时钟](https://gstreamer.freedesktop.org/documentation/application-development/advanced/clocks.html)。
记录的是流水线呈现时间，不是相机曝光时间。实时读取方也按这个时间检查帧是否过期，
避免把刚返回的旧帧当成新帧。各相机之间没有硬件同步。

`--measure-stages` 为每次观测写入 `capture-stages.jsonl`：源调用、
调用期间的进程 CPU、数据包哈希/编码、录制 I/O、SHM 发布
以及每个摄像头的 sample map/pack 开销。对于 GStreamer，它还会启用有界的
source/appsink pad 计时，以 PTS 为键（`native_pipeline_s`），覆盖队列、
解码、转换、缩放以及探针本身。进程 CPU 包含
原生流线程。阶段日志
有开销，并且不计入其自身的工作计时器。请使用单独的诊断
运行，或者对每个被比较的候选者同等启用它。

RLinf 的 `benchmark_gstreamer_capture.py` 通过 `appsrc` 输入现有 JPEG，复用实际采集流水线的转换和输出部分。
测试比较 OpenCV 解码、GStreamer 解码，以及先用 GStreamer 缩放再发布原始 SHM 帧的方案，之后都执行常规观测转换。
流水线在帧间持续运行，测试检查各方案重复运行时像素是否一致，并报告方案间的像素差异、源文件哈希、
配对数据块的统计区间和资源使用情况。文件回放的生产端与读取端位于同一 Python 进程，底层使用原生流线程。
相机驱动、跨进程调度、推理和 GPU 上传的开销需另在硬件上测量。

## Zenoh

`services.rollout.zenoh_endpoint.ZenohEndpoint` 接受调用方提供的载荷
编码器和解码器。安装 `embodirun[zenoh]` 以获得固定版本的 Zenoh 1.10.1
运行时。RLinf 的比较还安装了 `wireless` extra，并使用
`SegmentedCodec` 来保留现有的 dataclass/tensor 表示。
展平和重建数据段会增加内存复制，计入 RTT 测量。

Zenoh 需手动配置 TCP 监听地址，可用 `connect_to` 指定对端；自动发现和本机 SHM 均关闭。
已配置连接会在后台重试。可设置 QoS 优先级、拥塞策略、临时恢复缓存深度、心跳周期和各路由的接收上限。
每个对端使用独立执行器发送数据，并记录存活状态和消息丢失事件。

应用层始终执行 `StreamSession` 检查，队列溢出或序列断档时需重置轨迹。
本地发布完成不代表远端已处理消息；底层 put 也可能在调用方超时后才完成，此时路由判为失败。
历史记录默认关闭。读取端重启后若主动回放缓存，可能再次收到已处理的消息，这些消息会被流校验器拒绝。

本地测试发现了 Zenoh 自动选择的 SHM
路径中的映射失败。局域网实验使用 TCP。GStreamer 在本地成功解码了现有摄像头
JPEG，但其缩放后的像素与 Pillow 不同。Jetson 多媒体
性能和预处理等价性需要单独的硬件测试。

RLinf 的运行脚本和复现配置见其仓库中的
`RLinf/examples/embodiment/TRANSPORT_OPTIMIZATION.md`。
这些方案需手动启用，目前尚未在 A100/Thor/Orin 上测得加速或恢复能力提升。

## NIXL/UCX 张量传输

`services.rollout.nixl_tensors.NixlTensorTransport` 通过 NIXL/UCX 增加了一条通用的 CPU/CUDA
张量传输方式。在兼容的独立环境中安装 `nixl` 可选依赖（1.4.1）。
Torch/CUDA/UCX 和 Jetson wheel 兼容性
必须在实际主机上检查。本地 GPU 测试使用 TCP 和 CUDA 复制。

调用方提供有序的异步元数据发送/接收函数，并
在半双工连接上串行化操作。`connect(...,
initiator=True)` 先发送自己的 hello；另一侧使用 `initiator=False`。
张量字节使用已注册的有界 arena 和 NIXL WRITE；元数据和 credit
使用调用方的控制路径。在原生完成通知之后，
接收方复制到自己拥有的张量，等待 CUDA 复制完成，并在复用前
确认该 chunk。先前返回的张量保持独立于
arena。描述符校验会限制张量数量和总载荷大小。

超时、取消、无效序列或原生错误都会使连接失败。
发生故障后，内存池（arena）、注册信息和句柄保留至进程退出，因为未完成的远程 WRITE 仍可能写入这些地址。
`close()` 只注销正常且空闲的连接。调用方需重启故障 worker 并重新同步；
连接不会自动重建，也不会自动切换设备或传输方式。发送成功表示字节交接完成，不表示权重已应用或动作已执行。

RLinf 的可选适配器使用其现有的 Worker 控制路径，保留
key/shape/dtype 协商，并通过该传输发送初始检查点 bucket 和后续
patch。当前拓扑是一个 actor rank 和一个
rollout rank，并可在支持范围内接入多台相机设备。本地测试覆盖
CPU/GPU chunk 边界、标量/空/非连续张量、字节所有权、
反向方向、外来序列和扣留 credit。实际的 Ray Worker
校验还会检查完整初始化、更新、空 patch、
版本提交，以及 credit 停顿后 rollout 权重/版本保持不变。
