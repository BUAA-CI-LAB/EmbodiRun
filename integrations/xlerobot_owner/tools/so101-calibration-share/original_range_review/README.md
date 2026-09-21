# 之前使用的原脚本：双臂活动范围复核

这三个 Python 文件从原工作区直接复制，保留原来的中文交互流程：

- `review_xlerobot_calibration.py`：逐关节人工活动范围复核向导。
- `record_xlerobot_joint_range.py`：读取并记录单关节原始位置。
- `inspect_xlerobot_motors.py`：SDK 导入和总线只读检查。

**它们用于已有双臂标定的 XLeRobot，不是从零完成单臂标定的入口。** 需要带 `left_arm_*` / `right_arm_*` 键的旧标定 JSON，以及左、右两个不同的串口。预检要求电机力矩已关闭、当前设备参数与旧标定一致。程序记录人工移动时的位置，不主动驱动机械臂，也不会把读数自动写回成新的零位或限位。

示例在 USB 所在机器执行，路径全部替换成自己的：

```bash
/absolute/venv/bin/python review_xlerobot_calibration.py \
  --sdk-src /absolute/lerobot/src \
  --calibration /absolute/dual-arm-calibration.json \
  --port /dev/serial/by-id/LEFT_ARM \
  --port /dev/serial/by-id/RIGHT_ARM \
  --output-root /absolute/range-review-results
```

需要交互终端。按 Enter 进行预检，看到 `READY` 后再活动当前关节；Enter 结束当前关节，`s` 跳过，`q` 保存并退出。输出为每关节的 `joint_range.json` 和会话汇总 `session_summary.json`；它们是观察记录，不能直接作为标准 SO101 标定 JSON 使用。

我们原来 Mac 上的 SSH 启动文件写死了自己的 AGX 地址、密钥和双臂路径，所以没有放入本包。对四只独立 SO101 做完整标定，请使用上一级的 `calibrate.command`。
