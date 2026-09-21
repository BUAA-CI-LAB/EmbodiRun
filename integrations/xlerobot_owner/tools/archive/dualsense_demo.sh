#!/usr/bin/env bash
# Run on AGX. Reuse its robot API; never restart it or automatically arm.
set -euo pipefail
demo_repo=/home/operator/orin-bench/quest-teleop-src
demo_python=/home/operator/orin-bench/.sdk-venv/bin/python
demo_disk="${DEMO_DISK:-}"
if [[ -z "$demo_disk" ]]; then
  demo_disk=$(findmnt -nro TARGET -S UUID=6610-B9F6 || true)
fi
if [[ -z "$demo_disk" || "$demo_disk" == *$'\n'* ]] || ! mountpoint -q "$demo_disk"; then
  echo '找不到唯一已挂载的移动硬盘 111 (6610-B9F6)；不向系统盘回退。' >&2
  exit 1
fi
if [[ "$(findmnt -nro UUID --target "$demo_disk")" != "6610-B9F6" ]]; then
  echo '录制磁盘不是已确认的移动硬盘 111 (6610-B9F6)。' >&2
  exit 1
fi
mkdir -p "$demo_disk/xlerobot-demos"
exec 9>/run/lock/xlerobot-dualsense-demo.lock
flock -n 9 || { echo '手柄录制已在运行，请勿重复启动。' >&2; exit 1; }
demo_run=$(mktemp -d "$demo_disk/xlerobot-demos/demo-$(date +%Y%m%d-%H%M%S)-XXXXXX")
cd "$demo_repo"
export PYTHONNOUSERSITE=1 PYTHONPATH=.
echo "本段保存位置：$demo_run"
echo '摇杆归中、松开所有键后按住 R1 操作；×/○ 结束并保存。最多录10分钟。'
demo_exit=0
"$demo_python" -u -m embodirun_xlerobot_owner.dualsense_drive --token-file robot-token \
  --output "$demo_run" --max-seconds 600 || demo_exit=$?
echo '控制程序已退出；生成三路 MP4 预览，原始JPEG和JSONL同时保留。'
"$demo_python" -u -m embodirun_xlerobot_owner.dualsense_recording "$demo_run" || \
  echo 'MP4生成失败：原始JPEG和JSONL仍保留，请检查上方错误。' >&2
sync -f "$demo_run"
echo "保存目录：$demo_run"
exit "$demo_exit"
