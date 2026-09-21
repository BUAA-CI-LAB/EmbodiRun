#!/usr/bin/env bash
# Run on the AGX. The named tmux session survives SSH disconnects.
set -euo pipefail

collection_repo="${SO101_REPO:-/home/operator/orin-bench/quest-teleop-src}"
collection_python="${SO101_PYTHON:-/home/operator/orin-bench/.sdk-venv/bin/python}"
hardware_config="${SO101_HARDWARE_CONFIG:-$collection_repo/robot.orin.parallel-speed-20260915.json}"
collection_config="${SO101_COLLECTION_CONFIG:-$collection_repo/agx-leader-collection.json}"
robot_token="${SO101_ROBOT_TOKEN:-$collection_repo/robot-token}"
storage_mount="${SO101_STORAGE_MOUNT:-/mnt/robot-data}"
storage_uuid="${SO101_STORAGE_UUID:-6610-B9F6}"
collection_root="${SO101_OUTPUT_ROOT:-$storage_mount/so101-collection}"
collection_port="${SO101_ROBOT_PORT:-8766}"
collection_session="${SO101_TMUX_SESSION:-so101-collection}"
collection_script="$collection_repo/scripts/run_so101_agx_collection.sh"
held_state="${SO101_HELD_STATE:-}"

prepare_stationary_hold() {
  local candidate
  candidate="/tmp/so101-held-restart-$(date +%s%N)-$$.json"
  printf '读取从臂静止保持状态，准备安全接管（不会写电机）……\n'
  PYTHONNOUSERSITE=1 PYTHONPATH="$collection_repo/src:$collection_repo" \
    "$collection_python" "$collection_repo/scripts/probe_so101_acceleration.py" \
    --config "$hardware_config" --snapshot-output "$candidate" >/dev/null
  held_state="$candidate"
}

ensure_storage() {
  local device mounted_source mounted_uuid storage_uid storage_gid
  device="/dev/disk/by-uuid/$storage_uuid"
  storage_uid="$(id -u buaa)"
  storage_gid="$(id -g buaa)"

  if mountpoint -q "$storage_mount"; then
    mounted_source="$(findmnt -n -o SOURCE --target "$storage_mount")"
    mounted_uuid="$(blkid -s UUID -o value "$mounted_source" 2>/dev/null || true)"
    if [[ "$mounted_uuid" != "$storage_uuid" \
          && ! -e "$mounted_source" \
          && -e "$device" ]]; then
      if command -v fuser >/dev/null && ! fuser -m "$storage_mount" >/dev/null 2>&1; then
        printf '检测到失效旧挂点 %s（当前盘已重新枚举），正在按 UUID 重挂载。\n' \
          "$mounted_source"
        sudo -n umount "$storage_mount"
      else
        printf '拒绝启动：%s 是失效旧挂点且仍有进程占用，请先停止使用它的程序。\n' \
          "$storage_mount" >&2
        exit 1
      fi
    fi
  fi

  if ! mountpoint -q "$storage_mount"; then
    if [[ ! -e "$device" ]]; then
      printf '拒绝启动：未找到采集盘 UUID=%s。\n' "$storage_uuid" >&2
      exit 1
    fi
    if ! command -v mount.exfat-fuse >/dev/null; then
      echo '拒绝启动：AGX 缺少 exfat-fuse，无法挂载采集盘。' >&2
      exit 1
    fi
    sudo -n install -d -o buaa -g buaa -m 0755 "$storage_mount"
    sudo -n mount -t exfat-fuse \
      -o "uid=$storage_uid,gid=$storage_gid,umask=0022" \
      "$device" "$storage_mount"
  fi

  mounted_source="$(findmnt -n -o SOURCE --target "$storage_mount")"
  mounted_uuid="$(blkid -s UUID -o value "$mounted_source" 2>/dev/null || true)"
  if [[ "$mounted_uuid" != "$storage_uuid" ]]; then
    printf '拒绝启动：%s 当前挂载 UUID=%s，不是指定采集盘 UUID=%s。\n' \
      "$storage_mount" "${mounted_uuid:-未知}" "$storage_uuid" >&2
    exit 1
  fi
  if [[ ! -w "$storage_mount" ]]; then
    printf '拒绝启动：当前用户不能写入 %s。\n' "$storage_mount" >&2
    exit 1
  fi
}

service_loop() {
  local -a service_command
  cd "$collection_repo"
  while true; do
    if ss -ltn "sport = :$collection_port" | tail -n +2 | grep -q .; then
      printf '端口 %s 已有服务；采集器会自行校验其硬件元数据。\n' "$collection_port"
      while ss -ltn "sport = :$collection_port" | tail -n +2 | grep -q .; do sleep 2; done
      printf '原服务已退出，将启动本采集会话的硬件服务。\n'
    fi
    if [[ -z "$held_state" || ! -f "$held_state" || -e "$held_state.used" ]]; then
      until prepare_stationary_hold; do
        echo '尚不能确认从臂处于静止保持状态，2 秒后重试……' >&2
        sleep 2
      done
    fi
    printf '启动 AGX 硬件服务（失败后 2 秒重试；启动本身不会启用运动）……\n'
    service_command=(
      "$collection_python" -u -m embodirun_xlerobot_owner serve
      --mode hardware --robot-api --host 127.0.0.1 --port "$collection_port"
      --hardware-config "$hardware_config" --token-file "$robot_token"
    )
    if [[ -n "$held_state" && -f "$held_state" && ! -e "$held_state.used" ]]; then
      service_command+=(--resume-held-state "$held_state")
    fi
    set +e
    PYTHONNOUSERSITE=1 PYTHONPATH="$collection_repo/src:$collection_repo" \
      "${service_command[@]}"
    service_status=$?
    set -e
    printf '硬件服务退出（code=%s），2 秒后重试。\n' "$service_status"
    held_state=""
    sleep 2
  done
}

collector() {
  cd "$collection_repo"
  exec env PYTHONNOUSERSITE=1 PYTHONPATH="$collection_repo/src:$collection_repo" \
    "$collection_python" -u \
    scripts/collect_so101_on_agx.py \
    --config "$collection_config" --output "$collection_root"
}

case "${1:-}" in
  --service-loop) service_loop; exit 0 ;;
  --collector) collector ;;
esac

for required in "$collection_python" "$hardware_config" "$collection_config" "$robot_token"; do
  if [[ ! -e "$required" ]]; then
    printf '缺少必需文件：%s\n' "$required" >&2
    exit 1
  fi
done
if [[ ! -x "$collection_python" ]]; then
  printf 'Python 不可执行：%s\n' "$collection_python" >&2
  exit 1
fi
if ! command -v tmux >/dev/null || ! command -v ss >/dev/null; then
  echo '需要 tmux 和 ss。' >&2
  exit 1
fi
ensure_storage
available_kib="$(df -Pk "$storage_mount" | awk 'NR==2 {print $4}')"
minimum_kib=$((10 * 1024 * 1024))
if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < minimum_kib )); then
  printf '拒绝启动：%s 可用空间不足 10 GiB。\n' "$storage_mount" >&2
  exit 1
fi
mkdir -p "$collection_root"

if ! tmux has-session -t "$collection_session" 2>/dev/null; then
  if [[ -n "$held_state" ]]; then
    tmux new-session -d -s "$collection_session" -n robot \
      -e "SO101_HELD_STATE=$held_state" "$collection_script --service-loop"
  else
    tmux new-session -d -s "$collection_session" -n robot "$collection_script --service-loop"
  fi
fi
if ! tmux list-windows -t "$collection_session" -F '#{window_name}' | grep -Fxq collection; then
  tmux new-window -d -t "$collection_session" -n collection "$collection_script --collector"
fi
tmux select-window -t "$collection_session:collection"
printf '进入常驻采集会话；SSH 断开后用同一命令重新进入。\n'
exec tmux attach-session -t "$collection_session"
