#!/usr/bin/env bash
# Browser gateway only. Does not restart, configure, or arm the AGX robot.
set -euo pipefail
drive_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$drive_repo"
drive_python="$drive_repo/.venv/bin/python"
drive_ssh_host="${DRIVE_SSH_HOST:-agx-orin-123-4}"
drive_forward_port="${DRIVE_FORWARD_PORT:-18767}"
drive_page_port="${DRIVE_PAGE_PORT:-8844}"
drive_leader_command="${DRIVE_LEADER_COMMAND:-/path/to/so101-leader-calibration/start-dual-so101-direct-follow.command}"
drive_ssh_pid=""
if nc -z 127.0.0.1 "$drive_page_port" 2>/dev/null; then
  if curl --noproxy '*' --fail --silent --max-time 3 "http://127.0.0.1:${drive_page_port}/api/status" |
    "$drive_python" -c 'import json,sys; s=json.load(sys.stdin); leader=s.get("leader_follow", {}); sys.exit(0 if s.get("mode")=="remote" and s.get("control_scope")=="base" and leader.get("available") is True else 1)'; then
    printf 'Keyboard gateway is already running: http://127.0.0.1:%s/drive\n' "$drive_page_port"
    exit 0
  fi
  printf 'Port %s is occupied by another service; no process was stopped.\n' "$drive_page_port" >&2
  exit 1
fi
if [[ ! -x "$drive_leader_command" ]]; then
  printf 'Dual-leader launcher is missing or not executable: %s\n' "$drive_leader_command" >&2
  exit 1
fi
if nc -z 127.0.0.1 "$drive_forward_port" 2>/dev/null; then
  printf 'Tunnel port %s is occupied; use DRIVE_FORWARD_PORT with an unused port.\n' "$drive_forward_port" >&2
  exit 1
fi
cleanup() {
  if [[ -n "$drive_ssh_pid" ]]; then
    kill "$drive_ssh_pid" 2>/dev/null || true
    wait "$drive_ssh_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
ssh -n -N -T -o ControlMaster=no -o ControlPath=none -o BatchMode=yes \
  -o ExitOnForwardFailure=yes -o ConnectTimeout=8 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
  -L "127.0.0.1:${drive_forward_port}:127.0.0.1:8766" "$drive_ssh_host" &
drive_ssh_pid=$!
for ((attempt=0; attempt<40; attempt++)); do
  kill -0 "$drive_ssh_pid" 2>/dev/null || { echo 'SSH tunnel failed' >&2; exit 1; }
  if nc -z 127.0.0.1 "$drive_forward_port" 2>/dev/null; then break; fi
  sleep 0.25
done
nc -z 127.0.0.1 "$drive_forward_port" 2>/dev/null || { echo 'SSH tunnel unavailable' >&2; exit 1; }
printf 'Keyboard driving page: http://127.0.0.1:%s/drive\n' "$drive_page_port"
PYTHONPATH=src "$drive_python" -u -m embodirun_xlerobot_owner serve \
  --mode remote --host 127.0.0.1 --port "$drive_page_port" \
  --browser-no-token-cidr 127.0.0.0/8 \
  --token-file artifacts/quest/credentials/browser-token \
  --robot-url "http://127.0.0.1:${drive_forward_port}" \
  --robot-scope base \
  --robot-token-file artifacts/quest/credentials/robot-token \
  --leader-command "$drive_leader_command" \
  --mapping-config examples/quest-teleop/keyboard-mapping.json \
  --output artifacts/quest/keyboard-episodes --fps 20
