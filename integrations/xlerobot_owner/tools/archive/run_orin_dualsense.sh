#!/usr/bin/env bash
# Mac launcher. Bluetooth reception/control runs on AGX, not on the Mac.
set -euo pipefail
dualsense_mode="${1:-monitor}"
case "$dualsense_mode" in
  list) dualsense_args='dualsense_input --mode list' ;;
  monitor) dualsense_args='dualsense_input --mode monitor --seconds 60' ;;
  control) dualsense_args='dualsense_drive --token-file robot-token' ;;
  takeover) dualsense_args='dualsense_drive --token-file robot-token --takeover --motion-profile tethered --output /var/tmp/xlerobot-manual-takeover' ;;
  *) echo 'Usage: bash scripts/run_orin_dualsense.sh [list|monitor|control|takeover]' >&2; exit 2 ;;
esac
exec ssh -t -o BatchMode=yes -o ConnectTimeout=8 "${DUALSENSE_HOST:-agx-orin-wifi}" \
  "cd /home/operator/orin-bench/quest-teleop-src && PYTHONNOUSERSITE=1 PYTHONPATH=. /home/operator/orin-bench/.sdk-venv/bin/python -m embodirun_xlerobot_owner.$dualsense_args"
