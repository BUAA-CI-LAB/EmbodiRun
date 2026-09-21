#!/usr/bin/env bash
# Install EmbodiRun from this source checkout with uv.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [options]

Options:
  --group GROUP   capability group to install (default: host)
                  host | robot-so101 | robot-fr3 | robot-go2 | robot-arx5 |
                  robot-xlerobot-external-owner | sim-libero | sim-habitat |
                  sim-isaac | sim-vlabench
  --extra EXTRA   optional extra to install (sglang | wireless)
  --dev           include the development tools group
  -h, --help      show this help

Examples:
  scripts/install.sh
  scripts/install.sh --group robot-so101
  scripts/install.sh --group host --extra sglang
EOF
}

group="host"
extra=""
dev=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --group)
      group="${2:?--group needs a value}"
      shift 2
      ;;
    --extra)
      extra="${2:?--extra needs a value}"
      shift 2
      ;;
    --dev)
      dev=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv 0.12.x is required." >&2
  echo "Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

args=(sync --frozen)
if [ "$dev" -eq 0 ]; then
  args+=(--no-dev)
fi
args+=(--group "$group")
if [ -n "$extra" ]; then
  args+=(--extra "$extra")
fi

echo "+ uv ${args[*]}"
uv "${args[@]}"

echo
echo "Installed with group '$group'${extra:+, extra '$extra'}."
echo "Try: uv run embodirun --help"
