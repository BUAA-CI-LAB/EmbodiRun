#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$repo_root/examples/setup_environment.py" snack \
  --environment "${XLR_SNACK_VENV:-$repo_root/.venv-xlerobot-snack}" "$@"
