#!/usr/bin/env bash
set -euo pipefail
example_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
example_python="${EMBODIRUN_EXAMPLE_PYTHON:-$example_root/.venv/bin/python}"
if [[ ! -x "$example_python" ]]; then
  echo 'Install the Host environment with uv sync --frozen, or set EMBODIRUN_EXAMPLE_PYTHON.' >&2
  exit 2
fi
export PYTHONPATH="$example_root/src:$example_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$example_python" "$example_root/examples/runner.py" "$@"
