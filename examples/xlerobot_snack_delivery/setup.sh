#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${XLR_SNACK_VENV:-$repo_root/.venv-xlerobot-snack}"
install_owner=1

if [[ "${1:-}" == "--minimal" ]]; then
  install_owner=0
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--minimal]" >&2
  exit 2
fi

if [[ ! -d "$venv_dir" ]]; then
  python3 -m venv "$venv_dir"
fi
recipe_python="$venv_dir/bin/python"
if [[ ! -x "$recipe_python" ]]; then
  echo "virtual environment Python is missing: $recipe_python" >&2
  exit 1
fi

bootstrap_dir="$(mktemp -d "${TMPDIR:-/tmp}/xlerobot-setup.XXXXXX")"
trap 'rm -rf "$bootstrap_dir"' EXIT
python3 -m venv "$bootstrap_dir"
"$bootstrap_dir/bin/python" -m pip install 'uv>=0.12,<0.13'
setup_uv="$bootstrap_dir/bin/uv"
UV_PROJECT_ENVIRONMENT="$venv_dir" "$setup_uv" sync \
  --project "$repo_root" --frozen --no-default-groups --group host
if [[ "$install_owner" -eq 1 ]]; then
  "$setup_uv" pip install --python "$recipe_python" \
    -e "$repo_root/integrations/xlerobot_owner[hardware,camera,recording,web]"
fi

echo "Recipe environment ready: $recipe_python"
if [[ "$install_owner" -eq 1 ]]; then
  echo "Owner extras installed. Use the recipe services command to start one owner and two Control services."
else
  echo "Minimal mode installed; use --mode plan or --mode dry-run only."
fi
