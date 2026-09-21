#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname -- "$SCRIPT_DIR")"
readonly DEFAULT_ENV_ROOT="${HOME}/envs"
readonly UV_VERSION="${UV_VERSION:-0.12.7}"

venv_path=""
python_version=""
pytorch="skip"
pytorch_index="${EMBODIRUN_PYTORCH_INDEX:-${RLINF_PYTORCH_INDEX:-auto}}"
recreate=false
dry_run=false
declare -a extra_deps=()
declare -a requirement_files=()

usage() {
    cat <<'EOF'
Usage:
  bash requirements/install.sh --venv PATH --python VERSION [OPTIONS]

Options:
  --extra-deps NAMES    Comma-separated components; repeatable.
                        Available: so101, fr3, opencv, realsense, test
  --requirement PATH    Additional requirement file; repeatable.
  --pytorch VALUE       auto, skip, or an exact Torch version. Default: skip
  --pytorch-index VALUE auto, default, or a package-index URL. Default: auto
  --recreate            Remove and recreate an existing virtual environment.
  --dry-run             Print the installation plan without changing anything.
  -h, --help            Show this help.

Storage defaults may be overridden with UV_CACHE_DIR,
UV_PYTHON_INSTALL_DIR, and UV_BIN_DIR. They default under $HOME/envs.
EMBODIRUN_PYTORCH_INDEX (legacy RLINF_PYTORCH_INDEX) sets the default
for --pytorch-index, and EMBODIRUN_ENV_ROOT (legacy RLINF_ENV_ROOT) moves
the storage root.
EOF
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

append_extra_deps() {
    local value="$1"
    local -a names=()
    IFS=',' read -r -a names <<<"$value"
    local name
    for name in "${names[@]}"; do
        [[ -n "$name" ]] || fail "--extra-deps contains an empty component"
        case "$name" in
            so101|fr3|opencv|realsense|test) ;;
            *) fail "unknown --extra-deps component '$name'" ;;
        esac
        extra_deps+=("$name")
    done
}

while (($#)); do
    case "$1" in
        --venv)
            (($# >= 2)) || fail "--venv requires a path"
            venv_path="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || fail "--python requires a version"
            python_version="$2"
            shift 2
            ;;
        --extra-deps)
            (($# >= 2)) || fail "--extra-deps requires component names"
            append_extra_deps "$2"
            shift 2
            ;;
        --requirement)
            (($# >= 2)) || fail "--requirement requires a path"
            requirement_files+=("$2")
            shift 2
            ;;
        --pytorch)
            (($# >= 2)) || fail "--pytorch requires auto, skip, or a version"
            pytorch="$2"
            shift 2
            ;;
        --pytorch-index)
            (($# >= 2)) || fail "--pytorch-index requires auto, default, or a URL"
            pytorch_index="$2"
            shift 2
            ;;
        --recreate)
            recreate=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ -n "$venv_path" ]] || fail "--venv is required"
[[ -n "$python_version" ]] || fail "--python is required"
[[ "$venv_path" == /* ]] || fail "--venv must be an absolute path"

case "$pytorch" in
    auto|skip) ;;
    *[!0-9.]*) fail "--pytorch must be auto, skip, or a numeric version" ;;
    '') fail "--pytorch must not be empty" ;;
esac

case "$pytorch_index" in
    auto|default|http://*|https://*) ;;
    *) fail "--pytorch-index must be auto, default, or an HTTP(S) URL" ;;
esac

for requirement in "${requirement_files[@]}"; do
    [[ -f "$requirement" ]] || fail "requirement file not found: $requirement"
done

readonly ENV_ROOT="${EMBODIRUN_ENV_ROOT:-${RLINF_ENV_ROOT:-$DEFAULT_ENV_ROOT}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ENV_ROOT/uvcache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ENV_ROOT/uvpython}"
readonly UV_BIN_DIR="${UV_BIN_DIR:-$ENV_ROOT/bin}"
readonly MANAGED_UV="$UV_BIN_DIR/uv"

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_command "$@"
    if [[ "$dry_run" == false ]]; then
        "$@"
    fi
}

ensure_safe_recreate_path() {
    case "$venv_path" in
        /|"$HOME"|"$ENV_ROOT"|"$ENV_ROOT/venv")
            fail "refusing to recreate broad path: $venv_path"
            ;;
    esac
}

install_uv() {
    if [[ -x "$MANAGED_UV" ]]; then
        return
    fi
    if [[ "$dry_run" == true ]]; then
        printf '+ install uv %s into %q\n' "$UV_VERSION" "$UV_BIN_DIR"
        return
    fi
    mkdir -p -- "$UV_BIN_DIR"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" |
        env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
}

write_uv_environment() {
    local env_file="$ENV_ROOT/uv.env"
    if [[ "$dry_run" == true ]]; then
        printf '+ write uv environment to %q\n' "$env_file"
        return
    fi
    mkdir -p -- "$ENV_ROOT" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
    {
        printf 'export PATH=%q:$PATH\n' "$UV_BIN_DIR"
        printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
        printf 'export UV_PYTHON_INSTALL_DIR=%q\n' "$UV_PYTHON_INSTALL_DIR"
    } >"$env_file"
}

installed_cuda_version() {
    local version
    version="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
    if [[ -z "$version" && -e /usr/local/cuda ]]; then
        version="$(readlink -f /usr/local/cuda | sed -n 's/.*cuda-\([0-9][0-9.]*\)$/\1/p')"
    fi
    printf '%s\n' "$version"
}

detect_pytorch_platform() {
    if [[ "$(uname -m)" != "aarch64" || ! -f /etc/nv_tegra_release ]]; then
        printf '%s\n' generic
        return
    fi
    local l4t_release
    l4t_release="$(sed -n 's/^# R\([0-9][0-9]*\).*/\1/p' /etc/nv_tegra_release | head -n 1)"
    local cuda_version
    cuda_version="$(installed_cuda_version)"
    case "$l4t_release:$cuda_version" in
        36:12.6*) printf '%s\n' orin-cu126 ;;
        39:13.2*) printf '%s\n' thor-cu132 ;;
        *)
            fail "unsupported NVIDIA platform: L4T R${l4t_release:-unknown}, CUDA ${cuda_version:-unknown}"
            ;;
    esac
}

install_pytorch() {
    if [[ "$pytorch" == "skip" ]]; then
        return 0
    fi
    local -a packages=()
    if [[ "$pytorch" == "auto" ]]; then
        packages=(-r "$PROJECT_ROOT/requirements/pytorch.txt")
    else
        packages=("torch==$pytorch" 'torchvision>=0.22,<0.27')
    fi

    local resolved_platform=generic
    local resolved_index=""
    if [[ "$pytorch_index" == auto ]]; then
        resolved_platform="$(detect_pytorch_platform)"
        case "$resolved_platform" in
            orin-cu126) resolved_index='https://pypi.jetson-ai-lab.io/jp6/cu126' ;;
            thor-cu132) resolved_index='https://pypi.jetson-ai-lab.io/sbsa/cu132' ;;
        esac
    elif [[ "$pytorch_index" != default ]]; then
        resolved_index="$pytorch_index"
    fi

    if [[ "$resolved_platform" == orin-cu126 ]]; then
        case "$python_version" in
            3.10|3.10.*) ;;
            *) fail "orin-cu126 PyTorch wheels require Python 3.10" ;;
        esac
    fi

    if [[ -n "$resolved_index" ]]; then
        run "$MANAGED_UV" pip install \
            --python "$venv_path/bin/python" \
            --index-url "$resolved_index" \
            "${packages[@]}"
    else
        run "$MANAGED_UV" pip install \
            --python "$venv_path/bin/python" \
            --torch-backend auto \
            "${packages[@]}"
    fi
}

project_requirement() {
    if ((${#extra_deps[@]} == 0)); then
        printf '%s\n' "$PROJECT_ROOT"
        return
    fi
    local IFS=,
    printf '%s[%s]\n' "$PROJECT_ROOT" "${extra_deps[*]}"
}

printf 'EmbodiRun environment plan\n'
printf '  project: %s\n' "$PROJECT_ROOT"
printf '  venv: %s\n' "$venv_path"
printf '  python: %s\n' "$python_version"
printf '  uv cache: %s\n' "$UV_CACHE_DIR"
printf '  uv python: %s\n' "$UV_PYTHON_INSTALL_DIR"
printf '  pytorch: %s\n' "$pytorch"
printf '  pytorch index: %s\n' "$pytorch_index"
printf '  extra deps: %s\n' "${extra_deps[*]:-none}"

install_uv
write_uv_environment

if [[ -e "$venv_path" ]]; then
    [[ "$recreate" == true ]] || fail "venv already exists; pass --recreate: $venv_path"
    ensure_safe_recreate_path
    run rm -rf -- "$venv_path"
fi

run mkdir -p -- "$(dirname -- "$venv_path")" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
run "$MANAGED_UV" python install "$python_version"
run "$MANAGED_UV" venv --python "$python_version" "$venv_path"

install_pytorch

for requirement in "${requirement_files[@]}"; do
    run "$MANAGED_UV" pip install \
        --python "$venv_path/bin/python" \
        --requirement "$requirement"
done

run "$MANAGED_UV" pip install \
    --python "$venv_path/bin/python" \
    "$(project_requirement)"

if [[ "$dry_run" == false ]]; then
    "$venv_path/bin/python" -c 'import embodirun; print("embodirun import: ok")'
    if [[ "$pytorch" != "skip" ]]; then
        "$venv_path/bin/python" - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"torch CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("installed PyTorch cannot access CUDA")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
    fi
fi

printf 'Environment ready: source %q\n' "$venv_path/bin/activate"
