#!/usr/bin/env bash
# Create a local H3 training or VideoReward service environment.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
TORCH_BACKEND=${TORCH_BACKEND:-auto}

usage() {
    cat <<'EOF'
Usage: bash setup_uv.sh [train|reward]

  train   Install verl-omni with GPU and media extras into .venv (default).
  reward  Install the VideoReward service into .venv-video-reward.

Environment variables:
  PYTHON_VERSION     Python version for environment creation (default: 3.12).
  TORCH_BACKEND      Training PyTorch backend (default: auto; e.g. cu130).
  VIDEO_REWARD_ROOT  VideoReward checkout; required in reward mode.

Examples:
  bash setup_uv.sh
  VIDEO_REWARD_ROOT=/path/to/VideoReward bash setup_uv.sh reward

Requires uv. GPU dependencies require a compatible NVIDIA driver; the reward
environment also needs a CUDA Toolkit and C++ compiler to build FlashAttention.
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

MODE=${1:-train}
case "$MODE" in
    train)
        ENV_DIR="$REPO_ROOT/.venv"
        ;;
    reward)
        ENV_DIR="$REPO_ROOT/.venv-video-reward"
        : "${VIDEO_REWARD_ROOT:?Set VIDEO_REWARD_ROOT to your VideoReward checkout}"
        VIDEO_REWARD_ROOT=$(cd -- "$VIDEO_REWARD_ROOT" && pwd)
        if [[ ! -f "$VIDEO_REWARD_ROOT/pyproject.toml" || ! -f "$VIDEO_REWARD_ROOT/uv.lock" ]]; then
            echo "VideoReward checkout must contain pyproject.toml and uv.lock: $VIDEO_REWARD_ROOT" >&2
            exit 1
        fi
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install uv, then rerun this script." >&2
    exit 1
fi

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
    if [[ ! -f "$ENV_DIR/pyvenv.cfg" || ! -x "$ENV_DIR/bin/python" ]]; then
        echo "Existing environment is incomplete or has a broken Python link: $ENV_DIR" >&2
        echo "Move it aside before rerunning; this script does not delete existing environments." >&2
        exit 1
    fi
else
    uv venv --python "$PYTHON_VERSION" --seed "$ENV_DIR"
fi

if [[ "$MODE" == train ]]; then
    # Resolve from this project so its verl/vllm-omni dependency overrides apply.
    cd -- "$REPO_ROOT"
    uv pip install --python "$ENV_DIR/bin/python" \
        vllm==0.30.0 --torch-backend="$TORCH_BACKEND"
    uv pip install --python "$ENV_DIR/bin/python" \
        -e '.[gpu,omni]' --torch-backend="$TORCH_BACKEND"
else
    # Use VideoReward's lock and project configuration in a separate local env.
    cd -- "$VIDEO_REWARD_ROOT"
    UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync \
        --project "$VIDEO_REWARD_ROOT" --python "$PYTHON_VERSION" --frozen
    uv pip install --python "$ENV_DIR/bin/python" 'imageio[ffmpeg]>=2.37.2'
fi

printf '\nEnvironment ready. Activate it with:\n  source %q\n' "$ENV_DIR/bin/activate"
if [[ "$MODE" == reward ]]; then
    printf '\nStart the reward service with:\n  %q %q --source-root %q\n' \
        "$ENV_DIR/bin/python" \
        "$REPO_ROOT/examples/diffusionnft_trainer/minimax_h3/serve_video_edit_reward.py" \
        "$VIDEO_REWARD_ROOT"
fi
