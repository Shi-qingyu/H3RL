#!/usr/bin/env bash
# Pass RUN_DIR to compare completed checkpoints with a dedicated reward GPU,
# or RUN_DIR/checkpoints/global_step_N to score one checkpoint via VIDEO_REWARD_URL.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
cd -- "$repo_root"
source "$script_dir/video_edit_v6.env"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-10800}
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=${VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT:-3600}

exec "${H3_PYTHON:-$repo_root/.venv/bin/python}" \
    -m examples.diffusionnft_trainer.minimax_h3.validate_video_edit_checkpoint \
    --model-path "$MODEL_PATH" --template "$LORA_WARMSTART_PATH" --data-file "$DATA_DIR/test.parquet" \
    --reward-url "${VIDEO_REWARD_URL:-http://127.0.0.1:19085/score}" \
    --attention-backend "${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}" \
    --reference-image-short-edge "${VAL_REF_IMAGE_SHORT_EDGE:-${REF_IMAGE_SHORT_EDGE:-2048}}" "$@"
