#!/usr/bin/env bash
# Preview the RL starting policy: official H3 plus the prepared SFT v6 adapter.
# Uses the repository .venv; H3_PYTHON can select another environment's Python.
# NUM_SAMPLES=0 selects the full split.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd -- "$REPO_ROOT"
source "$SCRIPT_DIR/video_edit_v6.env"

H3_PYTHON=${H3_PYTHON:-$REPO_ROOT/.venv/bin/python}
if [[ ! -x "$H3_PYTHON" ]]; then
    echo "H3 Python is unavailable: $H3_PYTHON" >&2
    echo "Set H3_PYTHON to an executable Python path in your installed H3 training environment." >&2
    echo "If using the repository .venv, its Python interpreter must also exist on this node." >&2
    exit 1
fi

DATA_FILE=${DATA_FILE:-$DATA_DIR/train.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$V6_OUTPUT_DIR/preview_sft_$(date +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-8}
START_INDEX=${START_INDEX:-0}
INFER_STEPS=${INFER_STEPS:-40}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_TP}
REF_IMAGE_SHORT_EDGE=${REF_IMAGE_SHORT_EDGE:-2048}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-FLASH_ATTN_3_HUB}
# One request includes the full video forward, including first-use compilation.
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-3600}
# Output transfer can wait behind GPU work that is still in flight.
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=${VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT:-$VLLM_OMNI_DLO_DP_WAVE_TIMEOUT}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

printf 'H3 Python: %s\n' "$H3_PYTHON"
printf 'H3 execution RPC timeout: %s s; async output timeout: %s s\n' \
    "$VLLM_OMNI_DLO_DP_WAVE_TIMEOUT" "$VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT"
exec "$H3_PYTHON" -m examples.diffusionnft_trainer.minimax_h3.infer_minimax_h3_video_edit \
    --model-path "$MODEL_PATH" \
    --lora-path "$LORA_WARMSTART_PATH" \
    --data-file "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --start-index "$START_INDEX" \
    --num-inference-steps "$INFER_STEPS" \
    --tensor-parallel-size "$ROLLOUT_TP" \
    --text-encoder-tp-size "$TEXT_ENCODER_TP" \
    --reference-image-short-edge "$REF_IMAGE_SHORT_EDGE" \
    --attention-backend "$ROLLOUT_ATTN_BACKEND" \
    "$@"
