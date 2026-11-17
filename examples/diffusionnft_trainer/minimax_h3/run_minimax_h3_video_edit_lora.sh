#!/usr/bin/env bash
# H3 Ref2VA DiffusionNFT video editing with a three-head VideoReward service.
set -euo pipefail

# H3 reference preprocessing invokes FFmpeg with inherited stdin. Ray workers
# run in a background process group; terminal access can stop the whole group
# with SIGTTOU. This batch job never reads interactive input.
exec </dev/null

: "${MODEL_PATH:?Set MODEL_PATH to the H3 root containing Ref2VA/ and transformer_ref/}"
: "${DATA_DIR:?Set DATA_DIR to parquet files produced by prepare_video_edit_data.py}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/outputs/minimax_h3_video_edit_lora}
export WANDB_RUN_ID=${WANDB_RUN_ID:-minimax_h3_video_edit_lora}
VIDEO_REWARD_URL=${VIDEO_REWARD_URL:-http://127.0.0.1:19085/score}
VIDEO_REWARD_WEIGHTS=${VIDEO_REWARD_WEIGHTS:-[1.0,1.0,1.0]}
VIDEO_FPS=${VIDEO_FPS:-24.0}
# Match the preview timeout for a full video forward and queued output transfer.
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-3600}
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=${VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT:-$VLLM_OMNI_DLO_DP_WAVE_TIMEOUT}

printf 'H3 execution RPC timeout: %s s; async output timeout: %s s\n' \
    "$VLLM_OMNI_DLO_DP_WAVE_TIMEOUT" "$VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT"

exec bash "$SCRIPT_DIR/run_minimax_h3_ref2va_lora.sh" \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=${oc.env:VLLM_OMNI_DLO_DP_WAVE_TIMEOUT}' \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=${oc.env:VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT}' \
    '~reward.reward_functions.clap' \
    '~reward.reward_functions.imagebind' \
    '+reward.reward_functions.video_edit.path=pkg://verl_omni.utils.reward_score.video_edit_reward' \
    '+reward.reward_functions.video_edit.name=compute_score' \
    '+reward.reward_functions.video_edit.weight=1.0' \
    '+reward.reward_functions.video_edit.required=true' \
    "+reward.reward_functions.video_edit.server_url=$VIDEO_REWARD_URL" \
    "+reward.reward_functions.video_edit.fps=$VIDEO_FPS" \
    "+reward.reward_functions.video_edit.score_weights=$VIDEO_REWARD_WEIGHTS" \
    actor_rollout_ref.rollout.pipeline.frame_rate="$VIDEO_FPS" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate="$VIDEO_FPS" \
    'actor_rollout_ref.rollout.pipeline.requested_outputs=[video_preview]' \
    'actor_rollout_ref.rollout.val_kwargs.pipeline.requested_outputs=[video_preview]' \
    trainer.video_fps="$VIDEO_FPS" \
    trainer.experiment_name=minimax_h3_video_edit_lora \
    "$@"
