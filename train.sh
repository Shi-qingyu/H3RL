cd /mnt/geminihzceph/qingyushi/projects/rl/verl-omni
source .venv/bin/activate

export V6="$PWD/outputs/minimax_h3_video_edit_v6"
export SCRIPTS="$PWD/examples/diffusionnft_trainer/minimax_h3"
export OLD="$V6/gemini_1885013190_24gpu"

export RAY_ADDRESS=29.209.107.21:6379
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH="$OLD/.ray-token"
unset RAY_AUTH_TOKEN
export TRAIN_NODES=4

export VIDEO_REWARD_URL=http://29.209.105.115:19085/score
export VIDEO_REWARD_WEIGHTS='[1.0,1.0,1.0]'
export DATA_DIR="$V6/data_all_passerby_remove"
export TRAIN_480="$V6/train_passerby_remove_480p.parquet"
export LORA_WARMSTART_PATH="$V6/sft_lora"
export TOTAL_TRAINING_STEPS=1000

export ACTOR_ATTN_BACKEND=native
export ROLLOUT_ATTN_BACKEND=TORCH_SDPA
export FFBIN="$OLD/diagnostics/rollout-20261120-112626-vJfc1J/bin"

OUTPUT_DIR=$(mktemp -d "$V6/rl_480p_32gpu_newreward_$(date +%Y%m%d_%H%M%S)_XXXXXX")
export OUTPUT_DIR
echo "本次训练目录 $OUTPUT_DIR"

set -o pipefail
test -r "$TRAIN_480" && test -x "$FFBIN/ffmpeg" && \
bash "$SCRIPTS/run_minimax_h3_video_edit_v6_multinode.sh" \
  "data.train_files=$TRAIN_480" \
  'actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=8' \
  '++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_NET=Socket' \
  '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout=1800' \
  '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.diffusion_offload_config={mode:layer,components:[dit,text_encoder],pin_memory:false}' \
  "++ray_kwargs.ray_init.runtime_env.env_vars.PATH=$FFBIN:\$PATH" \
  2>&1 | tee "$OUTPUT_DIR/train.log"