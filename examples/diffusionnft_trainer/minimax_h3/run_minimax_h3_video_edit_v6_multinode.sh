#!/usr/bin/env bash
# Run on the head of an existing Ray cluster with eight H20s per training node.
set -euo pipefail

: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head IP:port}"
: "${VIDEO_REWARD_URL:?Set VIDEO_REWARD_URL to the reachable VideoReward /score endpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the shared directory for this run}"
TRAIN_NODES=${TRAIN_NODES:-3}
if [[ ! $TRAIN_NODES =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_NODES must be a positive integer" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export N_GPUS=8 ACTOR_SP=2 ROLLOUT_TP=2 TEXT_ENCODER_TP=2 ROLLOUT_N=4
# The single-node collector attaches to a local Ray temp directory.
export H3_AUTO_DIAGNOSTICS=0
rollout_workers=$((TRAIN_NODES * N_GPUS / ROLLOUT_TP))
# One candidate per actor data-parallel group, four candidates per prompt.
prompt_batch_size=$((TRAIN_NODES * N_GPUS / ACTOR_SP / ROLLOUT_N))

exec bash "$SCRIPT_DIR/run_minimax_h3_video_edit_v6_lora.sh" \
    "+ray_kwargs.ray_init.address=$RAY_ADDRESS" \
    ray_kwargs.ray_init.num_cpus=null \
    "trainer.nnodes=$TRAIN_NODES" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    "actor_rollout_ref.rollout.agent.num_workers=$rollout_workers" \
    "data.train_batch_size=$prompt_batch_size" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$prompt_batch_size" \
    trainer.val_before_train=False \
    data.val_max_samples=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    "$@"
