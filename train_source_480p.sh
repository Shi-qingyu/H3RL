#!/usr/bin/env bash
set -euo pipefail

export RUN_PREFIX=rl_source_and_output_480p_32gpu
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train.sh" \
  '++ray_kwargs.ray_init.runtime_env.env_vars.H3_REFERENCE_VIDEO_SHORT_EDGE="480"' \
  '+trainer.rollout_data_save_source_video=true' \
  'actor_rollout_ref.model.enable_gradient_checkpointing=false' \
  "$@"
