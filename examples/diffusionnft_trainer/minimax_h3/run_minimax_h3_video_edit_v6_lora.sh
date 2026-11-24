#!/usr/bin/env bash
# H3 SFT v6 initialization and V6 source-video prompts with DiffusionNFT.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd -- "$REPO_ROOT"
source "$SCRIPT_DIR/video_edit_v6.env"

export INFER_STEPS=${INFER_STEPS:-20}
# A 40-step validation forward on a native 124-frame video can exceed one hour.
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-10800}
export VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=${VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT:-3600}
export OUTPUT_DIR=${OUTPUT_DIR:-$V6_OUTPUT_DIR/training_sft_lora}
export WANDB_RUN_ID=${WANDB_RUN_ID:-minimax_h3_video_edit_v6_sft_lora}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}

if [[ ! -f "$DATA_DIR/conversion_manifest.json" ]]; then
    python3 "$SCRIPT_DIR/prepare_video_edit_v6_data.py" \
        --release-dir "$V6_RELEASE_DIR" \
        --frozen-manifest "$V6_FROZEN_MANIFEST" \
        --provenance "$V6_PROVENANCE_MANIFEST" \
        --output-dir "$DATA_DIR"
fi

# Convert only adapter factors; the official H3 base remains separate and unchanged.
if [[ ! -e "$LORA_WARMSTART_PATH" && ! -L "$LORA_WARMSTART_PATH" ]]; then
    python3 "$SCRIPT_DIR/prepare_video_edit_sft_lora.py" \
        --base-model "$H3_BASE_MODEL_PATH" \
        --checkpoint "$H3_SFT_CHECKPOINT" \
        --lora-alpha 32 \
        --output-dir "$LORA_WARMSTART_PATH"
fi

# Read the native geometry and reference-row capacity from the prepared dataset.
adapter_settings=$(python3 - "$MODEL_PATH" "$LORA_WARMSTART_PATH" "$DATA_DIR" "$H3_SFT_CHECKPOINT" \
    "$VIDEO_FPS" <<'PY'
import json
import math
import sys
from pathlib import Path

model_dir, adapter_dir, data_dir, checkpoint = (Path(value).expanduser().resolve() for value in sys.argv[1:5])
adapter_manifest = adapter_dir / "conversion_manifest.json"
data_manifest = data_dir / "conversion_manifest.json"
if not adapter_manifest.is_file():
    raise SystemExit("Prepare the separate SFT LoRA with prepare_video_edit_sft_lora.py in a fresh adapter directory.")
if not data_manifest.is_file():
    raise SystemExit("Prepare V6 data first with prepare_video_edit_v6_data.py.")
adapter = json.loads(adapter_manifest.read_text(encoding="utf-8"))
if adapter.get("format") != "minimax_h3_sft_lora" or adapter.get("base_weights_merged") is not False:
    raise SystemExit("LORA_WARMSTART_PATH must contain the separately converted SFT LoRA.")
if Path(adapter["checkpoint"]).resolve() != checkpoint or Path(adapter["base_model"]).resolve() != model_dir:
    raise SystemExit("Prepared SFT LoRA checkpoint/base differs from H3_SFT_CHECKPOINT/MODEL_PATH.")
weight_name = adapter["weight_name"]
if weight_name != "pytorch_lora_weights.safetensors" or not (adapter_dir / weight_name).is_file():
    raise SystemExit("Converted SFT LoRA weights are missing; choose a fresh adapter directory and prepare again.")
rank, alpha = adapter["rank"], adapter["lora_alpha"]
if any(type(value) is not int or value <= 0 for value in (rank, alpha)):
    raise SystemExit("Converted SFT LoRA must declare a positive integer rank and alpha.")
data = json.loads(data_manifest.read_text(encoding="utf-8"))
if data.get("format") != "video_edit_v6":
    raise SystemExit("DATA_DIR must contain data prepared by prepare_video_edit_v6_data.py.")
released_passerby = data.get("released_categories", {}).get("passerby_remove", 0)
if (
    data.get("category_filter") != "passerby_remove"
    or data.get("shape_policy") != "per_sample"
    or data.get("selection") != {"num_frames": None, "height": None, "width": None}
    or released_passerby <= 0
    or data.get("selected_records") != released_passerby
):
    raise SystemExit(
        "DATA_DIR must contain all passerby_remove records without frame/canvas filters. "
        "Use a fresh DATA_DIR such as $V6_OUTPUT_DIR/data_all_passerby_remove; the launcher will prepare it."
    )
buckets = data["shape_buckets"]
if not buckets:
    raise SystemExit("The prepared V6 data has no shape buckets.")
fps = float(sys.argv[5])
if not math.isfinite(fps) or fps != data["fps"]:
    raise SystemExit("VIDEO_FPS must match the prepared V6 source frame rate.")
# H3 maps 17n+5 frames to 5n+2 latent frames, with one row per 32x32 pixel patch.
# These native canvases bound the upstream reference canvas (short edge <=768).
reference_rows = max(
    ((bucket["num_frames"] - 5) // 17 * 5 + 2) * (bucket["height"] // 32) * (bucket["width"] // 32)
    for bucket in buckets
)
padding_length = math.ceil(max(12288, reference_rows) / 1024) * 1024
print(rank, alpha, padding_length)
print(
    f"V6 data: categories={data['category_filter']}, train={data['train_records']}, "
    f"validation={data['validation_records']}, shape_buckets={len(buckets)}, per_sample_shape=True",
    file=sys.stderr,
)
PY
)
read -r sft_lora_rank sft_lora_alpha reference_padding_length <<< "$adapter_settings"
export MAX_PROMPT_EMBEDS=${MAX_PROMPT_EMBEDS:-$reference_padding_length}

diagnostic_args=()
case ${H3_AUTO_DIAGNOSTICS:-1} in
    1)
        diagnostic_dir="$OUTPUT_DIR/diagnostics/$(date +%Y%m%d_%H%M%S)_$$"
        mkdir -p -- "$diagnostic_dir/workers"
        diagnostic_dir=$(cd -- "$diagnostic_dir" && pwd -P)
        ray_tmp_dir=$(mktemp -d /tmp/h3ray.XXXXXX)
        export VERL_OMNI_DIAGNOSTICS_DIR="$diagnostic_dir/workers"
        diagnostic_args+=(
            "+ray_kwargs.ray_init._temp_dir=$ray_tmp_dir"
            "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_OMNI_DIAGNOSTICS_DIR=$VERL_OMNI_DIAGNOSTICS_DIR"
        )
        RAY_DIAGNOSTICS_DIR="$diagnostic_dir/captures" RAY_DIAGNOSTICS_RAY_ROOT="$ray_tmp_dir" \
            bash "$SCRIPT_DIR/diagnose_video_edit_ray.sh" --watch > "$diagnostic_dir/collector.log" 2>&1 &
        diagnostic_pid=$!
        finish_diagnostics() {
            training_status=$?
            trap - EXIT
            set +e
            diagnostic_parent=$(ps -p "$diagnostic_pid" -o ppid=)
            if [[ "${diagnostic_parent// /}" == "$$" ]]; then
                kill -TERM "$diagnostic_pid" 2>/dev/null
            fi
            wait "$diagnostic_pid" 2>/dev/null || true
            RAY_DIAGNOSTICS_DIR="$diagnostic_dir/system/exit" \
                timeout --kill-after=2s 35s bash "$SCRIPT_DIR/diagnose_video_edit_ray.sh" --system \
                > "$diagnostic_dir/system_exit.log" 2>&1
            for session_logs in "$ray_tmp_dir"/session_20*/logs; do
                [[ -d "$session_logs" ]] || continue
                session_name=${session_logs%/logs}
                session_name=${session_name##*/}
                mkdir -p -- "$diagnostic_dir/ray_logs/$session_name"
                timeout --kill-after=2s 30s cp -a -- "$session_logs/." \
                    "$diagnostic_dir/ray_logs/$session_name/" || \
                    printf 'Ray log copy incomplete; originals remain in %s\n' "$session_logs" >&2
            done
            printf '\nTraining exit=%s; diagnostics: %s\n' "$training_status" "$diagnostic_dir"
            exit "$training_status"
        }
        trap finish_diagnostics EXIT
        printf 'Automatic worker diagnostics: %s\n' "$diagnostic_dir"
        RAY_DIAGNOSTICS_DIR="$diagnostic_dir/system/start" \
            timeout --kill-after=2s 35s bash "$SCRIPT_DIR/diagnose_video_edit_ray.sh" --system \
            > "$diagnostic_dir/system_start.log" 2>&1 || \
            printf 'Initial system capture incomplete; see %s/system_start.log\n' "$diagnostic_dir" >&2
        ;;
    0) ;;
    *) echo "H3_AUTO_DIAGNOSTICS must be 0 or 1" >&2; exit 2 ;;
esac

bash "$SCRIPT_DIR/run_minimax_h3_video_edit_lora.sh" \
    algorithm.timestep_fraction=0.5 \
    actor_rollout_ref.model.trust_remote_code=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.trust_remote_code=True' \
    "actor_rollout_ref.model.lora_rank=$sft_lora_rank" \
    "actor_rollout_ref.model.lora_alpha=$sft_lora_alpha" \
    actor_rollout_ref.model.lora.merge=False \
    'actor_rollout_ref.model.policy_state_adapters=[default,old,reference]' \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=12288 \
    data.max_prompt_length=12288 \
    data.filter_overlong_prompts=False \
    data.train_max_samples=-1 \
    "ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS" \
    "+data.sampler.class_path=$SCRIPT_DIR/video_edit_shape_sampler.py" \
    '+data.sampler.class_name=VideoEditShapeSampler' \
    data.val_batch_size=1 \
    trainer.experiment_name=minimax_h3_video_edit_v6_sft_lora \
    trainer.rollout_data_save_freq=5 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=5 \
    trainer.keep_best_validation_checkpoints=0 \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.log_val_generations=0 \
    'trainer.logger=[console]' \
    "${diagnostic_args[@]}" \
    "$@"
