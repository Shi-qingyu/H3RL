# H3 SFT v6 with the three-head editing reward

Last updated: 11/15/2026.

This recipe loads the official H3 model and the supplied DiffSynth SFT EMA LoRA
as separate weights, then trains the adapter with online DiffusionNFT. It prepares
source-video editing prompts from the October 29 V6 release. The configured paths
are in [video_edit_v6.env](video_edit_v6.env), which sets `H3_BASE_MODEL_PATH` to
`/mnt/geminihzceph/public/model/MiniMax-H3`. Sourcing the file derives `MODEL_PATH`
from this selected base. Other input and output defaults respect
environment variables set beforehand.

## Load the official H3 model and SFT LoRA

The supplied `step-1626-ema.safetensors` contains native H3 LoRA weights, with
rank 32 and alpha 32. It does not contain the frozen DiT, text encoder, tokenizer
or VAEs. Its training configuration identifies the base release at
`/mnt/geminihzceph/public/model/MiniMax-H3`.

`MODEL_PATH` points to the official H3 root through `H3_BASE_MODEL_PATH`.
The adapter preparation script remaps the native LoRA into
Diffusers LoRA format, handling H3's head-interleaved QKV and the FC1 row-order difference.
It writes the converted adapter to `LORA_WARMSTART_PATH`, which defaults to
`$V6_OUTPUT_DIR/sft_lora`. The SFT adapter stays separate from the base model;
preparation does not merge weights or write full transformer shards.

The training launcher converts the adapter automatically when
`LORA_WARMSTART_PATH` is absent. To prepare it separately, run from the repository
root:

```bash
source .venv/bin/activate
source examples/diffusionnft_trainer/minimax_h3/video_edit_v6.env

python examples/diffusionnft_trainer/minimax_h3/prepare_video_edit_sft_lora.py \
    --base-model "$H3_BASE_MODEL_PATH" \
    --checkpoint "$H3_SFT_CHECKPOINT" \
    --lora-alpha 32 \
    --output-dir "$LORA_WARMSTART_PATH"
```

The input checkpoint configured in the environment file is:

```text
/mnt/geminihzceph/toddhu/DiffSynthStudio/artifacts/h3_sft_v6/training_release_native_v5_20261029_r1/runs/world128_lr1e4_2ep/step-1626-ema.safetensors
```

Preparation writes `pytorch_lora_weights.safetensors`, `adapter_config.json` and
`conversion_manifest.json` into a new local directory. The safetensors file stores
the rank and alpha in Diffusers loader metadata. The official base release and
source checkpoint remain accessible at their configured locations. The output
directory must be absent.

On shared filesystems that reject `RENAME_NOREPLACE`, the adapter exporter creates
the destination exclusively and publishes `conversion_manifest.json` after the
other files are complete. Existing output directories are never replaced.

Training loads the converted SFT adapter into three policy states:

- `default` starts from the SFT weights and receives RL updates.
- `old` starts from the same SFT weights and follows the existing old-policy
  update schedule; rollout uses this adapter, including before the first update.
- `reference` retains the initial SFT weights and stays frozen for the reference
  prediction used by the NFT objective.

The launcher selects `policy_state_adapters=[default,old,reference]` and uses
`LORA_WARMSTART_PATH` for initialization. The official H3 weights remain the
shared frozen base throughout training, with `actor_rollout_ref.model.lora.merge=False`.
The reference snapshot is used without gradients by the FSDP DiffusionNFT engine;
copy and EMA updates only change the trainable/old policies.

## Prepare the V6 source videos and instructions

The two supplied release files are cache manifests:

```text
training_release_native_v5_20261029_r1/task_metadata.jsonl
training_release_native_v5_20261029_r1/cache_index.tsv
```

Online RL generates and scores new videos. The converter joins these files by
cache path and sample ID to the producer's `frozen_training.jsonl`,
`PROVENANCE.jsonl`, and per-lane `validation.jsonl` media receipts. Their paths
are configured alongside the release paths in `video_edit_v6.env`. Cached
`.pth` tensors and SFT target videos are not used as rollout inputs.

The converter selects all **1,724 `passerby_remove` records** in this release
before the train/validation split. It retains every native frame-count and canvas
bucket: 22, 39, 56, 73, 90, 107 and 124 frames, each in portrait and landscape,
for 14 shapes in total. Object removal and other V6 editing categories remain excluded. The
default `DATA_DIR` is `$V6_OUTPUT_DIR/data_all_passerby_remove`, separate from the
earlier 73-frame portrait subset in `data`.

H3 receives the full frozen IR. For passerby removal, the converter maps the
explicit keep-main/remove-all task to the matching canonical reward instruction.
Remove-all accepts both
`Remove every real person.` and its longer `including distant pedestrians and riders`
variant. For reworded IR without a recognized declaration, the converter reads
the original instruction from its provenance-bound source line. It requires the
raw line's SHA256, source `uid`, and rewritten training instruction to match the
recorded provenance and frozen IR before using that original task declaration.
`extra_info` and the conversion manifest record the instruction origin.
Conflicting declarations, failed provenance bindings, and unrecognized original
tasks stop conversion.

Each row carries its native `rollout_shape` (frames, height, width, 24 FPS).
`VideoEditShapeSampler` shuffles batches grouped by this shape, so all rollout
workers and actor ranks receive compatible tensors within a global prompt batch.
Each bucket's final batch repeats samples to fill the batch, without discarding
sources. The sampler saves its epoch and cursor for checkpoint resume. When enabled,
validation uses one source per batch and exports individual named previews without stacking
different video shapes. `NUM_FRAMES`, `HEIGHT`, and `WIDTH` only supply initial
engine defaults; each row overrides them with its own rollout shape.

Training and validation share the preview script's short-output preprocessing:
the 4-second output minimum is removed, while H3's `17n+5` alignment remains.
A configured 73-frame rollout therefore generates 73 frames. Short positive-duration
source clips are also accepted, including the release's sub-2-second removal clips.
H3's existing maximum-duration and media-format checks remain in effect.

```bash
python examples/diffusionnft_trainer/minimax_h3/prepare_video_edit_v6_data.py \
    --release-dir "$V6_RELEASE_DIR" \
    --frozen-manifest "$V6_FROZEN_MANIFEST" \
    --provenance "$V6_PROVENANCE_MANIFEST" \
    --output-dir "$DATA_DIR" \
    --val-fraction 0.05 \
    --seed 42
```

Splitting groups together shared original `source_group` values and identical
source-video paths, so related sources do not appear on both sides of this RL
split. The script writes `train.parquet`, `test.parquet`, and
`conversion_manifest.json` with all native geometries, counts,
and sample IDs. Existing output files are not overwritten.
The launcher runs this conversion automatically if the manifest is absent.
The converter no longer accepts frame-count or canvas filters. The launcher
rejects legacy or filtered manifests, so an old `DATA_DIR` cannot silently reuse
the 73-frame subset. Set `DATA_DIR` to the full-data directory shown below.

The original source clip can have fewer frames than the padded SFT target;
the example receipt has 67 valid frames and six padding frames. The converter
keeps the original source file and the IR's padding instruction, and records
these counts in `extra_info`. It does not decode, pad, crop or re-encode source
files. Rollout retains the existing upstream H3 reference preprocessing, and the
reward service reads the full original source file.

## Start reward scoring and training

Start the supplied reward checkpoint with the separate service environment,
following [the reward-service instructions](README_video_edit.md#start-the-reward-service).
Then launch training with the H3 training environment active:

```bash
export VIDEO_REWARD_URL=http://127.0.0.1:19085/score
export VIDEO_REWARD_WEIGHTS='[1.0,1.0,1.0]'
export DATA_DIR="$PWD/outputs/minimax_h3_video_edit_v6/data_all_passerby_remove"
export OUTPUT_DIR="$PWD/outputs/minimax_h3_video_edit_v6/rl_all_passerby_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
N_GPUS=6 ACTOR_SP=2 ROLLOUT_TP=2 TEXT_ENCODER_TP=2 ROLLOUT_N=4 \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_video_edit_v6_lora.sh \
    actor_rollout_ref.model.trust_remote_code=False \
    data.train_batch_size=3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=3 \
    ray_kwargs.ray_init.num_cpus=32 \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init.num_gpus=6 \
    'trainer.logger=[console]' \
    trainer.val_before_train=False \
    data.val_max_samples=4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.log_val_generations=0 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
```

The V6 launcher defaults to console logging and does not require TensorBoard or
W&B. The video-edit base launcher redirects stdin to `/dev/null` before starting
Ray. H3 reference preprocessing invokes FFmpeg without `-nostdin`; inherited
terminal input can cause `SIGTTOU` to stop a background worker process group,
which then surfaces as a Ray RPC handshake timeout. Redirecting stdin follows
[FFmpeg's background-task guidance](https://ffmpeg.org/faq.html#How-do-I-run-ffmpeg-as-a-background-task_003f),
while stdout/stderr continue through the caller's `tee` and Ctrl+C remains enabled.
This change requires a fresh local Ray instance; it does not resume stopped workers
from an earlier run. The launcher sets `actor_rollout_ref.model.trust_remote_code=False` for the built-in
Diffusers transformer and separately sets
`actor_rollout_ref.rollout.engine_kwargs.vllm_omni.trust_remote_code=True` for
the checkpoint's custom VAE code. The installed Diffusers `AutoModel` rejects
`trust_remote_code=True` when the transformer config has no `auto_map`; do not
override the actor setting to `True` in the launch command.

This example reserves physical GPUs 0–5 for training and runs the reward service
on physical GPU 6, with GPU 7 available. The V6 launcher skips validation by default.
Every five training steps, it exports training samples under `$OUTPUT_DIR/rollout_data/`
and saves a checkpoint under `$OUTPUT_DIR/checkpoints/global_step_<step>/`.
It keeps every checkpoint. Auto-resume uses the latest saved step.

`RAY_NUM_CPUS` defaults to 32. The six-GPU training resource group requests 18
CPUs (three per GPU), with additional CPU resources needed for the controller and
auxiliary workers. A limit of 16 cannot schedule this group. The launcher enables
unbuffered Python output and disables Ray log deduplication by default. The
command above saves terminal output and errors to `$OUTPUT_DIR/train.log`;
The V6 launcher enables automatic diagnostics by default; no second terminal or
manual capture is needed. Each invocation gets a separate directory under
`$OUTPUT_DIR/diagnostics/<timestamp>_<launcher-pid>/`:

- `workers/`: instrumented actors write Python thread stacks every ten seconds
  for the first hour after diagnostics start. Actor initialization records rank
  and assigned visible devices; rollout server initialization and agent RPC
  submission also enable sampling in their respective processes. JSONL events
  record initialization, readiness, RPC submission and receipt, LoRA resolution,
  generation and output.
  Diagnostics start inside actor/task methods, after Ray assigns GPU visibility.
  Do not use `verl_omni.utils.runtime_diagnostics.enable_worker_diagnostics` as a
  `worker_process_setup_hook`: importing the parent package at that earlier stage
  can initialize CUDA before per-actor device assignment.
  Sampling uses a Python thread and `sys._current_frames()`, avoiding the native
  `faulthandler.dump_traceback_later()` watchdog and its concurrent frame-read
  crash risk ([CPython #140815](https://github.com/python/cpython/issues/140815)).
  It cannot sample while native code holds the GIL; use the external captures for
  that case. Snapshots contain timestamps and are observations, not exceptions.
- `captures/`: an automatically started collector captures rollout server stacks
  for up to ten minutes after the first server is ready. Once the first rollout
  RPC is submitted, it also attempts one `gdb` snapshot per server when available,
  including native-only RPC threads. Each inspection has a five-second timeout;
  debugger attachment can briefly pause that process.
- `ray_logs/`: the launcher copies this invocation's local Ray session logs when
  training returns, preserving the training exit status. The original logs remain
  in a unique `/tmp/h3ray.*` directory printed in the resolved Ray configuration.
- `collector.log`: status and failures from the automatic collector.
- `system/start/` and `system/exit/`: Python version, host and cgroup memory/OOM
  counters, GPU memory and recent kernel records captured before training and
  after it returns. Permission failures are recorded when kernel logs are not
  accessible. These snapshots also run for failures before rollout starts; each
  command and the overall snapshot have time limits.

The collector is scoped to this invocation's local Ray directory and stops
automatically. It skips `ray status`, whose authentication environment may differ
from training. Set `H3_AUTO_DIAGNOSTICS=0` to disable the diagnostic setup once the
failure is resolved. The change to enable sampling from actor/task methods has
not been validated in a training run.

The editing launcher defaults `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` to 3600 seconds and
`VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` to the same value. Set either environment variable
before launching to override it. Both values are printed at startup and explicitly
forwarded through Ray's runtime environment, including when using an existing
cluster, so training and validation workers receive the configured timeouts.

The V6 launcher checks the adapter preparation metadata against the configured
SFT checkpoint and reads the shape buckets from the dataset manifest. It derives
the reference-row padding capacity from the largest native shape, keeping the
text sequence limit separate at 12,288. This avoids the old 12,288-row reference
limit being exceeded by larger source videos. Padding to this larger capacity
uses more memory. It uses the existing Ref2VA DiffusionNFT launcher with per-row
training and validation canvases. Configure resources with `N_GPUS`, `ACTOR_SP` and `ROLLOUT_TP`,
and allocate the reward service separately. These settings have not been measured
for memory or throughput. Additional Hydra overrides go at the end of the command.
The default training output is `$V6_OUTPUT_DIR/training_sft_lora`, separate from
the previous fused-base run directory.

The adapter conversion and training changes have not been executed or tested.

## Validate a saved RL checkpoint on a separate six-GPU node

Run the following from the repository root on the evaluation node. The example uses
three local TP=2 replicas on GPUs 1–6. The existing VideoReward service runs on its
own GPU, can read the source videos on the shared filesystem, and provides the
`/health` and `/score` endpoints. Set `RUN_DIR` to the training run and `STEP` to a
completed save. The checkpoint, prepared dataset and base model must be readable on this node.

```bash
STEP=5
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_video_edit_v6_validation.sh \
    "$RUN_DIR/checkpoints/global_step_$STEP" \
    --gpus 1,2,3,4,5,6 --reward-url http://127.0.0.1:19085/score
```

The evaluator reads only the selected RL adapter from the model shards and uses
the layout metadata of the prepared SFT adapter. It evaluates `default` (the
updated actor) by default. Pass `--adapter old` to evaluate the rollout policy used by
in-training validation. Both use the official H3 base. No training Ray cluster is required.

The default uses the same four held-out rows as the 24-GPU recipe, seed 42, native
per-row shapes, one video per row, and 40 denoising steps. Use `--num-samples 0`
to evaluate the full test split. Each generated video is scored from its decoded
frames using the same three-head VideoReward client as training. The evaluator
assigns each row to one replica and averages the per-video scores with equal weight.

Every invocation prints its new directory under `$RUN_DIR/offline_validation/`.
`run_config.json` records the selected rows, GPU assignments and reward checkpoint.
`group_0.log` through `group_2.log` show live progress. The group directories contain
the saved videos and individual scores. A completed run publishes `summary.json`
with `val-aggregate/reward/mean`, the three dimension means, and links to each row's result.
