# H3 DiffusionNFT video editing with VideoReward

Last updated: 11/10/2026.

This recipe uses H3's existing Ref2VA DiffusionNFT LoRA implementation to condition
on a source video. Generated decoded videos are scored by a separate VideoReward
HTTP service. The training workers send the generated `video_preview`, the source
video path, the editing instruction and FPS; the service loads the reward model
once and reuses it.

The supplied checkpoint is a Qwen3.5-9B LoRA reward model with three regression
heads: instruction following, source fidelity and visual quality. Its two
instruction templates represent two removal tasks: remove unwanted passersby
while preserving the main subject, or remove all people. Each sample carries
one editing instruction. The default scalar reward is the weighted mean of the
three raw head predictions, with weights `[1,1,1]`. The human targets use a 1–5
scale; predictions are not clipped, normalized or passed through a sigmoid.
Individual head predictions are also returned as metrics.

For the supplied DiffSynth `step-1626-ema.safetensors` and October 29 V6 cache
manifests, use the [SFT v6 recipe](README_video_edit_v6.md), which includes the
native-to-PEFT SFT LoRA conversion and manifest-to-source conversion steps. It
loads the official H3 base and SFT adapter separately without merging them.

## Create the local environments

From the verl-omni repository root, create the H3 training environment with uv:

```bash
bash setup_uv.sh
source .venv/bin/activate
```

The script defaults to Python 3.12 and installs the GPU and media extras. Set
`PYTHON_VERSION` or `TORCH_BACKEND` when creating the environment to choose a
different interpreter or PyTorch backend. It requires the `uv` command.

Create the reward service environment separately because VideoReward uses
different Transformers and vLLM versions:

```bash
VIDEO_REWARD_ROOT=/mnt/geminihzceph/qingyushi/projects/reward_model/VideoReward \
    bash setup_uv.sh reward
```

This uses VideoReward's frozen lockfile and installs into this repository's
`.venv-video-reward`, adding `imageio[ffmpeg]` for lossless video encoding. Its
FlashAttention dependency needs a CUDA Toolkit and C++ compiler for compilation.
Use `.venv-video-reward/bin/python` for the reward service and `.venv/bin/python`
for data preparation and training. Existing environments are reused; incomplete
environments or broken interpreter links require moving the old directory aside.

## Prepare the editing prompts

Create `train.jsonl` and `test.jsonl` in an input directory. Each record contains
one source video and its semantic editing instruction:

```json
{"source_video":"clips/source.mp4","edit_instruction":"Remove the unwanted passersby, including their shadows and reflections, from the video and naturally fill the removed regions. Preserve the main subject, other scene content, and camera motion."}
```

For the other checkpoint task, use:

```text
Remove all people, including their shadows and reflections, from the video and naturally fill the removed regions. Preserve all nonhuman objects, other scene content, and camera motion.
```

An optional `prompt` contains a separate H3 generation prompt. It defaults to
`edit_instruction`; changing it does not change the instruction given to the
reward model. Omit `<video>` and other media placeholders: the converter adds
the single source-video placeholder.

Paths are resolved relative to the JSONL input directory and saved as absolute
paths. The same source-video paths must be readable by the reward service and
every rollout worker. If optional `video` or `videos` conditioning fields are
provided, they must identify that same single source video. Additional references,
ambiguous references and nonzero `start_time_seconds` are rejected. Pretrim source
files to the desired editing interval and align them with the generated clip's
duration. The reward service reads the complete source file; H3 retains its
upstream reference preprocessing. This integration does not crop or offset a
source file for reward scoring. The base recipe generates 96 frames at 24 FPS
by default, corresponding to four-second outputs.

Run from the verl-omni repository root after setting the input and output
directories:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/prepare_video_edit_data.py \
    --input_dir "$EDIT_JSONL_DIR" \
    --output_dir "$DATA_DIR"
```

Use `--val_split val` when the validation input file is `val.jsonl`. The output
files are always `train.parquet` and `test.parquet`. The converter writes
`extra_info.source_video` and `extra_info.edit_instruction`, and independently
sets `reward_model.ground_truth` to the editing instruction. Reward training
pairs with two entries in `videos=[source, edited]` are not this prompt-only input
format; select the source and instruction when preparing the RL dataset.

## Start the reward service

Use a Python environment compatible with the existing VideoReward repository and
its checkpoint, and invoke the server from the verl-omni repository root. The
service also needs `fastapi`, `uvicorn`, `safetensors` and `imageio[ffmpeg]`, with
an FFmpeg build providing `libx264rgb` for lossless generated-video encoding. The
service uses the checkpoint's saved model, adapter and template configuration;
the generated uint8 frames are written to a temporary lossless RGB video for
the same VideoReward video preprocessing, including its saved 4 FPS sampling.

```bash
export VIDEO_REWARD_ROOT=/mnt/geminihzceph/qingyushi/projects/reward_model/VideoReward
export VIDEO_REWARD_CHECKPOINT="$VIDEO_REWARD_ROOT/checkpoints/three_dim_removal_two_instructions_20261109_112935/trials/two_instructions/removal/train/checkpoint-288"

.venv-video-reward/bin/python examples/diffusionnft_trainer/minimax_h3/serve_video_edit_reward.py \
    --source-root "$VIDEO_REWARD_ROOT" \
    --checkpoint "$VIDEO_REWARD_CHECKPOINT" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port 19085
```

The source root and checkpoint above are also the service defaults. `--base-model`
can override the base-model location if the saved path is unavailable on the
service host. The service defaults to the saved float32 model dtype;
`--dtype bfloat16` selects lower precision explicitly. Allocate reward-service device
resources separately from the devices managed by the H3 training job. For remote
workers, bind the service to a reachable interface and set `VIDEO_REWARD_URL` to
that host's `/score` endpoint.

## Train H3 with DiffusionNFT

Fill in the H3 model root and the prepared parquet directory. The model root must
contain both `Ref2VA/` for rollout and `transformer_ref/` for actor training; the
reward checkpoint is a separate model served by the command above.

```bash
export MODEL_PATH=""
export DATA_DIR=""
export VIDEO_REWARD_URL=http://127.0.0.1:19085/score
export VIDEO_REWARD_WEIGHTS='[1.0,1.0,1.0]'

bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_video_edit_lora.sh
```

Weights are ordered as instruction following, source fidelity and visual quality.
`VIDEO_FPS` defaults to `24.0` and is forwarded to both rollout and reward scoring.
The wrapper requests a decoded `video_preview`, removes the base recipe's CLAP
and ImageBind rewards, and marks this reward `required=true`, so service failures
stop training. It reuses the existing Ref2VA recipe's explicit v0 trainer path,
online DiffusionNFT objective, LoRA configuration and rollout settings. The H3
recipe's remaining settings are described in [README.md](README.md).

Additional Hydra overrides may be passed at the end of the launch command, for
example `trainer.total_training_steps=100`; caller overrides take precedence.
The model path and dataset path remain unset until supplied. This integration has
not been executed: data preparation, model loading, service requests and training
have not been validated.
