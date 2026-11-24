# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preview the H3 SFT policy or score an exported RL policy on prepared V6 validation rows.

Uses vLLM-Omni's MiniMaxH3Pipeline and the same LoRA mapping as the NFT rollout.
For 24 FPS sources, export exactly the source frame count after removing the
extra tail generated for H3's 17n+5 temporal alignment, without a 4-second floor.
Run through run_minimax_h3_video_edit_infer.sh from an H3 training environment.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm_omni.diffusion.lora.layers.base_linear import DiffusionBaseLinearLayerWithLoRA
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_align_frame_count
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import MINIMAX_H3_FPS
from vllm_omni.model_executor.models.minimax_h3.reference_video import _probe_video

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    MINIMAX_H3_TOKEN_ID_NATIVE_KEY,
    MiniMaxH3RolloutWeightSyncMixin,
    minimax_h3_short_video_requests,
    ref2va_reference_image_short_edge,
    validate_ref2va_reference_image_short_edge,
)
from verl_omni.utils.tracking import _export_video


class H3SFTInferencePipeline(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3Pipeline):
    """Load converted H3 LoRA factors into the native inference projections."""

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._install_lora_layout()

    def _load_diffusion_lora_adapter(self, *, lora_request, lora_path, dtype):
        adapter_dir = Path(lora_path)
        manifest = json.loads((adapter_dir / "conversion_manifest.json").read_text())
        if manifest.get("format") not in {"minimax_h3_sft_lora", "minimax_h3_rl_lora"}:
            raise ValueError("Expected a converted H3 SFT or exported RL adapter.")
        if manifest.get("base_weights_merged") is not False:
            raise ValueError("The LoRA must be separate from the official H3 base.")
        config = json.loads((adapter_dir / "adapter_config.json").read_text())
        tensors = load_file(str(adapter_dir / "pytorch_lora_weights.safetensors"), device="cpu")
        tensors, config = self.map_lora_update_to_engine(tensors, config)
        helper = PEFTHelper.from_dict(config)
        model = LoRAModel.from_lora_tensors(
            lora_model_id=lora_request.lora_int_id,
            tensors=tensors,
            peft_helper=helper,
            device="cpu",
            dtype=dtype,
            weights_mapper=None,
        )
        return model, helper

    def _validate_diffusion_lora_binding(self, *, lora_model, bound_lora_names) -> None:
        missing = set(lora_model.loras) - bound_lora_names
        if missing:
            raise ValueError(f"H3 adapter has unbound projections: {sorted(missing)[:8]}")

    def _prepare_request_inputs(self, raw_prompt, sampling):
        short_edge = sampling.extra_args["reference_image_short_edge"]
        with ref2va_reference_image_short_edge(short_edge), minimax_h3_short_video_requests():
            return super()._prepare_request_inputs(raw_prompt, sampling)

    def _encode_local_media(self, media):
        if media is not None and media.videos:
            processor = self.video_vae.model.processor
            min_frames = 1 + processor.align_video_length(1, mode="pad", granularity="chunk")
            videos = []
            for frames in media.videos:
                if 0 < len(frames) < min_frames:
                    # Only the VAE reference needs a complete first chunk.
                    tail = frames[-1:].expand(min_frames - len(frames), -1, -1, -1)
                    frames = torch.cat((frames, tail))
                videos.append(frames)
            media = replace(media, videos=tuple(videos))
        return super()._encode_local_media(media)

    def forward(self, request):
        """Return decoded media without collecting RL training tensors."""
        # LoRA stacks are plain tensor tuples, outside the base-weight offload lifecycle.
        # The worker activates the adapter before entering this pipeline forward.
        if request.sampling_params.lora_request is not None:
            for module in self.transformer.modules():
                if isinstance(module, DiffusionBaseLinearLayerWithLoRA):
                    module._set_diffusion_lora_buffer_device(self.device)
        self._ensure_prompt_text(request)
        try:
            extra = request.sampling_params.extra_args
            context = (
                ref2va_reference_image_short_edge(extra["reference_image_short_edge"])
                if extra.get(MINIMAX_H3_TOKEN_ID_NATIVE_KEY)
                else nullcontext()
            )
            with context:
                return super().forward(request)
        finally:
            self._h3_prompt_ids = None


def _validation_indices(value: str) -> list[int]:
    try:
        indices = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integer validation row indices.") from exc
    if len(set(indices)) != len(indices) or min(indices) < 0:
        raise argparse.ArgumentTypeError("Validation row indices must be unique and nonnegative.")
    return indices


def _archive_validation_source(source: Path, source_dir: Path) -> Path:
    stat = source.stat()
    identity = f"{source}\0{stat.st_size}\0{stat.st_mtime_ns}"
    name = hashlib.sha256(os.fsencode(identity)).hexdigest() + source.suffix
    saved = source_dir / name
    if not saved.is_file():
        # Publish a complete copy even when two replicas select the same source.
        with tempfile.TemporaryDirectory(dir=source_dir) as work:
            pending = Path(work) / name
            shutil.copyfile(source, pending)
            pending.replace(saved)
    return saved


async def _score_validation(frames: np.ndarray, info: dict, fps: float, url: str, weights: list[float]) -> dict:
    from verl_omni.utils.reward_score.video_edit_reward import compute_score

    try:
        return await compute_score(
            solution_image=torch.from_numpy(frames).permute(0, 3, 1, 2),
            extra_info={**info, "media_kind": "video", "fps": fps},
            server_url=url,
            fps=fps,
            score_weights=weights,
        )
    finally:
        session = getattr(compute_score, "_session", None)
        if session is not None and not session.closed:
            await session.close()


def main() -> None:
    """Generate prepared V6 rows and optionally score checkpoint validation outputs."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-path", type=Path, required=True, help="Official H3 root containing Ref2VA/.")
    parser.add_argument("--lora-path", type=Path, help="Converted SFT or exported RL adapter.")
    parser.add_argument("--base-only", action="store_true", help="Preview the official base without SFT LoRA.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for this preview or GPU group.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of consecutive rows; 0 means all remaining.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--validation-indices", type=_validation_indices, help="Comma-separated held-out row indices.")
    parser.add_argument("--validation-source-dir", type=Path, help="Shared directory for validation input videos.")
    parser.add_argument("--reward-url", help="VideoReward /score endpoint for offline validation.")
    parser.add_argument("--score-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("IF", "SF", "VQ"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--text-encoder-tp-size", type=int, default=2)
    parser.add_argument("--reference-image-short-edge", type=int, default=2048)
    parser.add_argument("--attention-backend")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or path for saving MP4 videos.")
    args = parser.parse_args()
    validate = args.validation_indices is not None
    if validate != bool(args.reward_url):
        parser.error("Use --validation-indices and --reward-url together.")
    if args.validation_source_dir is not None and not validate:
        parser.error("--validation-source-dir requires --validation-indices.")
    if validate and (args.base_only or args.lora_path is None):
        parser.error("Offline validation requires --lora-path pointing to an exported RL adapter.")
    if any(not math.isfinite(w) or w < 0 for w in args.score_weights) or not 0 < sum(args.score_weights) < math.inf:
        parser.error("--score-weights must be finite, nonnegative and have a positive finite sum.")
    if args.attention_backend is None:
        args.attention_backend = "TORCH_SDPA" if validate else "FLASH_ATTN_3_HUB"
    ffmpeg_exe = shutil.which(args.ffmpeg)
    if ffmpeg_exe is None:
        parser.error("FFmpeg is unavailable; add it to PATH or pass --ffmpeg /path/to/ffmpeg.")
    if shutil.which("ffprobe") is None:
        parser.error("FFprobe must be on PATH to read each source video's frame count and FPS.")
    if args.num_samples < 0 or args.start_index < 0 or args.seed < 0 or args.num_inference_steps < 2:
        parser.error("Sample count, start index and seed must be nonnegative; inference steps must be at least 2.")
    if args.tensor_parallel_size not in (1, 2, 4, 8):
        parser.error("--tensor-parallel-size must be one of 1, 2, 4, 8.")
    if args.text_encoder_tp_size not in (1, args.tensor_parallel_size):
        parser.error("Text encoder TP must be 1 or equal to tensor parallel size.")
    validate_ref2va_reference_image_short_edge(args.reference_image_short_edge)
    if not args.base_only and args.lora_path is None:
        parser.error("Provide --lora-path for the SFT policy, or explicitly select --base-only.")

    model_path = args.model_path.expanduser().resolve()
    data_file = args.data_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    lora_path = None if args.base_only else args.lora_path.expanduser().resolve()
    if not (model_path / "Ref2VA").is_dir():
        raise FileNotFoundError(f"H3 Ref2VA weights are missing: {model_path / 'Ref2VA'}")
    if lora_path is not None:
        adapter_manifest = json.loads((lora_path / "conversion_manifest.json").read_text())
        if Path(adapter_manifest["base_model"]).resolve() != model_path:
            raise ValueError("The adapter was prepared for a different H3 base.")
        if validate and adapter_manifest.get("format") != "minimax_h3_rl_lora":
            raise ValueError("Offline checkpoint validation requires an exported RL adapter.")
        if validate and (
            adapter_manifest.get("adapter") not in {"default", "old"}
            or type(adapter_manifest.get("step")) is not int
            or adapter_manifest["step"] < 0
        ):
            raise ValueError("The RL adapter manifest must identify its checkpoint step and policy.")
    data_manifest = json.loads((data_file.parent / "conversion_manifest.json").read_text())
    if data_manifest.get("format") != "video_edit_v6":
        raise ValueError("Use the parquet dataset produced by prepare_video_edit_v6_data.py.")
    table = pq.read_table(data_file, columns=["prompt", "videos", "extra_info"])
    indices = (
        args.validation_indices
        if validate
        else range(args.start_index, min(table.num_rows, args.start_index + (args.num_samples or table.num_rows)))
    )
    count = len(indices)
    if not count:
        raise ValueError(f"No samples at start index {args.start_index}; dataset has {table.num_rows} rows.")
    if max(indices) >= table.num_rows:
        raise ValueError(f"Row {max(indices)} exceeds the dataset size ({table.num_rows}).")
    rows = table.take(list(indices)).to_pylist()

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "videos").mkdir()
    if validate:
        from transformers import AutoTokenizer

        source_dir = (args.validation_source_dir or output_dir / "sources").expanduser().resolve()
        source_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path / "Ref2VA" / "tokenizer"), local_files_only=True)
        os.environ.setdefault("VLLM_OMNI_DLO_DP_WAVE_TIMEOUT", "10800")
        os.environ.setdefault("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", "3600")
    else:
        (output_dir / "sources").mkdir()
    settings = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_path": str(model_path),
        "lora_path": str(lora_path) if lora_path else None,
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "timing_policy": "full V6 rollout_shape" if validate else "align source to 17n+5 and trim the alignment tail",
        "canvas_policy": "per-row V6 rollout_shape" if validate else "per-row V6 source_height/source_width",
        "minimum_generation_request_frames": 1,
        "selected_samples": count,
    }
    (output_dir / "run_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    lora_request = LoRARequest("h3-rl-v6" if validate else "h3-sft-v6", 1, str(lora_path)) if lora_path else None
    print(f"Generating {count} samples into {output_dir}", flush=True)
    engine = Omni(
        model=str(model_path / "Ref2VA"),
        trust_remote_code=True,
        task_type="ref2va",
        dtype="bfloat16",
        num_gpus=args.tensor_parallel_size,
        parallel_config={
            "tensor_parallel_size": args.tensor_parallel_size,
            "text_encoder_tp_size": args.text_encoder_tp_size,
        },
        custom_pipeline_args={
            "pipeline_class": (
                "examples.diffusionnft_trainer.minimax_h3.infer_minimax_h3_video_edit.H3SFTInferencePipeline"
            ),
        },
        max_num_seqs=1,
        **(
            {"diffusion_offload_config": {"mode": "layer", "components": ["dit", "text_encoder"], "pin_memory": False}}
            if validate
            else {"enable_layerwise_offload": True}
        ),
        diffusion_attention_backend=args.attention_backend,
        step_execution=False,
        max_cpu_loras=1,
        init_timeout=1800,
        stage_init_timeout=1800,
    )
    try:
        with nullcontext() if validate else (output_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            for offset, (index, row) in enumerate(zip(indices, rows, strict=True)):
                info = row["extra_info"]
                messages = row["prompt"]
                if len(messages) != 1 or messages[0]["role"] != "user":
                    raise ValueError(f"Row {index} must contain one user instruction.")
                content = messages[0]["content"]
                if not isinstance(content, str) or not content.startswith("<video>"):
                    raise ValueError(f"Row {index} must start with one <video> placeholder.")
                prompt = content.removeprefix("<video>")
                references = row["videos"]
                if len(references) != 1:
                    raise ValueError(f"Row {index} must contain exactly one source video.")
                source = Path(references[0]["video"]).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"Row {index} source video is unavailable: {source}")
                start_time = references[0].get("start_time_seconds") or 0.0
                if start_time != 0.0:
                    raise ValueError("Source-length previews require the complete source video without an offset.")
                if validate:
                    shape = info["rollout_shape"]
                    requested_frames = shape["num_frames"]
                    height, width, fps = shape["height"], shape["width"], shape["frame_rate"]
                    if (
                        requested_frames % 17 != 5
                        or height % 32
                        or width % 32
                        or min(requested_frames, height, width) < 1
                    ):
                        raise ValueError(f"Row {index} has invalid H3 rollout shape: {shape}")
                else:
                    source_metadata = _probe_video(str(source))
                    fps = source_metadata["fps"]
                    source_frames = source_metadata["frame_count"]
                    if source_frames <= 0:
                        raise ValueError(f"Row {index} source video has no frames.")
                    requested_frames = source_frames
                    height, width = info["source_height"], info["source_width"]
                if fps != MINIMAX_H3_FPS:
                    raise ValueError(f"Row {index} has {fps} FPS; prepared V6 videos require 24 FPS.")
                generation_frames = minimax_h3_align_frame_count(requested_frames)
                print(f"Row {index}: generating {generation_frames} frames at {fps:g} FPS", flush=True)
                seed = args.seed if validate else args.seed + index
                engine_prompt = {
                    "prompt": prompt,
                    "multi_modal_data": {"video": [{"path": str(source), "start_time_seconds": start_time}]},
                }
                if validate:
                    engine_prompt["prompt_ids"] = tokenizer(
                        prompt.strip(), add_special_tokens=False, truncation=True, max_length=12288
                    )["input_ids"]
                    if references[0].get("start_time_seconds") is None:
                        engine_prompt["multi_modal_data"] = {"video": [str(source)]}
                sampling = OmniDiffusionSamplingParams(
                    seed=seed,
                    num_outputs_per_prompt=1,
                    num_inference_steps=args.num_inference_steps,
                    height=height,
                    width=width,
                    num_frames=requested_frames,
                    fps=fps,
                    frame_rate=fps,
                    true_cfg_scale=1.0,
                    max_sequence_length=12288,
                    output_type="np",
                    return_frames=True,
                    lora_request=lora_request,
                    lora_scale=1.0,
                    extra_args={
                        "task": "ref2va",
                        # Avoid upstream's num_frames=1 sentinel for its default duration.
                        "duration_seconds": requested_frames / fps,
                        "video_flow_shift": 12.0,
                        "reference_image_short_edge": args.reference_image_short_edge,
                        **({MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True} if validate else {}),
                    },
                )
                results = engine.generate(
                    engine_prompt,
                    sampling_params_list=[sampling],
                    use_tqdm=False,
                )
                if len(results) != 1 or len(results[0].images) != 1:
                    raise RuntimeError(f"Expected one generated video for row {index}.")
                frames = results[0].images[0]
                if isinstance(frames, torch.Tensor):
                    frames = frames.detach().cpu().numpy()
                if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4:
                    raise RuntimeError("H3 must return decoded uint8 video with shape [T, H, W, C].")
                if frames.shape[-1] != 3:
                    raise RuntimeError(f"Expected RGB video, got {frames.shape}.")
                if frames.shape[0] != generation_frames:
                    raise RuntimeError(f"Expected {generation_frames} generated frames, got {frames.shape[0]}.")
                if validate and frames.shape[1:3] != (height, width):
                    raise RuntimeError(f"Row {index} returned {frames.shape[1:3]} instead of {(height, width)}.")
                if not validate:
                    # Preview only the source timeline. Score the full decoded validation rollout.
                    frames = frames[:source_frames]
                name = f"{index:06d}"
                generated = output_dir / "videos" / f"{name}.mp4"
                _export_video(
                    torch.from_numpy(frames).permute(0, 3, 1, 2),
                    str(generated),
                    fps=int(fps),
                    ffmpeg_exe=ffmpeg_exe,
                )
                record = {
                    "row_index": index,
                    "sample_id": info["sample_id"],
                    "prompt": prompt,
                    "edit_instruction": info["edit_instruction"],
                    "instruction_task": info["instruction_task"],
                    "source_video": str(source),
                    "generated_video": str(generated.relative_to(output_dir)),
                    "seed": seed,
                    "generation_request_frames": requested_frames,
                    "generated_num_frames": generation_frames,
                    "trimmed_tail_frames": generation_frames - len(frames),
                    "sft_padded_num_frames": info["num_frames"],
                    "output_shape": list(frames.shape),
                    "fps": fps,
                }
                if validate:
                    archived = _archive_validation_source(source, source_dir)
                    scores = asyncio.run(_score_validation(frames, info, fps, args.reward_url, args.score_weights))
                    record.update(
                        uid=info["sample_id"],
                        source_preview=os.path.relpath(archived, output_dir),
                        rollout_shape=shape,
                        num_inference_steps=args.num_inference_steps,
                        adapter_name=adapter_manifest["adapter"],
                        step=adapter_manifest["step"],
                        **scores,
                    )
                else:
                    source_link = output_dir / "sources" / f"{name}{source.suffix}"
                    source_link.symlink_to(source)
                    record.update(
                        source_preview=str(source_link.relative_to(output_dir)),
                        source_num_frames=source_frames,
                        source_duration_seconds=source_frames / fps,
                    )
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                serialized = json.dumps(record, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
                generated.with_suffix(".json").write_text(serialized, encoding="utf-8")
                if validate:
                    row_path = output_dir / f"row_{index}.json"
                    pending = row_path.with_suffix(".json.tmp")
                    pending.write_text(serialized, encoding="utf-8")
                    pending.replace(row_path)
                    print(f"[{offset + 1}/{count}] Scored {row_path}: {scores}", flush=True)
                print(f"[{offset + 1}/{count}] Saved {generated}", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
