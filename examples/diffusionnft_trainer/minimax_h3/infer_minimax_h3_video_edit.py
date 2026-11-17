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
"""Preview H3's SFT policy on prepared V6 training prompts without starting RL.

Uses vLLM-Omni's MiniMaxH3Pipeline and the same LoRA mapping as the NFT rollout.
For 24 FPS sources, export exactly the source frame count after removing the
extra tail generated for H3's 17n+5 temporal alignment, without a 4-second floor.
Run through run_minimax_h3_video_edit_infer.sh from an H3 training environment.
"""

import argparse
import json
import shutil
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
    MiniMaxH3RolloutWeightSyncMixin,
    minimax_h3_short_video_requests,
    ref2va_reference_image_short_edge,
    validate_ref2va_reference_image_short_edge,
)
from verl_omni.utils.tracking import _export_video


class H3SFTInferencePipeline(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3Pipeline):
    """Load the converted SFT adapter into H3's native inference projections."""

    def __init__(self, *, od_config, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._install_lora_layout()

    def _load_diffusion_lora_adapter(self, *, lora_request, lora_path, dtype):
        adapter_dir = Path(lora_path)
        manifest = json.loads((adapter_dir / "conversion_manifest.json").read_text())
        if manifest.get("format") != "minimax_h3_sft_lora" or manifest.get("base_weights_merged") is not False:
            raise ValueError("Use the separate SFT adapter produced by prepare_video_edit_sft_lora.py.")
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
            raise ValueError(f"SFT adapter has unbound projections: {sorted(missing)[:8]}")

    def _prepare_request_inputs(self, raw_prompt, sampling):
        short_edge = sampling.extra_args["reference_image_short_edge"]
        with ref2va_reference_image_short_edge(short_edge), minimax_h3_short_video_requests():
            return super()._prepare_request_inputs(raw_prompt, sampling)

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
            return super().forward(request)
        finally:
            self._h3_prompt_ids = None


def main() -> None:
    """Generate selected training rows and save videos with their source metadata."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-path", type=Path, required=True, help="Official H3 root containing Ref2VA/.")
    parser.add_argument("--lora-path", type=Path, help="Converted SFT adapter; required unless --base-only.")
    parser.add_argument("--base-only", action="store_true", help="Preview the official base without SFT LoRA.")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this preview run.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of consecutive rows; 0 means all remaining.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--text-encoder-tp-size", type=int, default=2)
    parser.add_argument("--reference-image-short-edge", type=int, default=2048)
    parser.add_argument("--attention-backend", default="FLASH_ATTN_3_HUB")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or path for saving MP4 videos.")
    args = parser.parse_args()
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
            raise ValueError("The converted SFT adapter was prepared for a different H3 base.")
    data_manifest = json.loads((data_file.parent / "conversion_manifest.json").read_text())
    if data_manifest.get("format") != "video_edit_v6":
        raise ValueError("Use the parquet dataset produced by prepare_video_edit_v6_data.py.")
    table = pq.read_table(data_file, columns=["prompt", "videos", "extra_info"])
    count = min(args.num_samples or table.num_rows, table.num_rows - args.start_index)
    if count <= 0:
        raise ValueError(f"No samples at start index {args.start_index}; dataset has {table.num_rows} rows.")
    rows = table.slice(args.start_index, count).to_pylist()

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "videos").mkdir()
    (output_dir / "sources").mkdir()
    settings = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_path": str(model_path),
        "lora_path": str(lora_path) if lora_path else None,
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "timing_policy": "source frame count at 24 FPS; align to 17n+5 and trim only the alignment tail",
        "canvas_policy": "per-row V6 source_height/source_width",
        "minimum_generation_request_frames": 1,
        "selected_samples": count,
    }
    (output_dir / "run_config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    lora_request = LoRARequest("h3-sft-v6", 1, str(lora_path)) if lora_path else None
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
        enable_layerwise_offload=True,
        diffusion_attention_backend=args.attention_backend,
        step_execution=False,
        max_cpu_loras=1,
        init_timeout=1800,
        stage_init_timeout=1800,
    )
    try:
        with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            for offset, row in enumerate(rows):
                index = args.start_index + offset
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
                source_metadata = _probe_video(str(source))
                fps = source_metadata["fps"]
                if fps != MINIMAX_H3_FPS:
                    raise ValueError(
                        f"Row {index} has {fps} FPS; this H3 preview supports 24 FPS sources without resampling."
                    )
                source_frames = source_metadata["frame_count"]
                if source_frames <= 0:
                    raise ValueError(f"Row {index} source video has no frames.")
                requested_frames = source_frames
                generation_frames = minimax_h3_align_frame_count(requested_frames)
                print(
                    f"Row {index}: source={source_frames} frames at {fps:g} FPS; "
                    f"generate={generation_frames}, save={source_frames}",
                    flush=True,
                )
                seed = args.seed + index
                sampling = OmniDiffusionSamplingParams(
                    seed=seed,
                    num_outputs_per_prompt=1,
                    num_inference_steps=args.num_inference_steps,
                    height=int(info["source_height"]),
                    width=int(info["source_width"]),
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
                    },
                )
                results = engine.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"video": [{"path": str(source), "start_time_seconds": start_time}]},
                    },
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
                # Retain the original timeline; remove only the extra generated tail.
                frames = frames[:source_frames]
                name = f"{index:06d}"
                generated = output_dir / "videos" / f"{name}.mp4"
                _export_video(
                    torch.from_numpy(frames).permute(0, 3, 1, 2),
                    str(generated),
                    fps=int(fps),
                    ffmpeg_exe=ffmpeg_exe,
                )
                source_link = output_dir / "sources" / f"{name}{source.suffix}"
                source_link.symlink_to(source)
                record = {
                    "row_index": index,
                    "sample_id": info["sample_id"],
                    "prompt": prompt,
                    "edit_instruction": info["edit_instruction"],
                    "instruction_task": info["instruction_task"],
                    "source_video": str(source),
                    "source_preview": str(source_link.relative_to(output_dir)),
                    "generated_video": str(generated.relative_to(output_dir)),
                    "seed": seed,
                    "source_num_frames": source_frames,
                    "source_duration_seconds": source_frames / fps,
                    "generation_request_frames": requested_frames,
                    "generated_num_frames": generation_frames,
                    "trimmed_tail_frames": generation_frames - source_frames,
                    "sft_padded_num_frames": info["num_frames"],
                    "output_shape": list(frames.shape),
                    "fps": fps,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                (output_dir / "videos" / f"{name}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(f"[{offset + 1}/{count}] Saved {generated}", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
