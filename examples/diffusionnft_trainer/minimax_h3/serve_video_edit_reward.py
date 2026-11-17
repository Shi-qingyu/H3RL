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

"""Serve the three-dimensional editing reward in the separate VideoReward environment.

The model loader and scoring flow are adapted from the local VideoReward project's
``video_reward/inference/api.py``. This service restores the checkpoint's three
compact heads and accepts generated uint8 TCHW video through safetensors.
"""

import argparse
import json
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from unittest.mock import patch

DEFAULT_SOURCE_ROOT = Path("/mnt/geminihzceph/qingyushi/projects/reward_model/VideoReward")
DEFAULT_CHECKPOINT = DEFAULT_SOURCE_ROOT / (
    "checkpoints/three_dim_removal_two_instructions_20261109_112935/"
    "trials/two_instructions/removal/train/checkpoint-288"
)
DIMENSIONS = ("instruction_following", "source_fidelity", "visual_quality")
SAMPLING_ENV = {
    "fps": "FPS",
    "min_frames": "FPS_MIN_FRAMES",
    "max_frames": "FPS_MAX_FRAMES",
    "min_pixels": "VIDEO_MIN_PIXELS",
    "max_pixels": "VIDEO_MAX_PIXELS",
    "total_pixels": "VIDEO_TOTAL_PIXELS",
}
logger = logging.getLogger(__name__)


class VideoEditRewardScorer:
    """Restore the backbone, LoRA, reward tokens, and saved compact scalar heads."""

    def __init__(self, source_root, checkpoint, *, base_model=None, device="cuda:0", dtype=None, attn_impl=None):
        source_root = Path(source_root).expanduser().resolve()
        if not (source_root / "swift/model/models/qwen.py").is_file():
            raise FileNotFoundError(f"VideoReward's customized Swift source is unavailable: {source_root}")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        for name in ("config.json", "args.json", "adapter_config.json", "adapter_model.safetensors"):
            if not (self.checkpoint / name).is_file():
                raise FileNotFoundError(f"Missing checkpoint file: {self.checkpoint / name}")
        head = json.loads((self.checkpoint / "config.json").read_text(encoding="utf-8"))
        saved = json.loads((self.checkpoint / "args.json").read_text(encoding="utf-8"))
        adapter = json.loads((self.checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        if tuple(head.get("video_reward_active_dimensions", ())) != DIMENSIONS:
            raise ValueError(f"Checkpoint must have reward dimensions in this order: {DIMENSIONS}")
        self.preprocessing = head.get("video_reward_pointwise_ablation", {}).get("video_preprocessing", {})
        for key in SAMPLING_ENV:
            value = self.preprocessing.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Checkpoint has no valid saved video preprocessing value for {key!r}.")

        # Qwen's video utilities read their environment defaults when imported.
        os.environ.update({env: str(self.preprocessing[key]) for key, env in SAMPLING_ENV.items()})
        os.environ.update(
            VIDEO_MIN_TOKEN_NUM=str(math.ceil(self.preprocessing["min_pixels"] / 1024)),
            VIDEO_MAX_TOKEN_NUM=str(self.preprocessing["max_pixels"] // 1024),
            FORCE_QWENVL_VIDEO_READER="decord",
            SWIFT_SINGLE_DEVICE_MODE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        sys.path.insert(0, str(source_root))
        import torch
        from peft import PeftModel
        from swift.model import get_model_processor
        from swift.model.models.qwen import Qwen3_5VideoRewardLoader
        from swift.template import get_template
        from transformers import set_seed

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This VideoReward checkpoint requires a CUDA GPU.")
        torch.cuda.set_device(self.device)
        set_seed(saved.get("seed", 42))
        backbone = Path(base_model or saved.get("model") or adapter["base_model_name_or_path"]).expanduser()
        if not backbone.is_absolute():
            backbone = source_root / backbone
        if not (backbone / "config.json").is_file():
            raise FileNotFoundError(f"Base model is unavailable: {backbone}; pass --base-model.")
        model_dtype = dtype or saved.get("torch_dtype") or "float32"
        if model_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"Unsupported model weight dtype: {model_dtype}")
        original_get_model = Qwen3_5VideoRewardLoader.get_model

        def with_saved_heads(loader, model_dir, config, processor, model_kwargs):
            for key, value in head.items():
                if key.startswith("video_reward_"):
                    setattr(config, key, value)
            config.use_cache = False
            return original_get_model(loader, model_dir, config, processor, model_kwargs)

        # PEFT restores weights after the loader constructs the saved head shapes.
        with patch.object(Qwen3_5VideoRewardLoader, "get_model", with_saved_heads):
            model, processor = get_model_processor(
                str(backbone),
                model_type="qwen3_5_video_reward",
                task_type="seq_cls",
                num_labels=len(DIMENSIONS),
                video_reward_separate_heads=head.get("video_reward_separate_heads"),
                torch_dtype=getattr(torch, model_dtype),
                device_map={"": str(self.device)},
                attn_impl=attn_impl or saved.get("attn_impl") or "flash_attn",
            )
        if model.config.video_reward_token_ids != head["video_reward_token_ids"]:
            raise ValueError("Base tokenizer and checkpoint reward-token IDs differ.")
        model.requires_grad_(False)
        self.model = PeftModel.from_pretrained(model, str(self.checkpoint), is_trainable=False)
        self.model.eval()
        self.template = get_template(
            processor,
            template_type=head["video_reward_template"],
            default_system=head["video_reward_system_prompt"],
            max_length=saved.get("max_length"),
            truncation_strategy="raise",
            padding_side="right",
            padding_free=False,
        )
        self.template.set_mode("transformers")
        self.template.model = self.model
        self.template.max_pixels = None
        self.lock = Lock()
        logger.info("Loaded %s on %s", self.checkpoint, self.device)

    def decode_request(self, body: bytes):
        """Validate safetensors containing uint8 TCHW video and UTF-8 JSON metadata."""
        from safetensors.torch import load

        try:
            tensors = load(body)
        except Exception as error:
            raise ValueError("Request body must be a valid safetensors document.") from error
        if set(tensors) != {"generated_video", "metadata"}:
            raise ValueError("Request tensors must be exactly generated_video and metadata.")
        video, metadata = tensors["generated_video"], tensors["metadata"]
        if video.dtype != self.torch.uint8 or video.ndim != 4:
            raise ValueError("generated_video must be a uint8 tensor with shape [T, 3, H, W].")
        if video.shape[0] < 2 or video.shape[1] != 3 or min(video.shape[2:]) <= 0:
            raise ValueError("generated_video must contain at least two nonempty RGB frames.")
        if metadata.dtype != self.torch.uint8 or metadata.ndim != 1 or not 0 < metadata.numel() <= 1024 * 1024:
            raise ValueError("metadata must be a nonempty uint8 UTF-8 JSON vector of at most 1 MiB.")
        try:
            metadata = json.loads(metadata.numpy().tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("metadata must contain valid UTF-8 JSON.") from error
        required_metadata = {"protocol_version", "source_video", "edit_instruction", "fps"}
        if not isinstance(metadata, dict) or set(metadata) != required_metadata:
            raise ValueError("metadata requires exactly protocol_version, source_video, edit_instruction, and fps.")
        if metadata["protocol_version"] != "1":
            raise ValueError("Unsupported protocol_version; expected '1'.")
        source = metadata["source_video"]
        if not isinstance(source, str) or not source.strip() or not Path(source).is_absolute():
            raise ValueError("source_video must be an absolute path visible to the reward server.")
        if not Path(source).is_file():
            raise ValueError(f"Source video is unavailable to the reward server: {source}")
        instruction = metadata["edit_instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("edit_instruction must be a nonempty string.")
        reserved = (
            "<image>",
            "<video>",
            "<|video_reward|>",
            "<|instruction_following_reward|>",
            "<|source_fidelity_reward|>",
            "<|temporal_consistency_reward|>",
            "<|visual_quality_reward|>",
        )
        if any(token in instruction for token in reserved):
            raise ValueError("edit_instruction must not contain media placeholders or reward tokens.")
        fps = metadata["fps"]
        if isinstance(fps, bool) or not isinstance(fps, int | float):
            raise ValueError("fps must be a finite positive number.")
        try:
            fps = float(fps)
        except OverflowError as error:
            raise ValueError("fps must be a finite positive number.") from error
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number.")
        return video, source, instruction.strip(), fps

    def score(self, video, source_video: str, instruction: str, fps: float) -> dict[str, float]:
        """Score generated uint8 [T, 3, H, W] frames at their original playback FPS."""
        import imageio.v2 as imageio
        from swift.utils import to_device
        from video_reward.models.input import build_video_reward_input

        with TemporaryDirectory(prefix="verl_video_edit_reward_") as temporary:
            edited_video = Path(temporary) / "edited.mp4"
            with imageio.get_writer(
                str(edited_video),
                format="FFMPEG",
                fps=fps,
                codec="libx264rgb",
                quality=None,
                ffmpeg_params=["-crf", "0", "-preset", "ultrafast"],
                pixelformat="rgb24",
                macro_block_size=1,
            ) as writer:
                for frame in video:
                    writer.append_data(frame.permute(1, 2, 0).contiguous().numpy())
            value = build_video_reward_input(source_video, str(edited_video), instruction, media_type="video")
            value["chat_template_kwargs"] = dict(self.preprocessing)
            with self.lock:
                self.torch.cuda.set_device(self.device)
                with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                    encoded = self.template.encode(value)
                    inputs = self.template.data_collator([encoded])
                    inputs.pop("labels", None)
                    inputs = to_device(inputs, self.device)
                    if self.template.model_meta.is_multimodal:
                        _, inputs = self.template.pre_forward_hook(self.model, None, inputs)
                    outputs = self.model(**inputs, use_cache=False)
                    if tuple(outputs.logits.shape) != (1, len(DIMENSIONS)):
                        raise RuntimeError(
                            f"Expected [1, {len(DIMENSIONS)}] logits, got {tuple(outputs.logits.shape)}."
                        )
                    scores = dict(zip(DIMENSIONS, outputs.logits[0].float().cpu().tolist(), strict=True))
                    if not all(math.isfinite(value) for value in scores.values()):
                        raise RuntimeError("Model returned a non-finite reward score.")
                    return scores


def create_app(
    source_root=DEFAULT_SOURCE_ROOT,
    checkpoint=DEFAULT_CHECKPOINT,
    *,
    base_model=None,
    device="cuda:0",
    dtype=None,
    attn_impl=None,
    max_request_bytes=512 * 1024 * 1024,
    scorer_factory=VideoEditRewardScorer,
):
    """Create one persistent model service with a bounded safetensors request body."""
    from fastapi import FastAPI, HTTPException, Request
    from starlette.concurrency import run_in_threadpool

    if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int) or max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be a positive integer.")

    @asynccontextmanager
    async def lifespan(app):
        app.state.scorer = scorer_factory(
            source_root, checkpoint, base_model=base_model, device=device, dtype=dtype, attn_impl=attn_impl
        )
        yield
        del app.state.scorer

    app = FastAPI(title="Video editing reward", version="1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "protocol_version": "1",
            "checkpoint": str(app.state.scorer.checkpoint),
            "dimensions": list(DIMENSIONS),
        }

    @app.post("/score")
    async def score(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/octet-stream":
            raise HTTPException(status_code=422, detail="Content-Type must be application/octet-stream.")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                content_length = int(content_length)
            except ValueError as error:
                raise HTTPException(status_code=422, detail="Invalid Content-Length.") from error
            if content_length < 0:
                raise HTTPException(status_code=422, detail="Invalid Content-Length.")
            if content_length > max_request_bytes:
                raise HTTPException(status_code=413, detail="Request exceeds max_request_bytes.")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_request_bytes:
                raise HTTPException(status_code=413, detail="Request exceeds max_request_bytes.")
            body.extend(chunk)
        try:
            video, source, instruction, fps = await run_in_threadpool(app.state.scorer.decode_request, bytes(body))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            scores = await run_in_threadpool(app.state.scorer.score, video, source, instruction, fps)
        except Exception as error:
            logger.exception("VideoReward inference failed")
            raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error
        return {"protocol_version": "1", "scores": scores}

    return app


def main():
    """Launch with the Python environment installed for the VideoReward project."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, help="Override the local Qwen3.5 backbone directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), help="Default: saved checkpoint weight dtype.")
    parser.add_argument("--attn-impl", choices=("flash_attn", "sdpa", "eager"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19085)
    parser.add_argument("--max-request-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.max_request_bytes <= 0:
        parser.error("--max-request-bytes must be positive.")
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    app = create_app(
        args.source_root,
        args.checkpoint,
        base_model=args.base_model,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        max_request_bytes=args.max_request_bytes,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
