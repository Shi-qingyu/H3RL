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

"""HTTP adapter for the three-head Qwen3.5 VideoReward video-editing model."""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import aiohttp
import torch
from safetensors.torch import save as save_tensors

from .reward_utils import normalize_video_tensor, video_reward_tensor, visual_reward_frames

_PROTOCOL_VERSION = "1"
_DIMENSIONS = ("instruction_following", "source_fidelity", "visual_quality")
_RESERVED_TOKENS = (
    "<image>",
    "<video>",
    "<|video_reward|>",
    "<|instruction_following_reward|>",
    "<|source_fidelity_reward|>",
    "<|temporal_consistency_reward|>",
    "<|visual_quality_reward|>",
)


class _RetryableHTTPError(RuntimeError):
    pass


def _finite_number(value: Any, name: str, *, minimum: float = 0.0, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number) or number < minimum or (positive and number == minimum):
        raise ValueError(f"{name} must be finite and {'greater than' if positive else 'at least'} {minimum}.")
    return number


def _serialize_request(solution_image, ground_truth: str, extra_info: dict, fps: float) -> bytes:
    from verl_omni.pipelines.rollout_media import ArtifactContractError

    if "media_artifacts" in extra_info:
        video = video_reward_tensor(solution_image, extra_info, fps)
    else:
        if extra_info.get("media_kind") != "video" or extra_info.get("fps") != fps:
            raise ArtifactContractError("VideoEditReward requires decoded video with its explicit matching FPS.")
        video = visual_reward_frames(solution_image, extra_info)
    try:
        video = normalize_video_tensor(video)
    except ValueError as error:
        raise ArtifactContractError(str(error)) from error
    if video.shape[0] < 2:
        raise ArtifactContractError("VideoEditReward requires at least two decoded video frames.")

    source_video = extra_info.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip() or not Path(source_video).is_absolute():
        raise ArtifactContractError("VideoEditReward requires extra_info.source_video as an absolute file path.")
    instruction = extra_info.get("edit_instruction", ground_truth)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ArtifactContractError("VideoEditReward requires a non-empty edit_instruction or ground_truth.")
    if any(token in instruction for token in _RESERVED_TOKENS):
        raise ArtifactContractError(
            "VideoEditReward instructions must not contain media placeholders or reward tokens."
        )
    metadata = json.dumps(
        {
            "protocol_version": _PROTOCOL_VERSION,
            "source_video": source_video,
            "edit_instruction": instruction.strip(),
            "fps": fps,
        },
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    # Preserve all generated frames; the server applies the checkpoint's saved sampling once.
    return save_tensors(
        {
            "generated_video": video.detach().to(device="cpu").contiguous(),
            "metadata": torch.tensor(list(metadata), dtype=torch.uint8),
        }
    )


def _validate_response(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict) or payload.get("protocol_version") != _PROTOCOL_VERSION:
        raise RuntimeError("VideoEditReward returned an invalid response or protocol version.")
    scores = payload.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(_DIMENSIONS):
        raise RuntimeError(f"VideoEditReward must return exactly these score dimensions: {_DIMENSIONS}.")
    result = {}
    for name in _DIMENSIONS:
        value = scores[name]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise RuntimeError(f"VideoEditReward returned a non-finite or invalid {name} score: {value!r}.")
        result[name] = float(value)
    return result


async def _session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    session = getattr(compute_score, "_session", None)
    if session is not None and not session.closed:
        if getattr(compute_score, "_session_loop", None) is not loop:
            raise RuntimeError("VideoEditReward HTTP session cannot be shared across event loops.")
        return session
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    compute_score._session = session
    compute_score._session_loop = loop
    return session


async def _request_score(server_url: str, payload: bytes, timeout: float) -> dict[str, float]:
    session = await _session()
    try:
        async with session.post(
            server_url,
            data=payload,
            headers={"content-type": "application/octet-stream"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                detail = await response.text(errors="replace")
                message = f"VideoEditReward returned HTTP {response.status}: {detail}"
                if response.status in {408, 429} or 500 <= response.status < 600:
                    raise _RetryableHTTPError(message)
                raise RuntimeError(message)
            try:
                result = await response.json(content_type=None)
            except ValueError as error:
                raise RuntimeError("VideoEditReward returned malformed JSON.") from error
    except TimeoutError as error:
        raise _RetryableHTTPError(f"VideoEditReward timed out after {timeout} seconds.") from error
    return _validate_response(result)


async def compute_score(
    data_source: str | None = None,
    solution_image=None,
    ground_truth: str | None = None,
    extra_info: dict | None = None,
    *,
    server_url: str,
    fps: float,
    score_weights=(1.0, 1.0, 1.0),
    timeout: float = 300.0,
    max_retries: int = 2,
    retry_backoff: float = 0.5,
    **kwargs,
) -> dict[str, float]:
    """Return a weighted mean of raw IF/SF/VQ scores plus each head's metric.

    Selects canonical uint8 RGB ``[T, 3, H, W]`` decoded video. ``extra_info``
    supplies ``source_video`` (an absolute path shared with the service) and
    ``edit_instruction``; ``ground_truth`` is the instruction fallback. ``fps``
    must match the generated video's artifact metadata. Raw heads fit 1–5 human
    ratings and are neither clipped nor passed through a sigmoid.
    """
    del data_source, kwargs
    if not isinstance(server_url, str) or not server_url.strip():
        raise ValueError("server_url must be a non-empty URL ending in /score.")
    fps = _finite_number(fps, "fps", positive=True)
    timeout = _finite_number(timeout, "timeout", positive=True)
    retry_backoff = _finite_number(retry_backoff, "retry_backoff")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")
    if score_weights is None or len(score_weights) != len(_DIMENSIONS):
        raise ValueError(f"score_weights must contain three values in {_DIMENSIONS} order.")
    weights = [_finite_number(weight, "score_weights") for weight in score_weights]
    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("score_weights must have a finite positive sum.")
    weights = [weight / total_weight for weight in weights]
    payload = await asyncio.to_thread(_serialize_request, solution_image, ground_truth, extra_info or {}, fps)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            scores = await _request_score(server_url, payload, timeout)
            score = sum(weight * scores[name] for weight, name in zip(weights, _DIMENSIONS, strict=True))
            if not math.isfinite(score):
                raise RuntimeError("VideoEditReward weighted score must be finite.")
            return {"score": score, **scores}
        except (_RetryableHTTPError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as error:
            last_error = error
        if attempt < max_retries:
            await asyncio.sleep(retry_backoff * (2**attempt))
    raise RuntimeError(f"VideoEditReward failed after {max_retries + 1} attempts: {last_error}") from last_error
