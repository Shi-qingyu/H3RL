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
"""Prepare source videos and editing instructions for H3 Ref2VA DiffusionNFT.

Each JSONL record requires ``source_video`` and ``edit_instruction``. An optional
``prompt`` supplies a separate H3 generation prompt; the reward always receives
the original editing instruction. Input paths are relative to ``--input_dir``.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _path(input_dir: Path, value: Any) -> Path:
    path = Path(_text(value, "source_video")).expanduser()
    if not path.is_absolute():
        path = input_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source video not found: {path}")
    return path


def _check_start(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value != 0:
        raise ValueError("Video editing requires start_time_seconds=0 so conditioning and reward use the same source.")


def _source_video(input_dir: Path, example: dict[str, Any]) -> Path:
    source = _path(input_dir, example.get("source_video"))
    _check_start(example.get("start_time_seconds"))
    for field in ("images", "image", "audios", "audio"):
        if example.get(field):
            raise ValueError("Video editing accepts exactly one source video and no other reference media.")
    if "videos" in example and "video" in example:
        raise ValueError("Use only one of video/videos; ambiguous conditioning references are unsupported.")
    references = example.get("videos")
    if "video" in example:
        references = [example["video"]]
    if references is not None:
        if not isinstance(references, list) or len(references) != 1:
            raise ValueError("Video editing requires exactly one conditioning video matching source_video.")
        reference = references[0]
        if isinstance(reference, dict):
            _check_start(reference.get("start_time_seconds"))
            paths = [reference[key] for key in ("path", "video", "video_path") if key in reference]
            if not paths or any(_path(input_dir, path) != source for path in paths):
                raise ValueError("The conditioning video must match source_video.")
        elif _path(input_dir, reference) != source:
            raise ValueError("The conditioning video must match source_video.")
    return source


def _convert_split(input_dir: Path, split: str, max_samples: int) -> pd.DataFrame:
    rows = []
    with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            if not isinstance(example, dict):
                raise ValueError(f"{split} row {index} must be a JSON object.")
            instruction = _text(example.get("edit_instruction"), "edit_instruction")
            if any(marker in instruction for marker in ("<|", "<image>", "<video>", "<audio>")):
                raise ValueError("edit_instruction must contain plain editing text without model/media tokens.")
            prompt = _text(example.get("prompt", instruction), "prompt")
            if any(marker in prompt for marker in ("<image>", "<video>", "<audio>")):
                raise ValueError("prompt must omit media placeholders; the source-video placeholder is added here.")
            source = str(_source_video(input_dir, example))
            rows.append(
                {
                    "data_source": "minimax_h3_video_edit",
                    "prompt": [{"role": "user", "content": f"<video>{prompt}"}],
                    "ability": "video_editing",
                    "videos": [{"video": source, "start_time_seconds": None}],
                    "reward_model": {"style": "model", "ground_truth": instruction},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "source_video": source,
                        "source_videos": [source],
                        "edit_instruction": instruction,
                    },
                }
            )
    if not rows:
        raise ValueError(f"{split} contains no selected training examples.")
    return pd.DataFrame(rows)


def main() -> None:
    """Write train/test parquet files consumed by the video-editing recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    parser.add_argument("--val_split", choices=("test", "val"), default="test")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    train = _convert_split(input_dir, "train", args.train_size)
    validation = _convert_split(input_dir, args.val_split, args.val_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(args.output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
