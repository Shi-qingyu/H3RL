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
"""Prepare all V6 passerby-removal shapes for online H3 DiffusionNFT.

Join the released cache metadata to frozen source-video/IR records without
loading cached tensors or SFT target videos. The two reward-instruction templates
and task classification are adapted from VideoReward's
``video_reward/data/normalize_remove_instructions.py``, including the shortened
remove-all declaration used by V6. Reworded IR can use its provenance-bound
original instruction for task classification. H3 keeps its frozen IR.
"""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

_DATASET = "passerby_remove"
_KEEP_MAIN = (
    "Remove the unwanted passersby, including their shadows and reflections, from the video "
    "and naturally fill the removed regions. Preserve the main subject, other scene content, "
    "and camera motion."
)
_REMOVE_ALL = (
    "Remove all people, including their shadows and reflections, from the video "
    "and naturally fill the removed regions. Preserve all nonhuman objects, other "
    "scene content, and camera motion."
)


class _UnknownTaskError(ValueError):
    """An instruction has no recognized explicit task declaration."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}.")
            rows.append(row)
    return rows


def _by_id(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in indexed:
            raise ValueError(f"Missing or duplicate ID {identifier!r} in {path}.")
        indexed[identifier] = row
    return indexed


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _cache_path(row: dict[str, Any]) -> Path:
    value = row.get("cache", {}).get("cache_path")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Release ID {row['id']} has no absolute cache_path.")
    path = Path(value)
    lane = row.get("lane")
    data_id = row.get("data_id")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (lane, data_id)):
        raise ValueError(f"Release ID {row['id']} has invalid lane/data_id metadata.")
    if (
        path.name != f"{data_id}.pth"
        or path.parent.name != "0"
        or path.parent.parent.name != f"lane-{lane:03d}"
        or path.parent.parent.parent.name != "full"
    ):
        raise ValueError(f"Unsupported V6 native cache layout for {row['id']}: {path}")
    return path


def _release_rows(release_dir: Path) -> list[dict[str, Any]]:
    metadata = release_dir / "task_metadata.jsonl"
    rows = _read_jsonl(metadata)
    _by_id(rows, metadata)
    index_path = release_dir / "cache_index.tsv"
    lengths = {}
    with index_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) <= 0:
                raise ValueError(f"Invalid cache index row at {index_path}:{line_number}.")
            length, path = int(fields[0]), fields[1]
            if path in lengths or not Path(path).is_absolute():
                raise ValueError(f"Invalid or duplicate cache path at {index_path}:{line_number}.")
            lengths[path] = length
    seen = set()
    for row in rows:
        path = str(_cache_path(row))
        if path in seen or lengths.get(path) != row["cache"].get("sequence_length"):
            raise ValueError(f"Release metadata/index mismatch for {row['id']}.")
        seen.add(path)
    if seen != set(lengths):
        raise ValueError("task_metadata.jsonl and cache_index.tsv identify different cache sets.")
    return rows


def _task(instruction: str) -> tuple[str, str]:
    if instruction == _KEEP_MAIN:
        return "keep_main", _KEEP_MAIN
    if instruction == _REMOVE_ALL:
        return "remove_all", _REMOVE_ALL
    keep = bool(re.search(r"^(?:\[video editing\]\s*)?Keep only this main person:", instruction, re.MULTILINE))
    remove = bool(
        re.search(
            r"^(?:\[video editing\]\s*)?Remove every real person(?:, including distant pedestrians and riders)?\.",
            instruction,
            re.MULTILINE,
        )
    )
    if keep == remove:
        message = "Expected exactly one explicit keep-main or remove-all task in the frozen IR instruction."
        if keep:
            raise ValueError(message)
        raise _UnknownTaskError(message)
    return ("keep_main", _KEEP_MAIN) if keep else ("remove_all", _REMOVE_ALL)


def _instruction_task(
    record: dict[str, Any], selection: dict[str, Any], source_lines: dict[Path, list[bytes]]
) -> tuple[str, str, str, str | None]:
    """Resolve reworded IR through its bound original instruction when needed."""
    try:
        task, instruction = _task(record["instruction"])
        return task, instruction, "frozen_instruction", None
    except _UnknownTaskError:
        pass

    manifest = selection.get("input_manifest")
    line_number = selection.get("input_line")
    if not isinstance(manifest, str) or not Path(manifest).is_absolute():
        raise ValueError("An unrecognized IR instruction requires an absolute provenance input_manifest.")
    if isinstance(line_number, bool) or not isinstance(line_number, int) or line_number <= 0:
        raise ValueError("An unrecognized IR instruction requires a positive provenance input_line.")
    path = Path(manifest)
    if path not in source_lines:
        try:
            source_lines[path] = path.read_bytes().splitlines(keepends=True)
        except OSError as error:
            raise ValueError(f"Cannot read provenance input_manifest {path}: {error}") from error
    lines = source_lines[path]
    if line_number > len(lines):
        raise ValueError(f"Provenance input_line {line_number} is outside {path}.")
    raw_line = lines[line_number - 1]
    if hashlib.sha256(raw_line).hexdigest() != selection.get("input_record_sha256"):
        raise ValueError(f"Provenance source-line hash mismatch at {path}:{line_number}.")
    source = json.loads(raw_line)
    if not isinstance(source, dict) or not isinstance(selection.get("record_id"), str) or not selection["record_id"]:
        raise ValueError("Provenance requires a source object and record_id.")
    if source.get("uid") != selection["record_id"]:
        raise ValueError(f"Provenance record_id does not match the source uid at {path}:{line_number}.")
    if source.get("training_instruction") != record["instruction"]:
        raise ValueError(f"The frozen IR differs from its source training_instruction at {path}:{line_number}.")
    original = source.get("original_instruction")
    if not isinstance(original, str) or not original.strip():
        raise ValueError(f"No original_instruction for the reworded IR at {path}:{line_number}.")
    task, instruction = _task(original)
    return task, instruction, "source_original_instruction", original


def _source(row: dict[str, Any]) -> str:
    references = row.get("references")
    if not isinstance(references, list) or len(references) != 1 or references[0].get("type") != "video":
        raise ValueError(f"Frozen ID {row['id']} must have exactly one source-video reference.")
    reference = references[0]
    if reference.get("start_time_seconds") not in (None, 0):
        raise ValueError(f"Frozen ID {row['id']} requires an unsupported nonzero source offset.")
    value = reference.get("path")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"Frozen ID {row['id']} requires an absolute source-video path.")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source video not found for {row['id']}: {path}")
    return str(path)


def _receipts(selected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_lane = defaultdict(set)
    for row in selected:
        by_lane[_cache_path(row).parent.parent / "validation.jsonl"].add(row["id"])
    receipts = {}
    for path, identifiers in by_lane.items():
        indexed = _by_id(_read_jsonl(path), path)
        missing = identifiers - indexed.keys()
        if missing:
            raise ValueError(f"Missing native media receipts in {path}: {sorted(missing)[:5]}")
        receipts.update({identifier: indexed[identifier] for identifier in identifiers})
    return receipts


def _split(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict], list[dict], int]:
    parents = {row["extra_info"]["sample_id"]: row["extra_info"]["sample_id"] for row in rows}

    def find(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    anchors = {}
    for row in rows:
        info = row["extra_info"]
        identifier = info["sample_id"]
        for token in (("group", info["source_group"]), ("path", info["source_video"])):
            previous = anchors.setdefault(token, identifier)
            parents[find(identifier)] = find(previous)
    components = defaultdict(list)
    for row in rows:
        components[find(row["extra_info"]["sample_id"])].append(row)
    groups = [components[key] for key in sorted(components)]
    if len(groups) < 2:
        raise ValueError("At least two independent source groups are required for train/validation splitting.")
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(rows) * val_fraction))
    cumulative = 0
    candidates = []
    for index, group in enumerate(groups[:-1], 1):
        cumulative += len(group)
        candidates.append((abs(cumulative - target), index))
    _, split_at = min(candidates)
    validation = [row for group in groups[:split_at] for row in group]
    training = [row for group in groups[split_at:] for row in group]
    for split, samples in (("train", training), ("test", validation)):
        samples.sort(key=lambda row: row["extra_info"]["sample_id"])
        for index, row in enumerate(samples):
            row["extra_info"].update(split=split, index=index)
    return training, validation, len(groups)


def main() -> None:
    """Write prompt-only parquet splits for every released passerby-removal example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not math.isfinite(args.val_fraction) or not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be finite and strictly between zero and one.")

    release_dir = args.release_dir.expanduser().resolve()
    released = _release_rows(release_dir)
    if not released:
        raise ValueError("The V6 release contains no examples.")
    if args.frozen_manifest is None:
        producer_roots = {_cache_path(row).parents[3] for row in released}
        if len(producer_roots) != 1:
            parser.error("Cannot infer one native producer root; pass --frozen-manifest and --provenance explicitly.")
        frozen_path = producer_roots.pop() / "frozen_training.jsonl"
    else:
        frozen_path = args.frozen_manifest.expanduser().resolve()
    provenance_path = (
        args.provenance.expanduser().resolve() if args.provenance else frozen_path.parent / "PROVENANCE.jsonl"
    )
    frozen = _by_id(_read_jsonl(frozen_path), frozen_path)
    provenance = _by_id(_read_jsonl(provenance_path), provenance_path)
    selected = [row for row in released if row.get("dataset_name") == _DATASET]
    if not selected:
        raise ValueError(f"The V6 release has no {_DATASET} examples.")
    receipts = _receipts(selected)
    rows = []
    source_lines: dict[Path, list[bytes]] = {}
    for metadata in sorted(selected, key=lambda row: row["id"]):
        identifier = metadata["id"]
        if identifier not in frozen or identifier not in provenance:
            raise ValueError(f"Missing frozen record or provenance for released ID {identifier}.")
        record = frozen[identifier]
        origin = provenance[identifier]
        if origin.get("dataset") != _DATASET:
            raise ValueError(f"Release/provenance category mismatch for {identifier}.")
        receipt = receipts[identifier]
        if receipt.get("data_id") != metadata["data_id"] or receipt.get("media", {}).get("id") != identifier:
            raise ValueError(f"Native receipt does not match release ID/data_id {identifier}.")
        media = receipt["media"]["source"]
        num_frames = _positive_integer(media.get("num_frames"), "native num_frames")
        height = _positive_integer(media.get("height"), "native height")
        width = _positive_integer(media.get("width"), "native width")
        if num_frames < 22 or num_frames % 17 != 5 or height % 32 or width % 32:
            raise ValueError(f"Invalid native H3 frame count/canvas for {identifier}.")
        try:
            native_fps = Fraction(str(media.get("native_fps")))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f"Invalid source FPS in the native receipt for {identifier}.") from error
        if native_fps != 24:
            raise ValueError(f"This V6 recipe requires a 24 FPS source for {identifier}, got {native_fps}.")
        prompt = record.get("instruction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Frozen ID {identifier} has no H3 instruction.")
        if any(token in prompt for token in ("<|", "<image>", "<video>", "<audio>")):
            raise ValueError(f"Frozen ID {identifier} contains unsupported processor/media control tokens.")
        selection = origin.get("selection", {})
        try:
            instruction_task, instruction, task_origin, task_original = _instruction_task(
                record, selection, source_lines
            )
        except ValueError as error:
            raise ValueError(f"Cannot classify frozen ID {identifier}: {error}") from error
        if any(token in instruction for token in ("<|", "<image>", "<video>", "<audio>")):
            raise ValueError(f"Reward instruction for {identifier} contains unsupported control tokens.")
        source = _source(record)
        if Path(media.get("source_path", "")).resolve() != Path(source):
            raise ValueError(f"Frozen record and native receipt use different source paths for {identifier}.")
        group = selection.get("source_group")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"No original source group in provenance for {identifier}.")
        rows.append(
            {
                "data_source": "minimax_h3_video_edit_v6",
                "prompt": [{"role": "user", "content": f"<video>{prompt}"}],
                "ability": "video_editing",
                "videos": [{"video": source, "start_time_seconds": None}],
                "reward_model": {"style": "model", "ground_truth": instruction},
                "extra_info": {
                    "sample_id": identifier,
                    "dataset_name": _DATASET,
                    "source_video": source,
                    "source_videos": [source],
                    "source_group": group,
                    "edit_instruction": instruction,
                    "instruction_task": instruction_task,
                    "instruction_task_origin": task_origin,
                    "instruction_task_original_instruction": task_original,
                    "original_edit_instruction": prompt,
                    "num_frames": num_frames,
                    "source_height": height,
                    "source_width": width,
                    "rollout_shape": {"num_frames": num_frames, "height": height, "width": width, "frame_rate": 24.0},
                    "source_native_frames": media.get("native_frames"),
                    "source_native_fps": media.get("native_fps"),
                    "source_valid_frames": media.get("valid_frames"),
                    "source_padding_frames": media.get("padding_frames"),
                    "source_duration_rational": media.get("duration_rational"),
                    "sft_cache_path": str(_cache_path(metadata)),
                    "sft_cache_sequence_length": metadata["cache"]["sequence_length"],
                    "sft_lane": metadata["lane"],
                    "sft_data_id": metadata["data_id"],
                    "original_record_id": selection.get("record_id"),
                    "original_manifest": selection.get("input_manifest"),
                    "original_manifest_line": selection.get("input_line"),
                },
            }
        )
    train, validation, split_groups = _split(rows, args.val_fraction, args.seed)
    manifest = {
        "format": "video_edit_v6",
        "schema": "h3_video_edit_v6.v2",
        "release_dir": str(release_dir),
        "task_metadata": str(release_dir / "task_metadata.jsonl"),
        "cache_index": str(release_dir / "cache_index.tsv"),
        "frozen_manifest": str(frozen_path),
        "provenance": str(provenance_path),
        "category_filter": _DATASET,
        "shape_policy": "per_sample",
        "selection": {"num_frames": None, "height": None, "width": None},
        "fps": 24.0,
        "seed": args.seed,
        "val_fraction_requested": args.val_fraction,
        "split_policy": "connected components of original source_group and resolved source-video path",
        "independent_split_groups": split_groups,
        "released_records": len(released),
        "released_categories": dict(Counter(row.get("dataset_name", "unknown") for row in released)),
        "passerby_records": len(selected),
        "selected_categories": dict(Counter(row["extra_info"]["dataset_name"] for row in rows)),
        "shape_buckets": [
            {"num_frames": frames, "height": height, "width": width, "records": count}
            for (frames, height, width), count in sorted(
                Counter(
                    (
                        row["extra_info"]["num_frames"],
                        row["extra_info"]["source_height"],
                        row["extra_info"]["source_width"],
                    )
                    for row in rows
                ).items()
            )
        ],
        "skipped_other_frame_counts": 0,
        "skipped_other_canvases": 0,
        "selected_records": len(rows),
        "instruction_tasks": dict(Counter(row["extra_info"]["instruction_task"] for row in rows)),
        "instruction_task_origins": dict(Counter(row["extra_info"]["instruction_task_origin"] for row in rows)),
        "train_records": len(train),
        "validation_records": len(validation),
        "train_ids": [row["extra_info"]["sample_id"] for row in train],
        "validation_ids": [row["extra_info"]["sample_id"] for row in validation],
        "source_media_policy": "Original source files; no crop, temporal padding, cache reuse, or target-video input.",
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in ("train.parquet", "test.parquet", "conversion_manifest.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Output files already exist; choose a fresh --output-dir.")
    pd.DataFrame(train).to_parquet(outputs[0], row_group_size=500)
    pd.DataFrame(validation).to_parquet(outputs[1], row_group_size=500)
    outputs[2].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(train)} training and {len(validation)} validation prompts to {output_dir}")


if __name__ == "__main__":
    main()
