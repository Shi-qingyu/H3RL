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
"""Explicit V6 human-removal declarations map to the matching reward task."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_process():
    """Load the converter without preparing data or loading model checkpoints."""
    path = (
        Path(__file__).resolve().parents[2] / "examples/diffusionnft_trainer/minimax_h3/prepare_video_edit_v6_data.py"
    )
    spec = importlib.util.spec_from_file_location("_video_edit_v6_data_process", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "declaration",
    [
        "Remove every real person.",
        "Remove every real person, including distant pedestrians and riders.",
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_remove_all_short_and_long_declarations(data_process, declaration, wrapped):
    """Recognize both published forms, including repeated IR sections."""
    if wrapped:
        instruction = (
            "subject_definitions:\n<Video 1> is the source video.\nsummary:\n"
            f"[video editing] {declaration} Remove the removed people's shadows and reflections. "
            "Preserve the retained subject and unrelated background.\n"
            f"detailed_description:\n{declaration} Preserve unrelated scene content."
        )
    else:
        instruction = declaration
    assert data_process._task(instruction) == ("remove_all", data_process._REMOVE_ALL)


def test_keep_main_does_not_match_remove_other_people(data_process):
    instruction = (
        "summary:\n[video editing] Keep only this main person: the person in red. "
        "Remove every other real person.\ndetailed_description:\n"
        "Keep only this main person: the person in red."
    )
    assert data_process._task(instruction) == ("keep_main", data_process._KEEP_MAIN)


@pytest.mark.parametrize("template", ["_KEEP_MAIN", "_REMOVE_ALL"])
def test_canonical_reward_templates_remain_supported(data_process, template):
    instruction = getattr(data_process, template)
    task, reward_instruction = data_process._task(instruction)
    assert task == ("keep_main" if template == "_KEEP_MAIN" else "remove_all")
    assert reward_instruction == instruction


@pytest.mark.parametrize(
    "instruction",
    [
        "Keep only this main person: the person in red.\nRemove every real person.",
        "Keep only this main person: the person in red.\n"
        "Remove every real person, including distant pedestrians and riders.",
        "Remove every other real person.",
        "Remove every real person except the person in red.",
        "Remove the object behind the person.",
        "",
    ],
)
def test_ambiguous_or_unknown_tasks_fail_closed(data_process, instruction):
    with pytest.raises(ValueError, match="exactly one"):
        data_process._task(instruction)


@pytest.fixture
def reworded_case(tmp_path):
    """Bind a rewritten IR to the second raw JSONL line and its variant UID."""
    frozen_instruction = (
        "subject_definitions:\n<Video 1> is the source video.\nsummary:\n"
        "[video editing] Retain the foreground main hand with the monkey. "
        "Remove the background person.\ndetailed_description:\n"
        "Keep the hand, monkey, scene content and camera motion unchanged."
    )
    variant_uid = "c5a2_reworded::keep_main::reviewed"

    def prepare(*, source_overrides=None, omit_original=False, newline=b"\n"):
        record = {"id": "c5a2_reworded", "instruction": frozen_instruction}
        source = {
            "id": record["id"],
            "uid": variant_uid,
            "training_instruction": frozen_instruction,
            "original_instruction": (
                "Keep only this main person: the foreground main hand with the monkey.\nRemove every other real person."
            ),
        }
        source.update(source_overrides or {})
        if omit_original:
            source.pop("original_instruction")
        raw_line = json.dumps(source, ensure_ascii=False).encode("utf-8") + newline
        manifest = tmp_path / "source_input.jsonl"
        manifest.write_bytes(b'{"uid":"unrelated-first-line"}\n' + raw_line)
        selection = {
            "input_manifest": str(manifest),
            "input_line": 2,
            "input_record_sha256": hashlib.sha256(raw_line).hexdigest(),
            "record_id": variant_uid,
            "bucket": "all",
        }
        return record, selection, source, raw_line

    return prepare


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_reworded_keep_main_uses_bound_original_and_preserves_h3_prompt(data_process, reworded_case, newline):
    record, selection, source, raw_line = reworded_case(newline=newline)
    prompt = record["instruction"]
    source_lines = {}
    assert source["uid"] != source["id"]
    result = data_process._instruction_task(record, selection, source_lines)
    assert result == (
        "keep_main",
        data_process._KEEP_MAIN,
        "source_original_instruction",
        source["original_instruction"],
    )
    assert record["instruction"] == prompt
    assert record["instruction"] != result[1]
    assert source_lines[Path(selection["input_manifest"])][1] == raw_line


def test_source_record_hash_includes_the_trailing_newline(data_process, reworded_case):
    record, selection, _, raw_line = reworded_case()
    selection["input_record_sha256"] = hashlib.sha256(raw_line.removesuffix(b"\n")).hexdigest()
    with pytest.raises(ValueError, match="source-line hash mismatch"):
        data_process._instruction_task(record, selection, {})


@pytest.mark.parametrize(
    ("source_overrides", "message"),
    [
        ({"uid": "c5a2_reworded"}, "record_id does not match the source uid"),
        ({"training_instruction": "A different rewritten training prompt."}, "frozen IR differs"),
    ],
)
def test_reworded_task_rejects_uid_and_training_text_mismatches(data_process, reworded_case, source_overrides, message):
    record, selection, _, _ = reworded_case(source_overrides=source_overrides)
    with pytest.raises(ValueError, match=message):
        data_process._instruction_task(record, selection, {})


@pytest.mark.parametrize("original", [None, "", "   ", 17])
def test_reworded_task_rejects_unusable_original_instruction(data_process, reworded_case, original):
    record, selection, _, _ = reworded_case(source_overrides={"original_instruction": original})
    with pytest.raises(ValueError, match="No original_instruction"):
        data_process._instruction_task(record, selection, {})


def test_reworded_task_requires_original_instruction_field(data_process, reworded_case):
    record, selection, _, _ = reworded_case(omit_original=True)
    with pytest.raises(ValueError, match="No original_instruction"):
        data_process._instruction_task(record, selection, {})


@pytest.mark.parametrize("template", ["_KEEP_MAIN", "_REMOVE_ALL"])
def test_explicit_frozen_task_bypasses_source_manifest(data_process, tmp_path, monkeypatch, template):
    def reject_source_read(path):
        raise AssertionError("An explicit frozen task must not consult source rows.")

    monkeypatch.setattr(Path, "read_bytes", reject_source_read)
    instruction = getattr(data_process, template)
    source_lines = {}
    selection = {"input_manifest": str(tmp_path / "missing.jsonl"), "input_line": 2}
    task, reward, origin, original = data_process._instruction_task(
        {"instruction": instruction}, selection, source_lines
    )
    assert task == ("keep_main" if template == "_KEEP_MAIN" else "remove_all")
    assert reward == instruction
    assert origin == "frozen_instruction"
    assert original is None
    assert source_lines == {}


def test_conflicting_frozen_task_never_falls_back_to_matching_original(data_process, reworded_case):
    conflicting_instruction = (
        "Keep only this main person: the foreground main hand with the monkey.\nRemove every real person."
    )
    record, selection, _, _ = reworded_case(source_overrides={"training_instruction": conflicting_instruction})
    record["instruction"] = conflicting_instruction
    with pytest.raises(ValueError, match="exactly one"):
        data_process._instruction_task(record, selection, {})


@pytest.mark.parametrize(
    "original",
    [
        "Keep only this main person: the foreground hand.\nRemove every real person.",
        "Retain the foreground hand with the monkey and remove the background person.",
    ],
)
def test_bound_original_still_requires_one_explicit_task(data_process, reworded_case, original):
    record, selection, _, _ = reworded_case(source_overrides={"original_instruction": original})
    with pytest.raises(ValueError, match="exactly one"):
        data_process._instruction_task(record, selection, {})
