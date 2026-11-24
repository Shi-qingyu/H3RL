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
"""Exercise ranked retention using small local checkpoints."""

import importlib.util
import json
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def retention():
    path = Path(__file__).resolve().parents[2] / "verl_omni/utils/checkpoint_retention.py"
    spec = importlib.util.spec_from_file_location("checkpoint_retention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(step, staging):
    checkpoint = staging / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"dataloader")


def test_top_three_skip_writes_and_keep_earlier_ties(tmp_path, retention):
    for step, score in [(5, 2), (10, 4), (15, 0), (25, 3)]:
        assert retention.save_if_best_checkpoint(tmp_path, 3, step, score, partial(_save, step))
    writer = Mock()
    for step, score in [(25, 3), (30, 2), (35, -1), (40, float("nan")), (45, float("inf"))]:
        assert not retention.save_if_best_checkpoint(tmp_path, 3, step, score, writer)
    writer.assert_not_called()
    assert json.loads((tmp_path / "best_checkpoints.json").read_text()) == [
        {"step": 10, "score": 4},
        {"step": 25, "score": 3},
        {"step": 5, "score": 2},
    ]
    assert {p.name for p in tmp_path.glob("global_step_*")} == {"global_step_5", "global_step_10", "global_step_25"}
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "25"


def test_reconcile_after_interrupted_publication(tmp_path, retention, monkeypatch):
    retention.save_if_best_checkpoint(tmp_path, 1, 5, 2, partial(_save, 5))
    with monkeypatch.context() as patch:
        patch.setattr(retention, "_publish", Mock(side_effect=OSError("interrupted")))
        with pytest.raises(OSError, match="interrupted"):
            retention.save_if_best_checkpoint(tmp_path, 1, 10, 3, partial(_save, 10))
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "5"
    assert (tmp_path / "global_step_5/data.pt").is_file()
    retention.reconcile_best_checkpoints(tmp_path, 1)
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "10"
    assert not (tmp_path / "global_step_5").exists()
    assert (tmp_path / "global_step_10/data.pt").is_file()


def test_incomplete_save_preserves_best_and_can_retry(tmp_path, retention):
    retention.save_if_best_checkpoint(tmp_path, 1, 5, 3, partial(_save, 5))

    def partial_save(staging):
        (staging / "global_step_10/actor").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="did not complete"):
        retention.save_if_best_checkpoint(tmp_path, 1, 10, 4, partial_save)
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "5"
    assert not (tmp_path / "global_step_10").exists()
    assert retention.save_if_best_checkpoint(tmp_path, 1, 10, 4, partial(_save, 10))


def test_unranked_and_symlinked_checkpoints_stay_outside_ranking(tmp_path, retention):
    root = tmp_path / "checkpoints"
    _save(5, root)
    other_run = tmp_path / "other"
    retention.save_if_best_checkpoint(other_run, 1, 10, 1, partial(_save, 10))
    (root / "global_step_10").symlink_to(other_run / "global_step_10", target_is_directory=True)
    with pytest.raises(FileExistsError, match="Unranked checkpoint"):
        retention.save_if_best_checkpoint(root, 1, 5, 4, partial(_save, 5))
    retention.save_if_best_checkpoint(root, 1, 15, 5, partial(_save, 15))
    retention.reconcile_best_checkpoints(root, 1)
    assert (root / "global_step_5/data.pt").is_file()
    assert (root / "global_step_10").is_symlink()
    assert (other_run / "global_step_10/data.pt").is_file()
