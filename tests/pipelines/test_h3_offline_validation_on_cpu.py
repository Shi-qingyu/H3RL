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
"""Exercise H3 checkpoint evaluation orchestration without starting GPU workers."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.diffusionnft_trainer.minimax_h3 import validate_video_edit_checkpoint as validation


def test_checkpoint_to_three_workers_and_mean(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run" / "checkpoints" / "global_step_5"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "data.pt").touch()
    rows = [{"rollout_shape": {"height": 1376, "width": 768, "num_frames": 56 if i == 7 else 124}} for i in range(86)]
    data_file = tmp_path / "test.parquet"
    pq.write_table(pa.table({"extra_info": rows}), data_file)

    def export(command, *, env, check):
        assert check and env["CUDA_VISIBLE_DEVICES"] == ""
        assert command[command.index("--actor-dir") + 1] == str(checkpoint / "actor")
        assert command[command.index("--adapter") + 1] == "old"
        Path(command[command.index("--output-dir") + 1]).mkdir()

    workers = {}

    class Worker:
        def __init__(self, command, *, env, **kwargs):
            assert "MASTER_PORT" not in env and "RAY_ADDRESS" not in env
            assert kwargs["start_new_session"] and kwargs["stdin"] == validation.subprocess.DEVNULL
            assert command[command.index("--num-inference-steps") + 1] == "40"
            assert command[command.index("--text-encoder-tp-size") + 1] == "2"
            out = Path(command[command.index("--output-dir") + 1])
            indices = list(map(int, command[command.index("--validation-indices") + 1].split(",")))
            workers[env["CUDA_VISIBLE_DEVICES"]] = indices
            out.mkdir()
            for i in indices:
                record = {"row_index": i, **dict.fromkeys(validation.METRICS, float(i))}
                (out / f"row_{i}.json").write_text(json.dumps(record))

        def poll(self):
            return 0

    monkeypatch.setattr(validation, "_reward_health", lambda url: {"checkpoint": "test-reward"})
    monkeypatch.setattr(validation.subprocess, "run", export)
    monkeypatch.setattr(validation.subprocess, "Popen", Worker)
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("RAY_ADDRESS", "training-head")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            str(checkpoint / "actor"),
            "--model-path",
            str(tmp_path),
            "--template",
            str(tmp_path),
            "--data-file",
            str(data_file),
            "--reward-url",
            "http://localhost/score",
            "--adapter",
            "old",
        ],
    )
    validation.main()
    [summary_path] = (checkpoint.parent.parent / "offline_validation").glob("*/summary.json")
    result = json.loads(summary_path.read_text())
    assert workers == {"1,2": [37, 7], "3,4": [55], "5,6": [65]}
    assert [r["row_index"] for r in result["samples"]] == [37, 7, 55, 65]
    assert result["val-aggregate/reward/mean"] == (37 + 7 + 55 + 65) / 4
    assert list(summary_path.parent.glob("weights_*")) == []


def test_invalid_gpus_and_missing_score_cannot_produce_summary(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        validation._gpu_pairs("0,1,2,2,4,5")
    with pytest.raises(FileNotFoundError):
        validation._aggregate(tmp_path, [0, 1], [[0], [1], []])


def test_failed_replica_stops_sibling_processes(tmp_path, monkeypatch):
    package = tmp_path / "fake_h3"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "infer_minimax_h3_video_edit.py").write_text(
        """
import os
import signal
import sys
import time
from pathlib import Path

out = Path(sys.argv[sys.argv.index('--output-dir') + 1])
if os.environ['CUDA_VISIBLE_DEVICES'] == '1,2':
    deadline = time.monotonic() + 10
    while not all((out.parent / f'group_{i}.ready').exists() for i in (1, 2)):
        if time.monotonic() > deadline:
            sys.exit(99)
        time.sleep(0.05)
    sys.exit(13)

def stop(*args):
    out.with_suffix('.stopped').touch()
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
out.with_suffix('.ready').touch()
time.sleep(20)
"""
    )
    monkeypatch.setattr(validation, "PACKAGE", "fake_h3")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    args = argparse.Namespace(
        model_path=tmp_path,
        data_file=tmp_path / "test.parquet",
        reward_url="http://localhost/score",
        gpus=["1,2", "3,4", "5,6"],
        score_weights=[1, 1, 1],
        num_inference_steps=40,
        reference_image_short_edge=2048,
        attention_backend="TORCH_SDPA",
    )
    with pytest.raises(RuntimeError, match="status 13"):
        validation._generate_and_score(args, tmp_path, tmp_path / "adapter", [[3], [2], [1]])
    assert (tmp_path / "group_1.stopped").exists()
    assert (tmp_path / "group_2.stopped").exists()
