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
"""Evaluate one H3 checkpoint or compare a training run's saved policies on fixed validation rows."""

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pyarrow.parquet as pq

PACKAGE = "examples.diffusionnft_trainer.minimax_h3"
METRICS = ("score", "instruction_following", "source_fidelity", "visual_quality")


def _gpu_pairs(gpus: str) -> list[str]:
    ids = [item.strip() for item in gpus.split(",")]
    if len(ids) != 6 or any(not re.fullmatch(r"0|[1-9][0-9]*", i) for i in ids) or len(set(ids)) != 6:
        raise argparse.ArgumentTypeError("Pass six distinct local GPU indices, for example 1,2,3,4,5,6.")
    return [",".join(ids[i : i + 2]) for i in range(0, 6, 2)]


def _validation_groups(data_file: Path, count: int) -> tuple[list[int], list[list[int]]]:
    info = pq.read_table(data_file, columns=["extra_info"])["extra_info"]
    if not len(info) or count < 0:
        raise ValueError("The validation split must be nonempty and --num-samples must be >= 0.")
    # Match the V6 trainer's data.seed=42 selection; 0 requests the full split.
    indices = list(range(len(info)))
    if 0 < count < len(info):
        indices = np.random.default_rng(42).choice(len(info), count, replace=False).tolist()
    costs = {}
    for index in indices:
        shape = info[index].as_py()["rollout_shape"]
        costs[index] = shape["num_frames"] * shape["height"] * shape["width"]
    groups = [[], [], []]
    loads = [0, 0, 0]
    for index in sorted(indices, key=costs.__getitem__, reverse=True):
        group = loads.index(min(loads))
        groups[group].append(index)
        loads[group] += costs[index]
    return indices, groups


def _reward_health(url: str) -> dict:
    if not url.endswith("/score"):
        raise ValueError("--reward-url must point to the VideoReward /score endpoint.")
    # Match the scoring client's direct connection even in Pods with HTTP_PROXY set.
    with build_opener(ProxyHandler({})).open(url.removesuffix("/score") + "/health", timeout=15) as response:
        health = json.load(response)
    if (
        health.get("status") != "ok"
        or health.get("protocol_version") != "1"
        or set(health.get("dimensions", [])) != set(METRICS[1:])
    ):
        raise RuntimeError(f"VideoReward returned an incompatible health response: {health}")
    return health


def _worker_env(gpu_pair: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("RAY_ADDRESS", "MASTER_ADDR", "MASTER_PORT", "VLLM_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpu_pair, PYTHONUNBUFFERED="1")
    return env


def _stop_workers(processes: list[subprocess.Popen]) -> None:
    # Each evaluator owns a process group including its TP ranks.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + 5
            for process in processes:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
    for process in processes:
        process.wait()


def _generate_and_score(
    args, output: Path, adapter: Path, groups: list[list[int]], source_dir: Path | None = None
) -> None:
    running = []
    try:
        for number, (gpu_pair, rows) in enumerate(zip(args.gpus, groups, strict=True)):
            if not rows:
                continue
            worker_dir = output / f"group_{number}"
            log_path = worker_dir.with_suffix(".log")
            command = [
                sys.executable,
                "-m",
                f"{PACKAGE}.infer_minimax_h3_video_edit",
                "--model-path",
                str(args.model_path),
                "--lora-path",
                str(adapter),
                "--data-file",
                str(args.data_file),
                "--output-dir",
                str(worker_dir),
                "--validation-indices",
                ",".join(map(str, rows)),
                "--reward-url",
                args.reward_url,
                "--score-weights",
                *map(str, args.score_weights),
                "--tensor-parallel-size",
                "2",
                "--text-encoder-tp-size",
                "2",
                "--num-inference-steps",
                str(args.num_inference_steps),
                "--seed",
                "42",
                "--reference-image-short-edge",
                str(args.reference_image_short_edge),
                "--attention-backend",
                args.attention_backend,
            ]
            if source_dir is not None:
                command.extend(["--validation-source-dir", str(source_dir)])
            print(f"GPU {gpu_pair}: rows {rows}, log {log_path}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=_worker_env(gpu_pair),
                    start_new_session=True,
                )
            running.append((process, log_path))
        while True:
            statuses = [process.poll() for process, _ in running]
            for (_, log_path), status in zip(running, statuses, strict=True):
                if status not in (None, 0):
                    raise RuntimeError(f"Validation worker exited with status {status}. See {log_path}")
            if None not in statuses:
                return
            time.sleep(1)
    except BaseException:
        _stop_workers([process for process, _ in running])
        raise


def _aggregate(output: Path, indices: list[int], groups: list[list[int]]) -> dict:
    rows = {}
    for number, group in enumerate(groups):
        for index in group:
            report = Path(f"group_{number}/row_{index}.json")
            record = json.loads((output / report).read_text())
            if record["row_index"] != index or index in rows:
                raise ValueError(f"Unexpected or duplicate validation row in {report}")
            metrics = {name: record[name] for name in METRICS}
            if not all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()):
                raise ValueError(f"Invalid reward in {report}")
            rows[index] = {"row_index": index, "report": str(report), **metrics}
    if set(rows) != set(indices):
        raise ValueError("Incomplete validation results")
    means = {name: statistics.fmean(row[name] for row in rows.values()) for name in METRICS}
    return {"mean": means, "val-aggregate/reward/mean": means["score"], "samples": [rows[i] for i in indices]}


def _evaluate_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
    output: Path,
    indices: list[int],
    groups: list[list[int]],
    health: dict,
    source_dir: Path | None = None,
) -> None:
    print(f"Offline validation: {output}", flush=True)
    config = {
        "checkpoint": str(checkpoint),
        "adapter": args.adapter,
        "base_model": str(args.model_path.resolve()),
        "data_file": str(args.data_file.resolve()),
        "indices": indices,
        "seed": 42,
        "num_inference_steps": args.num_inference_steps,
        "reference_image_short_edge": args.reference_image_short_edge,
        "attention_backend": args.attention_backend,
        "reward_url": args.reward_url,
        "reward_model": health,
        "score_weights": args.score_weights,
        "gpu_pairs": args.gpus,
        "rows_by_group": groups,
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="weights_", dir=output) as work:
        adapter = Path(work) / "adapter"
        export_env = _worker_env("")
        export_env.setdefault("OMP_NUM_THREADS", "4")
        subprocess.run(
            [
                sys.executable,
                "-m",
                f"{PACKAGE}.export_video_edit_rl_adapter",
                "--actor-dir",
                str(checkpoint / "actor"),
                "--adapter",
                args.adapter,
                "--template",
                str(args.template),
                "--output-dir",
                str(adapter),
            ],
            env=export_env,
            check=True,
        )
        _generate_and_score(args, output, adapter, groups, source_dir)
    summary = {**config, **_aggregate(output, indices, groups)}
    pending = output / "summary.json.tmp"
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pending.replace(output / "summary.json")
    print(f"Mean reward: {summary['mean']['score']:.6f}. Saved {output / 'summary.json'}", flush=True)


def _completed_checkpoints(directory: Path, steps: list[int] | None) -> list[Path]:
    completed = {
        int(match[1]): path
        for path in directory.glob("global_step_*")
        if (match := re.fullmatch(r"global_step_(\d+)", path.name))
        and (path / "actor").is_dir()
        and (path / "data.pt").is_file()
    }
    if steps is not None and (missing := set(steps) - completed.keys()):
        raise ValueError(f"Requested checkpoints missing or incomplete in {directory}: {sorted(missing)}")
    if not completed:
        raise ValueError(f"No completed global_step_N checkpoints in {directory}")
    return [completed[step] for step in sorted(completed if steps is None else set(steps))]


def main() -> None:
    """Evaluate a saved policy or a snapshot of the completed checkpoints in a run."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("checkpoint", type=Path, help="Training run directory, global_step_N directory, or its actor.")
    parser.add_argument("--steps", type=int, nargs="+", help="Evaluate these saved steps when given a training run.")
    parser.add_argument("--gpus", type=_gpu_pairs, default="1,2,3,4,5,6", help="Six local GPU indices in TP pairs.")
    parser.add_argument("--reward-gpu", type=int, default=7, help="Local GPU for the sweep's dedicated reward model.")
    parser.add_argument("--reward-python", type=Path, help="Python executable in the VideoReward environment.")
    parser.add_argument("--adapter", choices=["default", "old"], default="default", help="default is current policy.")
    parser.add_argument("--output-root", type=Path, help="Parent for new runs; defaults to RUN_DIR/offline_validation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True, help="V6 SFT adapter directory for layout metadata.")
    parser.add_argument("--data-file", type=Path, required=True, help="Prepared V6 validation parquet.")
    parser.add_argument("--reward-url", required=True, help="Training reward endpoint identifying the scorer to use.")
    parser.add_argument("--score-weights", nargs=3, type=float, default=[1, 1, 1], metavar=("IF", "SF", "VQ"))
    parser.add_argument("--num-samples", type=int, default=4, help="Fixed validation subset size; 0 uses all rows.")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--reference-image-short-edge", type=int, default=2048)
    parser.add_argument("--attention-backend", default="TORCH_SDPA")
    args = parser.parse_args()
    path = args.checkpoint.resolve()
    if path.name == "actor":
        path = path.parent
    sweep = (path / "checkpoints").is_dir()
    if sweep:
        run_dir = path
        checkpoints = _completed_checkpoints(run_dir / "checkpoints", args.steps)
        if args.reward_gpu < 0 or str(args.reward_gpu) in ",".join(args.gpus).split(","):
            parser.error("--reward-gpu must be a separate, nonnegative local GPU index")
    else:
        if args.steps:
            parser.error("Pass a training run directory to select multiple --steps")
        if not re.fullmatch(r"global_step_[0-9]+", path.name) or not (path / "actor").is_dir():
            parser.error(f"Expected a training run or a saved global_step_N directory at {path}")
        if not (path / "data.pt").is_file():
            parser.error(f"Checkpoint save is incomplete: missing {path / 'data.pt'}")
        checkpoints = [path]
        run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    if args.num_inference_steps < 2:
        parser.error("--num-inference-steps must be at least 2")
    if any(w < 0 for w in args.score_weights) or not 0 < sum(args.score_weights) < math.inf:
        parser.error("--score-weights must be nonnegative and have a positive finite sum")
    root = (args.output_root or run_dir / "offline_validation").resolve()
    if root.is_relative_to(run_dir / "checkpoints") or any(root.is_relative_to(p) for p in checkpoints):
        parser.error("Choose an output directory outside the checkpoints.")
    indices, groups = _validation_groups(args.data_file, args.num_samples)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{'sweep' if sweep else path.name}_{args.adapter}_{datetime.now():%Y%m%d_%H%M%S}_"
    output = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    if not sweep:
        _evaluate_checkpoint(args, path, output, indices, groups, _reward_health(args.reward_url))
        return

    from .offline_reward_service import dedicated_reward_service
    from .summarize_offline_validation import summarize_offline_validation

    print(f"Comparing {[p.name for p in checkpoints]} on {len(indices)} fixed rows in {output}", flush=True)
    completed = []
    with dedicated_reward_service(output, str(args.reward_gpu), args.reward_url, args.reward_python) as local_url:
        args.reward_url = local_url
        health = _reward_health(local_url)
        for checkpoint in checkpoints:
            step_dir = output / checkpoint.name
            step_dir.mkdir()
            _evaluate_checkpoint(args, checkpoint, step_dir, indices, groups, health, output / "sources")
            completed.append(step_dir)
            summarize_offline_validation(output, completed)
            print(f"Updated {output / 'reward_trend.csv'}", flush=True)


if __name__ == "__main__":

    def _terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)
    main()
