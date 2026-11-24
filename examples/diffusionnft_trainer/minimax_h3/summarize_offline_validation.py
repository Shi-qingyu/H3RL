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
"""Compare H3 offline validation rewards on matched samples across checkpoints."""

import csv
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


def _write_csv(path: Path, rows: list[dict]) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


def summarize_offline_validation(run_dir: Path, outputs: list[Path]) -> None:
    """Write checkpoint means and per-sample scores with deltas from the earliest step."""
    if not outputs:
        raise ValueError("At least one completed validation output is required.")
    by_step = {}
    for output in outputs:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        match = re.fullmatch(r"global_step_(\d+)", Path(summary["checkpoint"]).name)
        if not match:
            raise ValueError(f"Invalid checkpoint in {output}: {summary['checkpoint']}")
        step = int(match[1])
        if step in by_step:
            raise ValueError(f"Duplicate checkpoint step {step} in {output} and {by_step[step][0]}")
        by_step[step] = (output, summary)

    baseline_step = min(by_step)
    baseline = by_step[baseline_step][1]
    config_fields = (
        "adapter", "base_model", "data_file", "seed", "num_inference_steps", "reference_image_short_edge",
        "attention_backend", "reward_model", "score_weights",
    )
    baseline_scores = {}
    trend = []
    cases = []
    for step, (output, summary) in sorted(by_step.items()):
        for field in config_fields:
            if summary[field] != baseline[field]:
                raise ValueError(f"Cannot compare steps {baseline_step} and {step}: {field} differs.")
        trend.append(
            {
                "step": step,
                "samples": len(summary["samples"]),
                **{f"mean_{metric}": value for metric, value in summary["mean"].items()},
                "baseline_step": baseline_step,
                "mean_score_delta": summary["mean"]["score"] - baseline["mean"]["score"],
            }
        )
        seen = set()
        for sample in summary["samples"]:
            report = output / sample["report"]
            record = json.loads(report.read_text(encoding="utf-8"))
            key = (sample["row_index"], record["sample_id"], record["seed"])
            if key in seen or key[0] != record["row_index"] or record["step"] != step:
                raise ValueError(f"Duplicate or incorrectly attributed sample in {report}")
            seen.add(key)
            if step == baseline_step:
                baseline_scores[key] = sample["score"]
            elif key not in baseline_scores:
                raise ValueError(f"Sample {key} at step {step} is missing from baseline step {baseline_step}")
            cases.append(
                {
                    "row_index": key[0],
                    "sample_id": key[1],
                    "seed": key[2],
                    "step": step,
                    **{metric: sample[metric] for metric in summary["mean"]},
                    "baseline_step": baseline_step,
                    "score_delta": sample["score"] - baseline_scores[key],
                    "prompt": record["prompt"],
                    "source_video": record["source_video"],
                    "generated_video": os.path.relpath(report.parent / record["generated_video"], run_dir),
                    "report": os.path.relpath(report, run_dir),
                }
            )
        if not seen or seen != baseline_scores.keys():
            raise ValueError(f"Incomplete sample cohort at step {step}")

    cases.sort(key=lambda row: (row["row_index"], row["seed"], row["step"]))
    _write_csv(run_dir / "paired_samples.csv", cases)
    _write_csv(run_dir / "reward_trend.csv", trend)
