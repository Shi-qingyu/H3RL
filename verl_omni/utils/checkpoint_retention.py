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
"""Retain local checkpoints by validation score."""

import fcntl
import json
import math
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

_SCORE_FILE = "validation_checkpoint_score.json"


@contextmanager
def _staging_area(root: Path, keep: int) -> Iterator[Path]:
    if type(keep) is not int or keep < 1:
        raise ValueError("Checkpoint retention requires a positive integer limit")
    work = root / ".best_checkpoints"
    if work.is_symlink():
        raise ValueError(f"Unexpected checkpoint workspace symlink: {work}")
    work.mkdir(parents=True, exist_ok=True)
    staging = work / "staging"
    lock = work / "writer.lock"
    if staging.exists() and not lock.is_file():
        raise FileExistsError(f"Unrecognized checkpoint staging directory: {staging}")
    with lock.open("a") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            yield staging
        finally:
            shutil.rmtree(staging)


def _scores(root: Path) -> dict[int, float]:
    scores = {}
    for path in root.glob(f"global_step_*/{_SCORE_FILE}"):
        if path.is_symlink() or path.parent.is_symlink():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        step, score = record["step"], record["score"]
        if type(step) is not int or path.parent.name != f"global_step_{step}" or not math.isfinite(score):
            raise ValueError(f"Invalid checkpoint score in {path}")
        scores[step] = score
    return scores


def _rank(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda step: (-scores[step], step))


def _publish(root: Path, staging: Path, scores: dict[int, float], keep: int) -> None:
    if not scores:
        return
    ranked = _rank(scores)
    best = [{"step": step, "score": scores[step]} for step in ranked[:keep]]
    # Publish both indexes before retiring any directory a previous index could reference.
    for name, value in (
        ("best_checkpoints.json", json.dumps(best, indent=2) + "\n"),
        ("latest_checkpointed_iteration.txt", str(max(ranked[:keep]))),
    ):
        path = staging / name
        path.write_text(value, encoding="utf-8")
        path.replace(root / name)
    for step in ranked[keep:]:
        (root / f"global_step_{step}").rename(staging / f"evicted_step_{step}")


def reconcile_best_checkpoints(checkpoint_dir: str | Path, keep: int) -> None:
    """Recover the local ranking and resume marker before loading a checkpoint."""
    root = Path(checkpoint_dir).resolve()
    with _staging_area(root, keep) as staging:
        _publish(root, staging, _scores(root), keep)


def save_if_best_checkpoint(
    checkpoint_dir: str | Path, keep: int, step: int, score: float, save_checkpoint: Callable[[Path], None]
) -> bool:
    """Save a qualifying checkpoint and atomically publish its rank on the shared filesystem.

    The callback must finish writing global_step_<step>/actor and data.pt under
    the supplied staging root before returning. Ties favor the earlier step.
    """
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    root = Path(checkpoint_dir).resolve()
    with _staging_area(root, keep) as staging:
        scores = _scores(root)
        qualifies = step not in scores and math.isfinite(score) and step in _rank({**scores, step: score})[:keep]
        if qualifies:
            name = f"global_step_{step}"
            if (root / name).exists() or (root / name).is_symlink():
                raise FileExistsError(f"Unranked checkpoint already exists: {root / name}")
            save_checkpoint(staging)
            saved = staging / name
            if saved.is_symlink() or not (saved / "actor").is_dir() or not (saved / "data.pt").is_file():
                raise RuntimeError(f"Checkpoint callback did not complete the actor and dataloader checkpoint: {saved}")
            # Keeping its score in the same rename makes a completed save recoverable before the indexes are written.
            (saved / _SCORE_FILE).write_text(json.dumps({"step": step, "score": score}), encoding="utf-8")
            saved.rename(root / name)
            scores[step] = score
        _publish(root, staging, scores, keep)
        return qualifies
