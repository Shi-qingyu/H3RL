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
"""Opt-in worker stack snapshots and request-stage events for stalled rollouts."""

import atexit
import json
import logging
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)
_owner_pid: int | None = None
_events: TextIO | None = None
_stacks: TextIO | None = None


def enable_worker_diagnostics() -> None:
    """Start stack sampling from an actor/task entry point, for at most one hour.

    Do not register this as a Ray worker-process setup hook: importing the parent
    package loads accelerator dependencies before Ray assigns GPU visibility.
    Snapshot Python frame objects from a Python thread, without the lock-free
    faulthandler watchdog (CPython issue 140815). Sampling pauses if native code
    holds the GIL; the external collector is needed for that case.
    """
    global _owner_pid, _events, _stacks
    directory = os.environ.get("VERL_OMNI_DIAGNOSTICS_DIR")
    if not directory or _owner_pid == os.getpid():
        return
    _owner_pid = os.getpid()
    try:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{socket.gethostname()}_{_owner_pid}"
        _events = (root / f"{stem}.events.jsonl").open("a", buffering=1, encoding="utf-8")
        _stacks = (root / f"{stem}.stacks.log").open("a", buffering=1, encoding="utf-8")
        _stacks.write(f"Started {datetime.now(UTC).isoformat()}; pid={_owner_pid}; interval=10s\n")
    except OSError:
        logger.exception("Could not initialize worker diagnostics in %s", directory)
        for stream in (_events, _stacks):
            if stream is not None:
                stream.close()
        _events = _stacks = None
        return

    stop = threading.Event()
    deadline = time.monotonic() + 3600
    stack_file = _stacks
    event_file = _events

    def sample_stacks() -> None:
        try:
            while not stop.wait(10):
                if time.monotonic() >= deadline:
                    break
                lines = [f"Snapshot {datetime.now(UTC).isoformat()} (most recent call first)\n"]
                frames = sys._current_frames()
                try:
                    for thread_id, frame in frames.items():
                        lines.append(f"Thread {thread_id:#x}:\n")
                        for depth, (entry, lineno) in enumerate(traceback.walk_stack(frame)):
                            if depth >= 100:
                                lines.append("  ... truncated at 100 frames\n")
                                break
                            lines.append(
                                f'  File "{entry.f_code.co_filename}", line {lineno}, in {entry.f_code.co_name}\n'
                            )
                        lines.append("\n")
                finally:
                    # Frames retain model tensors through locals; release them between samples.
                    frames.clear()
                    frame = entry = None
                stack_file.write("".join(lines))
                stack_file.flush()
        except (OSError, ValueError):
            logger.exception("Could not write worker stack snapshot")
        finally:
            stack_file.close()

    sampler = threading.Thread(target=sample_stacks, name="worker-stack-sampler", daemon=True)
    sampler.start()

    def close_files() -> None:
        stop.set()
        sampler.join(timeout=1)
        event_file.close()

    atexit.register(close_files)
    diagnostic_event("worker.started", python=sys.version, stack_sampler="python_thread")


def diagnostic_event(event: str, **fields: Any) -> None:
    """Append a timestamped process/request event when diagnostics are enabled."""
    if not os.environ.get("VERL_OMNI_DIAGNOSTICS_DIR"):
        return
    if _owner_pid != os.getpid():
        enable_worker_diagnostics()
    if _events is None or _events.closed:
        return
    record = {
        "time": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "thread": threading.get_ident(),
        "event": event,
        **fields,
    }
    try:
        _events.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError):
        logger.exception("Could not write worker diagnostic event %s", event)


@contextmanager
def diagnostic_span(stage: str, **fields: Any) -> Iterator[None]:
    """Record entry, completion or failure without changing exception propagation."""
    if not os.environ.get("VERL_OMNI_DIAGNOSTICS_DIR"):
        yield
        return
    diagnostic_event(f"{stage}.start", **fields)
    try:
        yield
    except BaseException as error:
        diagnostic_event(f"{stage}.error", error_type=type(error).__name__, error=str(error), **fields)
        raise
    else:
        diagnostic_event(f"{stage}.done", **fields)
