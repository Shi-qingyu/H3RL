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
"""Own a single VideoReward replica for a local offline validation sweep."""

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError

from .serve_video_edit_reward import DEFAULT_SOURCE_ROOT


@contextmanager
def dedicated_reward_service(
    run_dir: Path,
    gpu: str,
    training_reward_url: str,
    python: Path | None = None,
) -> Iterator[str]:
    """Serve the training reward checkpoint on one local GPU for the duration of a sweep."""
    from .validate_video_edit_checkpoint import _reward_health, _worker_env

    health = _reward_health(training_reward_url)
    checkpoint = health.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise ValueError(f"Training reward service returned an invalid checkpoint path: {checkpoint!r}")
    if not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"Training reward checkpoint is unavailable on this node: {checkpoint}")
    if not gpu.isascii() or not gpu.isdecimal():
        raise ValueError(f"Expected one physical GPU index for offline reward, got {gpu!r}")

    source_root = Path(os.environ.get("VIDEO_REWARD_ROOT", DEFAULT_SOURCE_ROOT)).expanduser()
    interpreter = python or source_root / ".venv/bin/python"
    server = Path(__file__).with_name("serve_video_edit_reward.py")
    log_path = run_dir / "offline_reward.log"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener, log_path.open("a") as log:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/score"
        fd = listener.fileno()
        process = subprocess.Popen(
            [
                str(interpreter), str(server),
                "--fd", str(fd),
                "--checkpoint", checkpoint,
                "--source-root", str(source_root),
            ],
            pass_fds=(fd,),
            env=_worker_env(gpu),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            listener.close()
            deadline = time.monotonic() + 1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Offline VideoReward exited with {process.returncode}. See {log_path}")
                try:
                    local_health = _reward_health(url)
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Offline VideoReward startup timed out. See {log_path}") from None
                    time.sleep(2)
                    continue
                if local_health.get("checkpoint") != checkpoint:
                    raise RuntimeError(f"Offline VideoReward loaded a different checkpoint: {local_health}")
                break
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
