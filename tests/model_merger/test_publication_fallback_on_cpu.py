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
"""No-overwrite publication on mounts without atomic directory rename flags."""

import errno
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_MARKER = "conversion_manifest.json"
_CONTENTS = {
    "adapter_config.json": b'{"r": 32}',
    "pytorch_lora_weights.safetensors": b"adapter tensors",
    _MARKER: b'{"format": "minimax_h3_sft_lora"}',
}


@pytest.fixture
def utils():
    path = Path(__file__).parents[2] / "verl_omni/model_merger/utils.py"
    spec = importlib.util.spec_from_file_location("_publication_test_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rename_error(utils, monkeypatch):
    def configure(error):
        rename = Mock(return_value=-1)
        monkeypatch.setattr(utils.ctypes, "CDLL", Mock(return_value=SimpleNamespace(renameat2=rename)))
        monkeypatch.setattr(utils.ctypes, "get_errno", lambda: error)
        return rename

    return configure


def _write_stage(staging):
    for name, content in _CONTENTS.items():
        (staging / name).write_bytes(content)


@pytest.mark.parametrize("error", sorted({errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}))
def test_unsupported_rename_publishes_complete_marker_last(tmp_path, monkeypatch, utils, rename_error, error):
    rename_error(error)
    target = tmp_path / "output"
    original_link = utils.os.link
    observed = []

    def publish_marker(source, destination):
        assert Path(destination) == target / _MARKER
        assert not Path(destination).exists()
        for name, content in _CONTENTS.items():
            if name != _MARKER:
                assert (target / name).read_bytes() == content
        assert Path(source).read_bytes() == _CONTENTS[_MARKER]
        observed.append(Path(destination))
        return original_link(source, destination)

    monkeypatch.setattr(utils.os, "link", publish_marker)
    with utils.publication_directory(target, completion_marker=_MARKER) as staging:
        _write_stage(staging)
    assert observed == [target / _MARKER]
    assert {path.name: path.read_bytes() for path in target.iterdir()} == _CONTENTS
    assert not list(tmp_path.glob(".output.merge*"))


def test_fallback_never_replaces_existing_empty_target(tmp_path, utils, rename_error):
    rename_error(errno.EINVAL)
    target = tmp_path / "output"
    with pytest.raises(FileExistsError):
        with utils.publication_directory(target, completion_marker=_MARKER) as staging:
            _write_stage(staging)
            target.mkdir()
    assert target.is_dir() and not list(target.iterdir())
    assert not list(tmp_path.glob(".output.merge*"))


@pytest.mark.parametrize("foreign_file", [False, True])
def test_copy_failure_removes_only_owned_partial_output(tmp_path, monkeypatch, utils, rename_error, foreign_file):
    rename_error(errno.EINVAL)
    target = tmp_path / "output"
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"original checkpoint")

    def fail_copy(reader, writer):
        writer.write(b"partial")
        if foreign_file:
            (target / "another-writer.txt").write_bytes(b"preserve me")
        raise OSError(errno.EIO, "injected copy failure")

    monkeypatch.setattr(utils.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        with utils.publication_directory(target, completion_marker=_MARKER) as staging:
            _write_stage(staging)
    assert source.read_bytes() == b"original checkpoint"
    if foreign_file:
        assert {path.name: path.read_bytes() for path in target.iterdir()} == {"another-writer.txt": b"preserve me"}
    else:
        assert not target.exists()
    assert not list(tmp_path.glob(".output.merge*"))


def test_permission_error_does_not_trigger_fallback(tmp_path, utils, rename_error):
    rename_error(errno.EACCES)
    target = tmp_path / "output"
    with pytest.raises(OSError) as failure:
        with utils.publication_directory(target, completion_marker=_MARKER) as staging:
            _write_stage(staging)
    assert failure.value.errno == errno.EACCES
    assert not target.exists()
    assert not list(tmp_path.glob(".output.merge*"))


def test_atomic_publication_remains_default(tmp_path, utils, rename_error):
    rename_error(errno.EINVAL)
    target = tmp_path / "output"
    with pytest.raises(OSError) as failure:
        with utils.publication_directory(target) as staging:
            _write_stage(staging)
    assert failure.value.errno == errno.EINVAL
    assert not target.exists()
    assert not list(tmp_path.glob(".output.merge*"))


@pytest.mark.parametrize("invalid_entry", ["directory", "symlink"])
def test_nonflat_stage_is_rejected_before_creating_target(tmp_path, utils, rename_error, invalid_entry):
    rename_error(errno.EINVAL)
    target = tmp_path / "output"
    with pytest.raises(ValueError, match="flat regular files"):
        with utils.publication_directory(target, completion_marker=_MARKER) as staging:
            _write_stage(staging)
            if invalid_entry == "directory":
                (staging / "nested").mkdir()
            else:
                (staging / "linked").symlink_to(staging / "adapter_config.json")
    assert not target.exists()
    assert not list(tmp_path.glob(".output.merge*"))
