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
"""CPU checks for exporting H3 RL policies without loading the base model."""

import json

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from examples.diffusionnft_trainer.minimax_h3.export_video_edit_rl_adapter import _merge_factor, export_adapter


@pytest.mark.parametrize("corrupt", [False, True])
def test_hsdp_uses_mesh_coordinates_and_checks_replicas(corrupt):
    # One logical process group is enough to construct serialized local DTensor shards.
    # No distributed communication is used to reconstruct them.
    dist.init_process_group("fake", rank=0, world_size=24, store=FakeStore())
    try:
        grid = torch.arange(24).reshape(8, 3).T.contiguous()
        mesh = DeviceMesh("cpu", grid, mesh_dim_names=("ddp", "fsdp"), _init_backend=False)
        original = torch.arange(38, dtype=torch.float32).reshape(19, 2)
        ranks = [None] * 24
        for replica in grid:
            for index, rank in enumerate(replica.tolist()):
                part = original[index * 3 : (index + 1) * 3].clone()
                if corrupt and rank == 1:
                    part += 1
                ranks[rank] = DTensor.from_local(
                    part, mesh, (Replicate(), Shard(0)), shape=original.shape, stride=original.stride()
                )
        if corrupt:
            with pytest.raises(ValueError, match="replicas disagree"):
                _merge_factor(ranks, tuple(original.shape))
        else:
            torch.testing.assert_close(_merge_factor(ranks, tuple(original.shape)), original)
    finally:
        dist.destroy_process_group()


def test_select_named_policy_and_require_all_files(tmp_path):
    template = tmp_path / "sft"
    template.mkdir()
    stem = "transformer_blocks.0.attn.to_q.lora_A"
    name = f"transformer.{stem}.weight"
    save_file({name: torch.zeros(2, 4)}, template / "pytorch_lora_weights.safetensors")
    (template / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2, "base_model_name_or_path": "/h3/transformer_ref"})
    )
    (template / "conversion_manifest.json").write_text(
        json.dumps(
            {"format": "minimax_h3_sft_lora", "base_weights_merged": False, "converted_tensors": 1, "base_model": "/h3"}
        )
    )
    actor = tmp_path / "global_step_5/actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text('{"world_size":2,"FSDP_version":2}')
    (actor / "lora_train_meta.json").write_text('{"r":2,"lora_alpha":2}')
    state = {
        f"{stem}.old.weight": torch.full((2, 4), 7.0),
        f"{stem}.default.weight": torch.ones(2, 4),
        "attn.to_q.base_layer.weight": torch.zeros(4, 4),
    }
    for rank in range(2):
        torch.save(state, actor / f"model_world_size_2_rank_{rank}.pt")
    for adapter in ("old", "default"):
        result = tmp_path / adapter
        export_adapter(actor, adapter, template, result)
        metadata = json.loads((result / "conversion_manifest.json").read_text())
        assert (metadata["adapter"], metadata["step"], metadata["format"]) == (adapter, 5, "minimax_h3_rl_lora")
        published = load_file(result / "pytorch_lora_weights.safetensors")
        assert set(published) == {name}
        torch.testing.assert_close(published[name], state[f"{stem}.{adapter}.weight"])

    hf_dir = actor / "huggingface"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text('{"_name_or_path":"/another/transformer_ref"}')
    with pytest.raises(ValueError, match="Training base"):
        export_adapter(actor, "old", template, tmp_path / "wrong_base")

    (actor / "model_world_size_2_rank_1.pt").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        export_adapter(actor, "old", template, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
