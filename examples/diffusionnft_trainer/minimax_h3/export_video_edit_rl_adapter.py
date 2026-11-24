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
"""Export one H3 policy adapter from trusted FSDP2 training shards on CPU."""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.utils.serialization import config as serialization_config

if __package__:
    from .prepare_video_edit_sft_model import _MODEL_IO
else:
    from prepare_video_edit_sft_model import _MODEL_IO

_WEIGHTS = "pytorch_lora_weights.safetensors"
_FACTOR = re.compile(r"(.+\.lora_[AB])\.([^.]+)\.weight")


def _merge_factor(values: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    """Recover one factor using mesh coordinates and verify replicated copies."""
    first = values[0]
    if any(type(v) is not type(first) or tuple(v.shape) != shape or v.dtype != first.dtype for v in values):
        raise ValueError("Inconsistent tensor type, global shape or dtype across ranks")
    if not first.is_floating_point():
        raise ValueError("Expected floating-point LoRA factors")
    if type(first) is torch.Tensor:
        parts = values
        groups = [[rank] for rank in range(len(values))]
        shard_dim = None
    elif isinstance(first, DTensor):
        mesh = first.device_mesh.mesh
        placements = first.placements
        if mesh.ndim not in (1, 2) or sorted(mesh.flatten().tolist()) != list(range(len(values))):
            raise ValueError("Expected a 1D or 2D FSDP mesh covering all model ranks")
        if len(placements) != mesh.ndim or any(type(p) not in (Replicate, Shard) for p in placements):
            raise ValueError(f"Unsupported FSDP placement {placements}")
        if any(v.placements != placements or not torch.equal(v.device_mesh.mesh, mesh) for v in values):
            raise ValueError("Inconsistent DTensor mesh or placements across ranks")
        axes = [i for i, p in enumerate(placements) if isinstance(p, Shard)]
        if len(axes) > 1:
            raise ValueError(f"Expected at most one sharded dimension on the FSDP mesh, got {placements}")
        parts = [v.to_local() for v in values]
        if axes:
            axis = axes[0]
            shard_dim = placements[axis].dim
            if not 0 <= shard_dim < len(shape):
                raise ValueError(f"Invalid tensor sharding dimension {shard_dim}")
            groups = mesh.movedim(axis, -1).reshape(-1, mesh.size(axis)).tolist()
        else:
            groups = mesh.reshape(-1, 1).tolist()
            shard_dim = None
    else:
        raise ValueError(f"Unsupported factor tensor type {type(first)}")

    ordered = [parts[rank] for rank in groups[0]]
    for replica in groups[1:]:
        if any(not torch.equal(part, parts[rank]) for part, rank in zip(ordered, replica, strict=True)):
            raise ValueError("LoRA replicas disagree across ranks")
    for coordinate, part in enumerate(ordered):
        expected = list(shape)
        if shard_dim is not None:
            chunk = (shape[shard_dim] + len(ordered) - 1) // len(ordered)
            expected[shard_dim] = max(0, min(chunk, shape[shard_dim] - coordinate * chunk))
        if tuple(part.shape) != tuple(expected):
            raise ValueError(f"Invalid local factor shape {tuple(part.shape)}, expected {tuple(expected)}")
    merged = ordered[0].clone() if shard_dim is None else torch.cat(ordered, dim=shard_dim)
    if not torch.isfinite(merged).all():
        raise ValueError("Non-finite LoRA factor")
    return merged.contiguous()


def _read_rank(path: Path, adapter: str, shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    # Load metadata for the flat rank dictionary. mmap avoids reading base/other policy tensor data.
    # Derive offsets from the ZIP layout instead of seeking to thousands of headers on Ceph.
    with serialization_config.patch({"load.calculate_storage_offsets": True}):
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    factors = {}
    for key, value in state.items():
        match = _FACTOR.fullmatch(key.replace("_fsdp_wrapped_module.", "").removeprefix("base_model.model."))
        if match and match[2] == adapter:
            name = f"transformer.{match[1]}.weight"
            if name in factors:
                raise ValueError(f"Duplicate adapter factor {name} in {path}")
            factors[name] = value
    if factors.keys() != shapes.keys():
        missing = sorted(shapes.keys() - factors.keys())[:3]
        extra = sorted(factors.keys() - shapes.keys())[:3]
        raise ValueError(f"{path.name}: incomplete {adapter} adapter (missing={missing}, extra={extra})")
    return factors


def export_adapter(actor_dir: Path, adapter: str, template: Path, output_dir: Path) -> None:
    """Publish the selected policy using the SFT adapter's configuration and tensor schema."""
    actor_dir = actor_dir.resolve(strict=True)
    template = template.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if output_dir.resolve().is_relative_to(actor_dir.parent) or output_dir.resolve().is_relative_to(template):
        raise ValueError("Publish the adapter outside the training checkpoint and the SFT template")
    step = re.fullmatch(r"global_step_(\d+)", actor_dir.parent.name)
    if not step or adapter not in {"default", "old"}:
        raise ValueError("Expected global_step_N/actor and an adapter name of default or old")
    fsdp = _MODEL_IO.read_json(actor_dir / "fsdp_config.json")
    world = fsdp["world_size"]
    if fsdp.get("FSDP_version") != 2 or type(world) is not int or world < 1:
        raise ValueError(f"Invalid FSDP2 checkpoint configuration: {fsdp}")
    paths = [actor_dir / f"model_world_size_{world}_rank_{r}.pt" for r in range(world)]
    if set(paths) != set(actor_dir.glob("model_world_size_*_rank_*.pt")):
        raise ValueError("Checkpoint model shard set is incomplete or ambiguous")

    manifest = _MODEL_IO.read_json(template / "conversion_manifest.json")
    if manifest.get("format") != "minimax_h3_sft_lora" or manifest.get("base_weights_merged") is not False:
        raise ValueError("Expected a prepared H3 SFT adapter template with separate base weights")
    config = _MODEL_IO.read_json(template / "adapter_config.json")
    meta = _MODEL_IO.read_json(actor_dir / "lora_train_meta.json")
    if meta["r"] != config["r"] or meta["lora_alpha"] != config["lora_alpha"]:
        raise ValueError("FSDP checkpoint LoRA rank/alpha differ from the H3 template")
    base_transformer = (Path(manifest["base_model"]) / "transformer_ref").resolve()
    if Path(config["base_model_name_or_path"]).resolve() != base_transformer:
        raise ValueError("The SFT template declares inconsistent base model paths")
    saved_config = actor_dir / "huggingface/config.json"
    if saved_config.is_file():
        saved_base = _MODEL_IO.read_json(saved_config).get("_name_or_path")
        if saved_base and Path(saved_base).resolve() != base_transformer:
            raise ValueError(f"Training base {saved_base} differs from the evaluation base {base_transformer}")
    with safe_open(template / _WEIGHTS, framework="pt", device="cpu") as weights:
        shapes = {k: tuple(weights.get_slice(k).get_shape()) for k in weights.keys()}
    if not shapes or len(shapes) != manifest["converted_tensors"]:
        raise ValueError("Incomplete SFT template tensor schema")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with _MODEL_IO.publication_directory(output_dir, completion_marker="conversion_manifest.json") as stage:
        ranks = [_read_rank(path, adapter, shapes) for path in paths]
        tensors = {k: _merge_factor([rank.pop(k) for rank in ranks], shape) for k, shape in shapes.items()}
        metadata = {
            "format": "pt",
            "lora_adapter_metadata": json.dumps({f"transformer.{k}": v for k, v in config.items()}),
        }
        save_file(tensors, stage / _WEIGHTS, metadata=metadata)
        _MODEL_IO.write_json(stage / "adapter_config.json", config)
        _MODEL_IO.write_json(
            stage / "conversion_manifest.json",
            {
                "schema_version": 1,
                "format": "minimax_h3_rl_lora",
                "base_model": manifest["base_model"],
                "base_weights_merged": False,
                "checkpoint": str(actor_dir),
                "step": int(step[1]),
                "adapter": adapter,
                "rank": config["r"],
                "lora_alpha": config["lora_alpha"],
                "converted_tensors": len(tensors),
                "weight_name": _WEIGHTS,
            },
        )


def main() -> None:
    """Export the requested adapter into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", type=Path, required=True)
    parser.add_argument("--adapter", choices=("default", "old"), default="default")
    parser.add_argument("--template", type=Path, required=True, help="Prepared H3 SFT LoRA directory, used for schema.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_adapter(args.actor_dir, args.adapter, args.template, args.output_dir)
    print(f"Exported {args.adapter} from {args.actor_dir} to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
