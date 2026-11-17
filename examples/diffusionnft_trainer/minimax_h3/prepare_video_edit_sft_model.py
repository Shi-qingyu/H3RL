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
"""Fuse a native DiffSynth H3 SFT LoRA into actor and rollout base checkpoints.

The source layout follows DiffSynthStudio's ``minimax_h3_dit.py`` and its
trainable-only checkpoint exporter. QKV rows are interleaved by attention head;
native FC1 uses [gate, up], whereas Diffusers uses [up, gate]. Assets remain
symlinked to the official release. Both model components receive the SFT update,
so an NFT reference with its training adapter disabled also evaluates the SFT.
"""

import argparse
import importlib.util
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Import the existing offline helpers without triggering verl_omni's GPU registration imports.
_UTILS_PATH = Path(__file__).resolve().parents[3] / "verl_omni/model_merger/utils.py"
_UTILS_SPEC = importlib.util.spec_from_file_location("_h3_model_publishing_utils", _UTILS_PATH)
if _UTILS_SPEC is None or _UTILS_SPEC.loader is None:
    raise ImportError(f"Unable to load model publishing helpers from {_UTILS_PATH}.")
_MODEL_IO = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(_MODEL_IO)

_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
_LORA_KEY = re.compile(
    r"(?P<module>(?:blocks\.\d+|token_refiner\.blocks\.\d+)\."
    r"(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)))\.lora_(?P<part>[AB])\.default\.weight"
)


def _config_dimensions(base_model: Path) -> dict[str, int]:
    native = _MODEL_IO.read_json(base_model / "Ref2VA/transformer/config.json")
    actor = _MODEL_IO.read_json(base_model / "transformer_ref/config.json")
    if native.get("_class_name") != "MiniMaxH3DiTModel":
        raise ValueError("Ref2VA/transformer must be the native MiniMaxH3DiTModel.")
    if actor.get("_class_name") != "MiniMaxH3Transformer3DModel":
        raise ValueError("transformer_ref must be the Diffusers MiniMaxH3Transformer3DModel.")
    fields = {
        "hidden_size": "hidden_size",
        "num_layers": "num_layers",
        "token_refiner_num_layers": "num_refiner_layers",
        "num_attention_heads": "num_attention_heads",
        "attention_head_dim": "attention_head_dim",
        "ffn_hidden_size": "ffn_dim",
        "latents_dim": "in_channels",
        "audio_latents_dim": "audio_in_channels",
        "text_dim": "text_dim",
        "timestep_input_dim": "freq_dim",
        "time_embed_hidden_size": "time_embed_hidden_dim",
        "time_embed_dim": "time_embed_dim",
        "rope_inv_freq_len": "rope_freq_dim",
    }
    dimensions = {}
    for native_name, actor_name in fields.items():
        value = native.get(native_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value != actor.get(actor_name):
            raise ValueError(f"Native/Diffusers H3 configuration mismatch: {native_name}/{actor_name}.")
        dimensions[native_name] = value
    for field in ("patch_size", "norm_eps", "qk_norm_eps", "final_norm_eps"):
        if field not in native or native[field] != actor.get(field):
            raise ValueError(f"Native/Diffusers H3 configuration mismatch: {field}.")
    return dimensions


def _load_lora(checkpoint: Path, dimensions: dict[str, int]):
    tensors = load_file(str(checkpoint), device="cpu")
    pairs = {}
    for name, tensor in tensors.items():
        match = _LORA_KEY.fullmatch(name)
        if match is None:
            raise ValueError(f"Unsupported SFT checkpoint tensor: {name}; expected native H3 LoRA A/B weights.")
        if tensor.ndim != 2 or not tensor.dtype.is_floating_point or not torch.isfinite(tensor).all():
            raise ValueError(f"LoRA tensor must be a finite floating-point matrix: {name}.")
        pairs.setdefault(match["module"], {})[match["part"]] = tensor
    blocks = [f"blocks.{index}" for index in range(dimensions["num_layers"])]
    blocks += [f"token_refiner.blocks.{index}" for index in range(dimensions["token_refiner_num_layers"])]
    expected = {f"{block}.{target}" for block in blocks for target in _TARGETS}
    if set(pairs) != expected:
        raise ValueError(
            f"Incomplete SFT LoRA module coverage: missing={sorted(expected - set(pairs))[:8]}, "
            f"unexpected={sorted(set(pairs) - expected)[:8]}."
        )
    hidden = dimensions["hidden_size"]
    inner = dimensions["num_attention_heads"] * dimensions["attention_head_dim"]
    ffn = dimensions["ffn_hidden_size"]
    shapes = {
        "attn.qkv_proj": (hidden, 3 * inner),
        "attn.out_proj": (inner, hidden),
        "mlp.fc1": (hidden, 2 * ffn),
        "mlp.fc2": (ffn, hidden),
    }
    rank = None
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing LoRA A/B pair: {name}.")
        a, b = pair["A"], pair["B"]
        module_rank = a.shape[0]
        if module_rank <= 0 or (rank is not None and module_rank != rank):
            raise ValueError(f"Inconsistent or empty SFT LoRA rank: {name}.")
        rank = module_rank
        target = next(target for target in _TARGETS if name.endswith(f".{target}"))
        input_size, output_size = shapes[target]
        if tuple(a.shape) != (rank, input_size) or tuple(b.shape) != (output_size, rank):
            raise ValueError(f"SFT LoRA shape does not match the base model: {name}.")
    return pairs, rank, len(tensors)


def _weight_map(directory: Path, index_name: str) -> dict[str, str]:
    mapping = _MODEL_IO.read_json(directory / index_name).get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Missing sharded weight_map in {directory / index_name}.")
    for name, shard in mapping.items():
        if (
            not isinstance(name, str)
            or not isinstance(shard, str)
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
            or not (directory / shard).is_file()
        ):
            raise ValueError(f"Invalid checkpoint shard mapping: {name!r}: {shard!r}.")
    if {path.name for path in directory.glob("*.safetensors")} != set(mapping.values()):
        raise ValueError(f"Safetensors files do not match the shard index: {directory}.")
    return mapping


def _plans(pairs):
    native = {f"{name}.weight": (name, "identity", None) for name in pairs}
    actor = {}
    for name in pairs:
        renamed = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        if renamed.startswith("blocks."):
            renamed = "transformer_blocks." + renamed.removeprefix("blocks.")
        if renamed.endswith(".attn.qkv_proj"):
            prefix = renamed.removesuffix("qkv_proj")
            for index, projection in enumerate(("q", "k", "v")):
                actor[f"{prefix}to_{projection}.weight"] = (name, "qkv", index)
        elif renamed.endswith(".mlp.fc1"):
            actor[renamed.removesuffix("mlp.fc1") + "ff.net.0.proj.weight"] = (name, "fc1", None)
        elif renamed.endswith(".mlp.fc2"):
            actor[renamed.removesuffix("mlp.fc2") + "ff.net.2.weight"] = (name, "identity", None)
        else:
            actor[renamed.removesuffix("attn.out_proj") + "attn.to_out.0.weight"] = (name, "identity", None)
    return native, actor


def _clone_entries(source: Path, target: Path, rewritten: set[str]) -> None:
    target.mkdir()
    for entry in source.iterdir():
        if entry.name in rewritten:
            continue
        destination = target / entry.name
        if entry.is_file() and entry.suffix == ".json":
            shutil.copy2(entry, destination)
        else:
            destination.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())


def _write_component(source, target, index_name, plan, pairs, scale, dimensions):
    mapping = _weight_map(source, index_name)
    if not set(plan) <= set(mapping):
        raise ValueError(f"Base checkpoint lacks trained LoRA targets: {sorted(set(plan) - set(mapping))[:8]}.")
    shards = set(mapping.values())
    _clone_entries(source, target, shards)
    seen = set()
    for shard in sorted(shards):
        shard_keys = {name for name, filename in mapping.items() if filename == shard}
        with safe_open(source / shard, framework="pt", device="cpu") as archive:
            if set(archive.keys()) != shard_keys:
                raise ValueError(f"Weight names disagree with the index in {source / shard}.")
            trained = shard_keys & set(plan)
            if not trained:
                (target / shard).symlink_to((source / shard).resolve())
                continue
            output = {}
            for name in sorted(shard_keys):
                value = archive.get_tensor(name)
                if name in trained:
                    if not value.dtype.is_floating_point or not torch.isfinite(value).all():
                        raise ValueError(f"Base weight must be a finite floating-point tensor: {name}.")
                    module, kind, projection = plan[name]
                    a, b = pairs[module]["A"], pairs[module]["B"]
                    if kind == "qkv":
                        b = b.reshape(
                            dimensions["num_attention_heads"], 3, dimensions["attention_head_dim"], a.shape[0]
                        )[:, projection].reshape(-1, a.shape[0])
                    elif kind == "fc1":
                        gate, up = b.split(dimensions["ffn_hidden_size"], dim=0)
                        b = torch.cat((up, gate), dim=0)
                    elif kind != "identity":
                        raise ValueError(f"Unsupported H3 LoRA mapping: {kind}.")
                    if tuple(value.shape) != (b.shape[0], a.shape[1]):
                        raise ValueError(f"Base and SFT update shapes disagree for {name}.")
                    # Out-of-place addition also protects FP32 tensors backed by the source mmap.
                    merged = value.float() + (b.float() @ a.float()) * scale
                    if not torch.isfinite(merged).all():
                        raise ValueError(f"SFT fusion produced a non-finite value for {name}.")
                    value = merged.to(dtype=value.dtype)
                    if not torch.isfinite(value).all():
                        raise ValueError(f"SFT fusion overflowed the base dtype for {name}.")
                    del merged
                    seen.add(name)
                output[name] = value.contiguous()
            save_file(output, str(target / shard), metadata=archive.metadata())
            del output
    if seen != set(plan):
        raise ValueError(f"SFT fusion did not apply every planned update: {sorted(set(plan) - seen)[:8]}.")
    return {"base_tensors": len(mapping), "updated_tensors": len(seen), "shards": len(shards)}


def main() -> None:
    """Write a fused SFT base for both NFT actor/reference and native rollout."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    args = parser.parse_args()
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        parser.error("--lora-alpha must be finite and positive.")
    base_model = args.base_model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory must not exist: {output_dir}")
    output_dir = output_dir.resolve()
    if output_dir == base_model or output_dir.is_relative_to(base_model):
        raise ValueError("Output directory must be outside the source model tree.")
    dimensions = _config_dimensions(base_model)
    pairs, rank, tensor_count = _load_lora(checkpoint, dimensions)
    native_plan, actor_plan = _plans(pairs)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with _MODEL_IO.publication_directory(output_dir) as staging:
        # The staging directory already exists; mirror the release's other top-level assets.
        for entry in base_model.iterdir():
            if entry.name not in {"Ref2VA", "transformer_ref"}:
                destination = staging / entry.name
                if entry.is_file() and entry.suffix == ".json":
                    shutil.copy2(entry, destination)
                else:
                    destination.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        _clone_entries(base_model / "Ref2VA", staging / "Ref2VA", {"transformer"})
        scale = args.lora_alpha / rank
        native = _write_component(
            base_model / "Ref2VA/transformer",
            staging / "Ref2VA/transformer",
            "model.safetensors.index.json",
            native_plan,
            pairs,
            scale,
            dimensions,
        )
        actor = _write_component(
            base_model / "transformer_ref",
            staging / "transformer_ref",
            "diffusion_pytorch_model.safetensors.index.json",
            actor_plan,
            pairs,
            scale,
            dimensions,
        )
        manifest = {
            "schema_version": 1,
            "format": "minimax_h3_fused_sft_base",
            "base_model": str(base_model),
            "checkpoint": str(checkpoint),
            "rank": rank,
            "lora_alpha": args.lora_alpha,
            "lora_scale": scale,
            "lora_tensors": tensor_count,
            "lora_modules": len(pairs),
            "native": native,
            "actor": actor,
            "qkv_layout": "head_projection_channel",
            "native_fc1_layout": ["gate", "up"],
            "actor_fc1_layout": ["up", "gate"],
            "reference_policy": "fused_sft_base_with_training_adapter_disabled",
        }
        _MODEL_IO.write_json(staging / "conversion_manifest.json", manifest)
    print(f"Wrote fused H3 SFT base to {output_dir}; use it as MODEL_PATH without LORA_WARMSTART_PATH.")


if __name__ == "__main__":
    main()
