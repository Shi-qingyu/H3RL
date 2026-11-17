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
"""Convert DiffSynth H3 SFT LoRA tensors for Diffusers without merging base weights.

The source layout follows DiffSynthStudio's native ``minimax_h3_dit.py``. Reuse
the sibling exporter's configuration and LoRA schema checks, then only reshape
and rename adapter factors. The result loads with the default transformer prefix
of Diffusers ``load_lora_adapter``; the official H3 weights remain separate.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

_HELPERS_PATH = Path(__file__).with_name("prepare_video_edit_sft_model.py")
_HELPERS_SPEC = importlib.util.spec_from_file_location("_h3_sft_checkpoint_helpers", _HELPERS_PATH)
if _HELPERS_SPEC is None or _HELPERS_SPEC.loader is None:
    raise ImportError(f"Unable to load H3 checkpoint helpers from {_HELPERS_PATH}.")
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)

_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")


def _convert_lora(pairs, dimensions: dict[str, int]) -> dict[str, torch.Tensor]:
    _, actor_plan = _HELPERS._plans(pairs)
    output = {}
    consumed = set()
    for actor_weight, (source_module, kind, projection) in sorted(actor_plan.items()):
        a, b = pairs[source_module]["A"], pairs[source_module]["B"]
        if kind == "qkv":
            # Native QKV is [head, projection, channel], not three contiguous blocks.
            b = b.reshape(dimensions["num_attention_heads"], 3, dimensions["attention_head_dim"], a.shape[0])
            b = b[:, projection].reshape(-1, a.shape[0])
        elif kind == "fc1":
            gate, up = b.split(dimensions["ffn_hidden_size"], dim=0)
            b = torch.cat((up, gate), dim=0)
        elif kind != "identity":
            raise ValueError(f"Unsupported H3 LoRA mapping: {kind}.")
        prefix = "transformer." + actor_weight.removesuffix(".weight")
        # Q/K/V share the source A storage; safetensors requires independent entries.
        output[f"{prefix}.lora_A.weight"] = a.clone().contiguous()
        output[f"{prefix}.lora_B.weight"] = b.clone().contiguous()
        consumed.add(source_module)
    if consumed != set(pairs) or len(output) != 2 * len(actor_plan):
        raise ValueError("SFT LoRA conversion did not cover every source and target module exactly.")
    return output


def main() -> None:
    """Publish a standalone SFT LoRA directory without reading any base weight shards."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()
    if args.lora_alpha <= 0:
        parser.error("--lora-alpha must be a positive integer.")
    base_model = args.base_model.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory must not exist: {output_dir}")
    output_dir = output_dir.resolve()
    if output_dir == base_model or output_dir.is_relative_to(base_model):
        raise ValueError("Output directory must be outside the official model tree.")
    dimensions = _HELPERS._config_dimensions(base_model)
    pairs, rank, source_tensor_count = _HELPERS._load_lora(checkpoint, dimensions)
    output = _convert_lora(pairs, dimensions)
    config = {
        "peft_type": "LORA",
        "task_type": None,
        "base_model_name_or_path": str(base_model / "transformer_ref"),
        "inference_mode": False,
        "r": rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": list(_TARGET_MODULES),
        "fan_in_fan_out": False,
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
    }
    # Diffusers filters both state keys and adapter metadata by its transformer prefix.
    metadata = {
        "format": "pt",
        "lora_adapter_metadata": json.dumps({f"transformer.{key}": value for key, value in config.items()}),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with _HELPERS._MODEL_IO.publication_directory(output_dir, completion_marker="conversion_manifest.json") as staging:
        save_file(output, str(staging / _WEIGHT_NAME), metadata=metadata)
        _HELPERS._MODEL_IO.write_json(staging / "adapter_config.json", config)
        _HELPERS._MODEL_IO.write_json(
            staging / "conversion_manifest.json",
            {
                "schema_version": 1,
                "format": "minimax_h3_sft_lora",
                "base_model": str(base_model),
                "checkpoint": str(checkpoint),
                "rank": rank,
                "lora_alpha": args.lora_alpha,
                "source_tensors": source_tensor_count,
                "source_modules": len(pairs),
                "converted_tensors": len(output),
                "weight_name": _WEIGHT_NAME,
                "loader_prefix": "transformer",
                "qkv_layout": "head_projection_channel",
                "base_weights_merged": False,
            },
        )
    print(f"Wrote H3 SFT LoRA to {output_dir}; keep MODEL_PATH pointing to the official H3 root.")


if __name__ == "__main__":
    main()
