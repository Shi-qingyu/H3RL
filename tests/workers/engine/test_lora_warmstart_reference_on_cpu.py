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
"""Warmstarted LoRA references retain the initial policy during actor updates."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


class _AdapterModule(torch.nn.Module):
    """Implement the adapter boundary with scalar CPU parameters."""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.weights = torch.nn.ParameterDict()
        self.peft_config = {}
        self.active = None
        self.enabled = True
        self.loads = []
        self.additions = []

    def load_lora_adapter(self, path, *, adapter_name, **kwargs):
        """Load deterministic fake checkpoint values and its saved configuration."""
        self.loads.append((adapter_name, path, kwargs))
        self.peft_config[adapter_name] = SimpleNamespace(r=3, lora_alpha=9, target_modules={"projection"})
        self.weights[adapter_name] = torch.nn.Parameter(torch.tensor(2.0))
        self.set_adapter(adapter_name)

    def add_adapter(self, config, *, adapter_name):
        """Create a fresh policy adapter with the supplied configuration."""
        self.additions.append(adapter_name)
        self.peft_config[adapter_name] = config
        self.weights[adapter_name] = torch.nn.Parameter(torch.tensor(0.0))
        self.set_adapter(adapter_name)

    def set_adapter(self, name):
        """Model PEFT's selection-dependent parameter trainability."""
        if name not in self.peft_config:
            raise ValueError(f"Unknown adapter {name!r}")
        self.active = name
        for adapter_name, weight in self.weights.items():
            weight.requires_grad_(adapter_name == name)

    def disable_adapters(self):
        """Select the base-only forward path."""
        self.enabled = False

    def enable_adapters(self):
        """Restore the selected adapter after a base-only forward."""
        self.enabled = True

    def forward(self, inputs):
        """Apply the base scalar and the selected adapter's update."""
        delta = self.weights[self.active] if self.enabled else 0.0
        return inputs * (self.base + delta)


class _WrappedModule(torch.nn.Module):
    """Expose the same wrapped-module boundary as an FSDP1 engine."""

    def __init__(self, module):
        super().__init__()
        self._fsdp_wrapped_module = module

    def disable_adapters(self):
        """Forward the adapter-disable operation to the wrapped model."""
        self._fsdp_wrapped_module.disable_adapters()

    def enable_adapters(self):
        """Forward the adapter-enable operation to the wrapped model."""
        self._fsdp_wrapped_module.enable_adapters()

    def forward(self, inputs):
        """Run the wrapped model's scalar forward."""
        return self._fsdp_wrapped_module(inputs)


class _Engine(LoRAAdapterMixin):
    def __init__(self, model_config):
        self.model_config = model_config


@pytest.fixture
def build_engine(monkeypatch):
    """Build fake adapters without reading checkpoint files or creating FSDP."""
    import verl.utils.fs as filesystem

    checkpoint = "/mock/video_edit_sft"
    monkeypatch.setattr(filesystem, "copy_to_local", lambda path, **kwargs: path)

    def build(*, warmstart=True, reference=True, diffusers_weights=False, wrapped=False):
        weights_path = Path(checkpoint) / "pytorch_lora_weights.safetensors"
        adapters = ("default", "old", "reference") if reference else ("default", "old")
        config = SimpleNamespace(
            lora_rank=8,
            lora_alpha=16,
            lora_adapter_path=checkpoint if warmstart else None,
            policy_state_adapters=adapters,
            lora={},
            use_shm=False,
            lora_init_weights="gaussian",
            target_modules=["projection"],
            target_parameters=None,
            exclude_modules=None,
            lora_dtype=None,
        )
        engine = _Engine(config)
        module = _AdapterModule()
        with monkeypatch.context() as filesystem_patch:
            filesystem_patch.setattr(Path, "is_file", lambda path: diffusers_weights and path == weights_path)
            engine.module = engine._build_lora_module(module)
        if wrapped:
            engine.module = _WrappedModule(module)
        return engine, module

    return build


@pytest.mark.parametrize("diffusers_weights", [False, True])
def test_warmstart_loads_matching_reference_and_leaves_default_trainable(build_engine, diffusers_weights):
    """Both snapshots use the saved LoRA rather than fresh random adapters."""
    _, module = build_engine(diffusers_weights=diffusers_weights)
    assert [name for name, _, _ in module.loads] == ["default", "reference"]
    assert {path for _, path, _ in module.loads} == {"/mock/video_edit_sft"}
    expected_kwargs = (
        {"weight_name": "pytorch_lora_weights.safetensors", "use_safetensors": True} if diffusers_weights else {}
    )
    assert all(kwargs == expected_kwargs for _, _, kwargs in module.loads)
    assert module.additions == ["old"]
    for name in ("old", "reference"):
        assert module.peft_config[name].r == module.peft_config["default"].r == 3
        assert module.peft_config[name].lora_alpha == module.peft_config["default"].lora_alpha == 9
    torch.testing.assert_close(module.weights["reference"], module.weights["default"])
    assert module.active == "default"
    assert module.weights["default"].requires_grad
    assert not module.weights["old"].requires_grad
    assert not module.weights["reference"].requires_grad
    assert not module.base.requires_grad


def test_warmstart_without_reference_state_preserves_existing_adapter_set(build_engine):
    _, module = build_engine(reference=False)
    assert [name for name, _, _ in module.loads] == ["default"]
    assert set(module.peft_config) == {"default", "old"}
    assert module.active == "default"


@pytest.mark.parametrize("wrapped", [False, True])
def test_real_reference_context_restores_default_after_exception(build_engine, wrapped):
    engine, module = build_engine(wrapped=wrapped)
    with pytest.raises(RuntimeError, match="forward failed"), engine.use_adapter("reference"):
        assert module.active == "reference"
        assert module.enabled
        raise RuntimeError("forward failed")
    assert module.active == "default"
    assert module.weights["default"].requires_grad
    assert not module.weights["reference"].requires_grad


@pytest.mark.parametrize("warmstart", [False, True])
def test_missing_named_reference_uses_base_and_reenables_after_exception(build_engine, warmstart):
    engine, module = build_engine(warmstart=warmstart, reference=not warmstart)
    assert "reference" not in module.peft_config
    with pytest.raises(RuntimeError, match="forward failed"), engine.use_adapter("reference"):
        assert not module.enabled
        torch.testing.assert_close(engine.module(torch.tensor(1.0)), torch.tensor(1.0))
        raise RuntimeError("forward failed")
    assert module.enabled
    assert module.active == "default"


@pytest.mark.parametrize("wrapped", [False, True])
def test_actor_optimizer_step_keeps_initial_reference_unchanged(build_engine, wrapped):
    """Selecting the frozen policy for no-grad inference must not update it."""
    engine, module = build_engine(wrapped=wrapped)
    optimizer = torch.optim.SGD(engine.module.parameters(), lr=0.1)
    initial_reference = module.weights["reference"].detach().clone()
    initial_actor = module.weights["default"].detach().clone()
    inputs = torch.tensor(1.0)

    with torch.no_grad(), engine.use_adapter("reference"):
        expected_reference = engine.module(inputs).clone()
    actor_prediction = engine.module(inputs)
    with torch.no_grad(), engine.use_adapter("reference"):
        reference_prediction = engine.module(inputs)
    assert not reference_prediction.requires_grad
    actor_prediction.square().backward()
    assert module.weights["default"].grad is not None
    assert module.weights["old"].grad is None
    assert module.weights["reference"].grad is None
    optimizer.step()

    assert not torch.allclose(module.weights["default"], initial_actor)
    torch.testing.assert_close(module.weights["reference"], initial_reference, rtol=0, atol=0)
    with torch.no_grad(), engine.use_adapter("reference"):
        torch.testing.assert_close(engine.module(inputs), expected_reference, rtol=0, atol=0)
    assert module.active == "default"
    assert not module.weights["reference"].requires_grad


@pytest.mark.parametrize("method", ["copy_adapter", "ema_update_adapter"])
def test_reference_cannot_be_overwritten_by_policy_updates(build_engine, method):
    engine, module = build_engine()
    reference = module.weights["reference"].detach().clone()
    with pytest.raises(ValueError, match="reference"):
        getattr(engine, method)(source="default", target="reference")
    torch.testing.assert_close(module.weights["reference"], reference, rtol=0, atol=0)
    assert module.active == "default"
