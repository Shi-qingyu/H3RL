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
"""Exercise validation-ranked checkpointing through the direct-preference training loop."""

from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.trainer.diffusion import ray_diffusion_trainer
from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer


def test_ranks_synced_actor_by_validation_sample_mean(tmp_path, monkeypatch):
    trainer = object.__new__(DirectPreferenceRayTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "keep_best_validation_checkpoints": 3,
                "save_freq": 1,
                "test_freq": 2,
                "val_before_train": False,
                "log_val_generations": 0,
                "total_epochs": 1,
                "default_local_dir": str(tmp_path),
                "project_name": "test",
                "experiment_name": "ranking",
                "logger": ["console"],
                "esi_redundant_time": 0,
            },
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}, "agent": {"num_workers": 1}}},
        }
    )
    trainer.is_offline = True
    trainer.use_rm = trainer.use_reference_policy = trainer._has_old_adapter = False
    trainer.train_dataset = trainer.train_dataloader = [{"sample_level_scores": torch.tensor([90.0])}] * 2
    trainer.total_training_steps = len(trainer.train_dataloader)
    trainer.val_dataloader = [
        {"prompts": torch.zeros(len(group), 1), "data_source": np.array(group, dtype=object)}
        for group in (["source-a", "source-a"], ["source-b"])
    ]
    trainer.tokenizer = Mock()
    trainer.actor_rollout_wg = Mock()
    trainer.resource_pool_manager = Mock()
    trainer.resource_pool_manager.get_n_gpus.return_value = 1
    trainer._load_checkpoint = Mock()
    trainer._save_checkpoint = Mock()
    trainer._update_actor = Mock(return_value=DataProto.from_dict({}, meta_info={"metrics": {}}))
    trainer.checkpoint_manager = Mock()

    def generate_validation_samples(batch):
        assert trainer.checkpoint_manager.update_weights.call_args == call(2)
        assert trainer._update_actor.call_count == 2
        scores = [1.0 if source == "source-a" else 7.0 for source in batch.non_tensor_batch["data_source"]]
        return DataProto.from_dict(
            {"responses": torch.zeros(len(scores), 1), "rm_scores": torch.tensor(scores)[:, None]}
        )

    trainer.async_rollout_manager = Mock(generate_sequences=generate_validation_samples)
    rank_checkpoint = Mock()
    monkeypatch.setattr(ray_diffusion_trainer, "save_if_best_checkpoint", rank_checkpoint)

    trainer.fit()

    # Every video contributes once, including the lone video from the second source.
    rank_checkpoint.assert_called_once_with(
        str(tmp_path), 3, 2, pytest.approx(3.0), trainer._save_validation_checkpoint
    )
    trainer._save_checkpoint.assert_not_called()
