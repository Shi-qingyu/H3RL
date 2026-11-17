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
"""Group V6 native video shapes into complete RL prompt batches."""

import math
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from torch.utils.data import Sampler


class VideoEditShapeSampler(Sampler[int]):
    """Shuffle homogeneous batches and repeat each bucket's tail without dropping sources.

    Rollout repeats each prompt ``n`` times after sampling. A global prompt batch
    must therefore have one frame count and canvas before it is split across
    rollout workers and actor data-parallel ranks.
    """

    def __init__(self, data_source: Any, data_config: Any) -> None:
        self.batch_size = int(data_config.get("gen_batch_size") or data_config.train_batch_size)
        if self.batch_size <= 0:
            raise ValueError("VideoEditShapeSampler requires a positive prompt batch size.")
        self.seed = int(data_config.get("seed") or 0)
        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, info in enumerate(data_source.dataframe["extra_info"]):
            shape = info.get("rollout_shape")
            if not isinstance(shape, dict):
                raise ValueError("Prepare V6 data with per-sample rollout_shape before using this sampler.")
            key = tuple(shape[name] for name in ("num_frames", "height", "width"))
            if any(type(value) is not int or value <= 0 for value in key):
                raise ValueError(f"Invalid rollout shape at dataset row {index}: {key}.")
            self.buckets[key].append(index)
        self.epoch = 0
        self.position = 0

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) * self.batch_size for indices in self.buckets.values())

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for key in sorted(self.buckets):
            indices = list(self.buckets[key])
            rng.shuffle(indices)
            remainder = (-len(indices)) % self.batch_size
            if remainder:
                indices.extend(rng.choices(indices, k=remainder))
            batches.extend(
                indices[start : start + self.batch_size] for start in range(0, len(indices), self.batch_size)
            )
        rng.shuffle(batches)
        order = [index for batch in batches for index in batch]
        while self.position < len(order):
            index = order[self.position]
            self.position += 1
            yield index
        self.position = 0
        self.epoch += 1

    def state_dict(self) -> dict[str, int]:
        """Save the current shuffled epoch and cursor for StatefulDataLoader."""
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore the same batch order and next sample on checkpoint resume."""
        self.epoch = int(state_dict["epoch"])
        self.position = int(state_dict["position"])
        if self.epoch < 0 or not 0 <= self.position <= len(self):
            raise ValueError("Invalid VideoEditShapeSampler checkpoint state.")
