"""Fixed-size replay memory for offline DARL DQN training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    """Vectorized transition batch sampled from replay memory."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray


class ReplayBuffer:
    """Deterministic bounded replay buffer."""

    def __init__(self, capacity: int = 100_000, seed: int = 42):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[tuple] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        observation,
        action: int,
        reward: float,
        next_observation,
        terminated: bool,
    ) -> None:
        """Append one transition."""
        self._items.append(
            (
                np.asarray(observation, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.float32),
                bool(terminated),
            )
        )

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample transitions uniformly without replacement."""
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds replay buffer size")
        indices = self._rng.choice(len(self._items), size=batch_size, replace=False)
        rows = [self._items[index] for index in indices]
        observations, actions, rewards, next_observations, terminated = zip(*rows)
        return TransitionBatch(
            np.stack(observations),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_observations),
            np.asarray(terminated, dtype=np.float32),
        )
