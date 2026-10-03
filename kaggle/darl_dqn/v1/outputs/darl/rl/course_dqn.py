"""Course-style DQN: two 128-unit layers, FIFO replay and hard target copies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from darl.rl.replay_buffer import ReplayBuffer
from darl.rl.training import QNetwork


@dataclass(frozen=True)
class CourseDQNConfig:
    """Hyperparameters matching the course DQN pattern with DARL dimensions."""

    state_dim: int = 28
    action_dim: int = 4
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    target_interval: int = 200
    replay_capacity: int = 10_000
    batch_size: int = 32
    seed: int = 42


class CourseDQN:
    """Online epsilon-greedy DQN with 24h-delayed transition insertion."""

    def __init__(self, config: CourseDQNConfig | None = None, device: str | None = None):
        self.config = config or CourseDQNConfig()
        torch.manual_seed(self.config.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(self.config.state_dim, self.config.action_dim, self.config.hidden).to(self.device)
        self.target = QNetwork(self.config.state_dim, self.config.action_dim, self.config.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.loss = nn.MSELoss()
        self.replay = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.epsilon = self.config.epsilon_start
        self.updates = 0

    def select(self, state: np.ndarray, greedy: bool = False) -> int:
        """Choose an action by epsilon-greedy exploration or greedy evaluation."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.config.action_dim))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def learn(self, matured: list[dict]) -> float | None:
        """Insert newly revealed transitions, then perform one Bellman update."""
        for item in matured:
            self.replay.add(item["observation"], item["action"], item["reward"], item["next_observation"], item["terminated"])
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        state = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        following = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=self.device)
        terminal = torch.as_tensor(batch.terminated, dtype=torch.float32, device=self.device)
        q = self.online(state).gather(1, action).squeeze(1)
        with torch.no_grad():
            target = reward + self.config.gamma * (1 - terminal) * self.target(following).max(dim=1).values
        loss = self.loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def end_episode(self) -> None:
        """Decay exploration at the episode boundary, as in the course lab."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
