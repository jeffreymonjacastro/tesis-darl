"""Custom PyTorch DQN and offline training helpers for DARL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from darl.rl.replay_buffer import ReplayBuffer

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Seed NumPy and PyTorch deterministically."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DQNConfig:
    """Serializable DQN hyperparameters and observation schema."""

    obs_dim: int = 28
    n_actions: int = 4
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    seed: int = SEED
    observation_schema: tuple[str, ...] = (
        "p_covariate",
        "p_concept",
        "p_both",
        "severity",
        "confidence",
        "delta_auc",
        "last_action_cost",
    )


class QNetwork(nn.Module):
    """Two-hidden-layer action-value approximator."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one action value per action."""
        return self.layers(observations)


class DQNAgent:
    """DQN with target network, replay updates and epsilon-greedy actions."""

    def __init__(self, config: DQNConfig | None = None):
        self.config = config or DQNConfig()
        set_global_seed(self.config.seed)
        self.online = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden
        )
        self.target = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden
        )
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.loss_function = nn.SmoothL1Loss()
        self.rng = np.random.default_rng(self.config.seed)

    def select_action(self, observation, epsilon: float = 0.0) -> int:
        """Select an epsilon-greedy discrete maintenance action."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.config.n_actions))
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> float:
        """Run one Bellman update and softly update target parameters."""
        batch = replay_buffer.sample(batch_size)
        observations = torch.as_tensor(batch.observations, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)
        next_observations = torch.as_tensor(batch.next_observations, dtype=torch.float32)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.float32)
        predicted = self.online(observations).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_values = self.target(next_observations).max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - terminated) * next_values
        loss = self.loss_function(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target.parameters(), self.online.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(self.config.tau * parameter)
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        """Save weights, hyperparameters and observation schema."""
        torch.save(
            {
                "config": asdict(self.config),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DQNAgent":
        """Restore a saved DQN agent."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        config_data = payload["config"]
        config_data["observation_schema"] = tuple(config_data["observation_schema"])
        agent = cls(DQNConfig(**config_data))
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        return agent


def replay_from_transitions(
    transitions: pd.DataFrame,
    capacity: int | None = None,
    seed: int = SEED,
) -> ReplayBuffer:
    """Convert long-form empirical transitions into replay memory."""
    buffer = ReplayBuffer(capacity or max(len(transitions), 1), seed=seed)
    for row in transitions.itertuples(index=False):
        buffer.add(
            row.observation,
            row.action,
            row.reward,
            row.next_observation,
            row.terminated,
        )
    return buffer


def split_transition_episodes(
    transitions: pd.DataFrame,
    validation_fraction: float = 0.25,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole empirical episodes, preventing transition leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    episode_ids = transitions["episode_id"].drop_duplicates().to_numpy()
    if len(episode_ids) < 2:
        raise ValueError("At least two episodes are required for train/validation")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episode_ids)
    n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_ids = set(shuffled[:n_validation])
    validation = transitions[transitions["episode_id"].isin(validation_ids)].copy()
    train = transitions[~transitions["episode_id"].isin(validation_ids)].copy()
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def train_dqn(
    transitions: pd.DataFrame,
    agent: DQNAgent | None = None,
    updates: int = 1000,
    batch_size: int = 64,
    seed: int = SEED,
) -> tuple[DQNAgent, pd.DataFrame]:
    """Train custom DQN offline and return per-update loss history."""
    if transitions.empty:
        raise ValueError("transitions must not be empty")
    set_global_seed(seed)
    trained = agent or DQNAgent(DQNConfig(seed=seed))
    replay = replay_from_transitions(transitions, seed=seed)
    effective_batch = min(batch_size, len(replay))
    losses = [trained.update(replay, effective_batch) for _ in range(updates)]
    return trained, pd.DataFrame({"update": np.arange(updates), "loss": losses})


def evaluate_policy(agent: DQNAgent, env, n_episodes: int = 1) -> pd.DataFrame:
    """Evaluate a deterministic DQN policy in the live environment."""
    rows: list[dict] = []
    for episode in range(n_episodes):
        observation, _ = env.reset(seed=SEED)
        terminated = False
        step = 0
        while not terminated:
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            terminated = bool(terminated or truncated)
            rows.append({"episode": episode, "step": step, "action": action, "reward": reward, **info})
            step += 1
    return pd.DataFrame(rows)
