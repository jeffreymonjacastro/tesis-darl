"""Custom DQN replay, learning and serialization tests."""

import numpy as np
import pandas as pd
import torch

from darl.rl import (
    DQNAgent,
    DQNConfig,
    ReplayBuffer,
    split_transition_episodes,
    train_dqn,
)


def _toy_transitions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(40):
        observation = rng.normal(size=28).astype(np.float32)
        for action in range(4):
            rows.append(
                {
                    "observation": observation.tolist(),
                    "action": action,
                    "reward": float(action == 2),
                    "next_observation": np.zeros(28, dtype=np.float32).tolist(),
                    "terminated": True,
                }
            )
    return pd.DataFrame(rows)


def test_replay_buffer_is_reproducible():
    first = ReplayBuffer(10, seed=42)
    second = ReplayBuffer(10, seed=42)
    for index in range(10):
        transition = (np.full(28, index), index % 4, index, np.zeros(28), False)
        first.add(*transition)
        second.add(*transition)
    assert np.array_equal(first.sample(4).actions, second.sample(4).actions)


def test_dqn_learns_best_action_and_round_trips(tmp_path):
    agent, history = train_dqn(
        _toy_transitions(),
        DQNAgent(DQNConfig()),
        updates=60,
        batch_size=32,
    )
    assert np.isfinite(history["loss"]).all()
    assert agent.select_action(np.zeros(28), epsilon=0.0) == 2
    target_before = [parameter.clone() for parameter in agent.target.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(target_before, agent.online.parameters()))
    path = tmp_path / "dqn.pt"
    agent.save(path)
    restored = DQNAgent.load(path)
    assert restored.select_action(np.zeros(28), epsilon=0.0) == 2


def test_transition_split_keeps_episodes_disjoint():
    transitions = _toy_transitions()
    transitions["episode_id"] = [f"episode-{index // 40}" for index in range(len(transitions))]
    train, validation = split_transition_episodes(transitions, validation_fraction=0.25)
    assert set(train["episode_id"]).isdisjoint(validation["episode_id"])
    assert len(train) + len(validation) == len(transitions)
