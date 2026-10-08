import pandas as pd
import numpy as np
from darl.data.window import create_sequential_windows
from darl.rl.env import DarlUpdateEnv
from darl.rl.transition_table import build_transition_row, get_reward

def test_create_sequential_windows():
    df = pd.DataFrame({'a': range(2500)})
    windows = create_sequential_windows(df, window_size=1000)

    assert len(windows) == 2
    assert len(windows[0]) == 1000
    assert len(windows[1]) == 1000
    assert windows[0]['a'].iloc[0] == 0
    assert windows[1]['a'].iloc[0] == 1000

def test_env():
    scenarios = pd.DataFrame({
        "state": [[0.5, 0.1, 0.05] for _ in range(12)],
        "A1_AUC": np.random.uniform(0.5, 0.7, 12),
        "A1_time": np.random.uniform(0, 1, 12),
        "A2_AUC": np.random.uniform(0.6, 0.8, 12),
        "A2_time": np.random.uniform(1, 2, 12),
        "A3_AUC": np.random.uniform(0.7, 0.9, 12),
        "A3_time": np.random.uniform(2, 5, 12),
        "A4_AUC": np.random.uniform(0.8, 1.0, 12),
        "A4_time": np.random.uniform(5, 10, 12),
    })

    env = DarlUpdateEnv(scenarios, episode_length=5)
    obs, info = env.reset()
    assert len(obs) == 5 # 3 state + 2 cost

    obs, reward, done, trunc, info = env.step(0)
    assert isinstance(reward, float)
    assert not done

    for _ in range(4):
        obs, reward, done, trunc, info = env.step(0)

    assert done
