"""
Environment utilities for MiniGrid FourRooms.

Paper Appendix A:
- "observation excludes a mission... compact MLP conditioned on flattened image"
- ImgObsWrapper: image-only obs, 7x7x3 = 147 dim (flattened)
- Custom reward: 1 - 0.5 * t/H (not MiniGrid default 1 - 0.9*t/H)
- max_steps = 100
- 50 fixed configurations
"""

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import ImgObsWrapper


NUM_CONFIGS = 50
MAX_STEPS = 100


class CustomRewardWrapper(gym.Wrapper):
    """Paper Appendix A: reward = 1 - 0.5 * t/H on success, 0 otherwise."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and reward > 0:
            steps = self.env.unwrapped.step_count
            reward = 1.0 - 0.5 * (steps / MAX_STEPS)
        return obs, reward, terminated, truncated, info


def make_env(seed=None):
    """Create FourRooms with ImgObsWrapper + custom reward."""
    env = gym.make("MiniGrid-FourRooms-v0", max_steps=MAX_STEPS)
    env = ImgObsWrapper(env)
    env = CustomRewardWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def obs_to_tensor(obs):
    """Flatten 7x7x3 image obs to 147-dim float vector, normalized to [0,1]."""
    return np.asarray(obs, dtype=np.float32).reshape(-1) / 10.0


def get_obs_dim():
    """Return obs dimension (147) and num actions (7)."""
    env = make_env(seed=0)
    obs, _ = env.reset()
    dim = obs_to_tensor(obs).shape[0]
    n_act = env.action_space.n
    env.close()
    return dim, n_act


if __name__ == "__main__":
    d, a = get_obs_dim()
    print(f"Obs dim: {d}, Num actions: {a}")
