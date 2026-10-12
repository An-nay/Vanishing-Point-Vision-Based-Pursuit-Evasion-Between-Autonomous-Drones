"""Frozen opponents; PPO remains separately loadable, never copied into MADDPG."""
from pathlib import Path
import numpy as np


class RandomEvader:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        shape = (*np.asarray(observation).shape[:-1], 3)
        return self.rng.uniform(-1., 1., shape).astype(np.float32)


class PPOEvader:
    def __init__(self, path):
        path = Path(path)
        if not path.is_file() and not path.with_suffix(".zip").is_file():
            raise FileNotFoundError(f"PPO opponent not found: {path}")
        from stable_baselines3 import PPO
        self.model = PPO.load(str(path), device="cpu")
        if self.model.observation_space.shape != (21,) or self.model.action_space.shape != (3,):
            raise ValueError("PPO opponent must use Stage 1's 21 observations and 3 actions")

    def act(self, observation):
        action, _ = self.model.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
        return np.clip(action, -1., 1.).astype(np.float32)


class CheckpointOpponent:
    def __init__(self, path, agent):
        from maddpg import MADDPG
        self.learner = MADDPG.load(path)
        self.agent = agent

    def act(self, observation):
        obs = np.asarray(observation)
        if obs.ndim == 1:
            return self.learner.act(self.agent, obs)
        return np.stack([self.learner.act(self.agent, row) for row in obs])


def make_evader(kind, seed, model_path=None):
    if kind == "random":
        return RandomEvader(seed)
    if not model_path:
        raise ValueError(f"--opponent-model is required for {kind}")
    if kind == "ppo":
        return PPOEvader(model_path)
    if kind == "maddpg":
        return CheckpointOpponent(model_path, "evader")
    raise ValueError(f"unknown opponent: {kind}")
