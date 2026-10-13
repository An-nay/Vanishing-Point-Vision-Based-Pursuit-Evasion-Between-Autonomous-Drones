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
        self.obs_dim = self.model.observation_space.shape[0]
        if self.model.observation_space.shape not in ((21,), (30,)) or self.model.action_space.shape != (3,):
            raise ValueError("PPO opponent must use 21 (Stage 1) or 30 (obstacle) observations and 3 actions")

    def act(self, observation):
        # A Stage 1 (21-value) PPO dropped into an obstacle world simply cannot see the obstacles.
        observation = np.asarray(observation, dtype=np.float32)[..., :self.obs_dim]
        action, _ = self.model.predict(observation, deterministic=True)
        return np.clip(action, -1., 1.).astype(np.float32)


class CheckpointOpponent:
    def __init__(self, path, agent):
        from vp.policy_io import load_policy
        self.learner = load_policy(path)  # MADDPG or MAPPO checkpoint
        self.agent = agent

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            return self.learner.act(self.agent, obs)
        import torch
        with torch.no_grad():  # whole batch in one forward pass (used for critic targets)
            return np.clip(self.learner.actors[self.agent](torch.as_tensor(obs)).numpy(), -1., 1.).astype(np.float32)


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


class OpponentProvider:
    """Frozen policies for ONE agent: id 0 = a fixed policy (PPO Evader / pretrained Seeker), ids >= 1 = past snapshots.

    Snapshots are copies of the learner's own actor taken during training (an opponent pool). `batch` evaluates
    each row with the snapshot that actually played it, so critic targets stay consistent; rows whose snapshot
    was evicted come back as NaN and the learner falls back to its own target actor for them.
    """
    uses_ids = True

    def __init__(self, agent, action_dim, fixed=None, capacity=20, build_actor=None):
        self.agent, self.action_dim, self.fixed = agent, action_dim, fixed
        self.capacity, self.build_actor = capacity, build_actor
        self.nets, self.next_id = {}, 1

    def add_snapshot(self, state_dict):
        net = self.build_actor()
        net.load_state_dict(state_dict)
        net.requires_grad_(False)
        self.nets[self.next_id] = net
        self.next_id += 1
        while len(self.nets) > self.capacity:
            self.nets.pop(min(self.nets))
        return self.next_id - 1

    def snapshot_ids(self):
        return sorted(self.nets)

    def act(self, oid, observation):
        import torch
        if oid == 0:
            return self.fixed.act(observation)
        with torch.no_grad():
            action = self.nets[oid](torch.as_tensor(np.asarray(observation, dtype=np.float32)))
        return np.clip(action.numpy(), -1., 1.).astype(np.float32)

    def batch(self, observations, ids):
        import torch
        observations = np.asarray(observations, dtype=np.float32)
        out = np.full((len(ids), self.action_dim), np.nan, np.float32)
        for oid in np.unique(ids):
            rows = ids == oid
            if oid == 0:
                part = np.asarray(self.fixed.act(observations[rows]), dtype=np.float32)
            elif oid in self.nets:
                with torch.no_grad():
                    part = np.clip(self.nets[oid](torch.as_tensor(observations[rows])).numpy(), -1., 1.).astype(np.float32)
            else:
                part = None
            if part is not None:
                out[rows] = part
        return out
