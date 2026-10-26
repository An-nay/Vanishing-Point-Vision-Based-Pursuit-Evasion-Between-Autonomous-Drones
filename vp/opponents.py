from pathlib import Path
import numpy as np


class RandEvader:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        shape = (*np.asarray(observation).shape[:-1], 3)
        return self.rng.uniform(-1., 1., shape).astype(np.float32)


class PPOOpp:
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
        observation = np.asarray(observation, dtype=np.float32)[..., :self.obs_dim]
        action, _ = self.model.predict(observation, deterministic=True)
        return np.clip(action, -1., 1.).astype(np.float32)


class SavedNet:
    def __init__(self, path, agent):
        from vp.policy_io import load_net
        self.learner = load_net(path)
        self.agent = agent

    def act(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            return self.learner.act(self.agent, obs)
        import torch
        with torch.no_grad():
            return np.clip(self.learner.actors[self.agent](torch.as_tensor(obs)).numpy(), -1., 1.).astype(np.float32)


def make_opp(kind, seed, model_path=None):
    if kind == "random":
        return RandEvader(seed)
    if not model_path:
        raise ValueError(f"--opponent-model is required for {kind}")
    if kind == "ppo":
        return PPOOpp(model_path)
    if kind == "maddpg":
        return SavedNet(model_path, "evader")
    raise ValueError(f"unknown opponent: {kind}")


class OppPool:
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
