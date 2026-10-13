"""MAPPO: per-agent Gaussian actors with centralised value critics, trained with clipped PPO.

On-policy and model-free (policy-gradient). Used as a comparison to MADDPG on the same environment,
observations, opponents and evaluation. Each agent has its own actor (local observation) and its own
critic (global state, training only), because the two agents' rewards differ.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from vanishing_point_parallel import AGENTS
from maddpg import mlp

DEFAULT_MAPPO_CONFIG = Path(__file__).with_name("mappo_config.json")
REQUIRED = {"hidden_sizes", "actor_lr", "critic_lr", "rollout_steps", "epochs", "minibatch",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "max_grad_norm", "init_log_std",
            "reward_scale", "total_steps", "seed", "torch_threads", "checkpoint_every", "environment"}
OPTIONAL = {"alternate_every", "alternate_pattern", "mix_ppo_evader", "mix_ppo_model", "mix_frozen_seeker"}


def load_mappo_config(path=DEFAULT_MAPPO_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not REQUIRED <= set(config) <= REQUIRED | OPTIONAL:
        raise ValueError(f"mappo config keys differ: missing={REQUIRED - set(config)}, "
                         f"extra={set(config) - REQUIRED - OPTIONAL}")
    for key in ("rollout_steps", "epochs", "minibatch", "total_steps", "checkpoint_every", "torch_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["minibatch"] > config["rollout_steps"]:
        raise ValueError("minibatch cannot exceed rollout_steps")
    for key in ("actor_lr", "critic_lr", "clip_range", "max_grad_norm", "reward_scale"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= config["gamma"] <= 1 or not 0 <= config["gae_lambda"] <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    for key in ("mix_ppo_evader", "mix_frozen_seeker"):
        value = config.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a probability in [0, 1]")
    if config.get("mix_ppo_evader", 0.0) + config.get("mix_frozen_seeker", 0.0) > 1:
        raise ValueError("mix probabilities cannot sum above 1")
    if config.get("mix_ppo_evader", 0.0) > 0 and not config.get("mix_ppo_model"):
        raise ValueError("mix_ppo_evader requires mix_ppo_model")
    return config


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, init_log_std):
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

    def forward(self, observation):
        return self.net(observation)  # mean action (unclipped)

    def distribution(self, observation):
        return Normal(self.net(observation), self.log_std.exp())


class MAPPO:
    algo = "mappo"

    def __init__(self, obs_dims, action_dims, state_dim, config, device="cpu"):
        if set(obs_dims) != set(AGENTS) or set(action_dims) != set(AGENTS):
            raise ValueError("MAPPO requires seeker and evader spaces")
        self.obs_dims, self.action_dims = dict(obs_dims), dict(action_dims)
        self.state_dim, self.config = state_dim, copy.deepcopy(config)
        self.device = torch.device(device)
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(config["seed"])
        self.actors = {a: GaussianActor(obs_dims[a], action_dims[a], config["hidden_sizes"],
                                        config["init_log_std"]).to(self.device) for a in AGENTS}
        self.critics = {a: mlp(state_dim, 1, config["hidden_sizes"]).to(self.device) for a in AGENTS}
        self.actor_optimizers = {a: torch.optim.Adam(self.actors[a].parameters(), lr=config["actor_lr"])
                                 for a in AGENTS}
        self.critic_optimizers = {a: torch.optim.Adam(self.critics[a].parameters(), lr=config["critic_lr"])
                                  for a in AGENTS}
        self.update_count = 0
        self.training_step = 0

    # ---- acting -------------------------------------------------------------------------------
    def act(self, agent, observation, noise_std=0.0):
        """Deterministic action (the Gaussian mean, clipped): used for evaluation and as a frozen opponent."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.obs_dims[agent],) or not np.all(np.isfinite(obs)):
            raise ValueError(f"{agent} observation must be a finite vector of {self.obs_dims[agent]} values")
        with torch.no_grad():
            action = self.actors[agent](torch.as_tensor(obs, device=self.device)).cpu().numpy()
        return np.clip(action, -1., 1.).astype(np.float32)

    @torch.no_grad()
    def sample(self, agent, observation):
        """Stochastic action for training: returns (raw action, log-probability)."""
        dist = self.actors[agent].distribution(torch.as_tensor(observation, dtype=torch.float32, device=self.device))
        raw = dist.sample()
        return raw.cpu().numpy(), float(dist.log_prob(raw).sum())

    @torch.no_grad()
    def value(self, agent, state):
        return float(self.critics[agent](torch.as_tensor(state, dtype=torch.float32, device=self.device)))

    # ---- learning -----------------------------------------------------------------------------
    def update(self, agent, data, rng):
        """Clipped-PPO epochs on one agent's rollout. `data['valid']` masks steps the agent did not act in."""
        cfg = self.config
        tensors = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}
        size = len(tensors["obs"])
        totals = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
        batches = 0
        for _ in range(cfg["epochs"]):
            order = rng.permutation(size)
            for start in range(0, size - cfg["minibatch"] + 1, cfg["minibatch"]):
                idx = torch.as_tensor(order[start:start + cfg["minibatch"]], device=self.device)
                weight = tensors["valid"][idx]
                denom = weight.sum().clamp(min=1.0)
                dist = self.actors[agent].distribution(tensors["obs"][idx])
                logp = dist.log_prob(tensors["act"][idx]).sum(-1)
                ratio = (logp - tensors["logp"][idx]).exp()
                adv = tensors["adv"][idx]
                surrogate = torch.min(ratio * adv, ratio.clamp(1 - cfg["clip_range"], 1 + cfg["clip_range"]) * adv)
                entropy = (dist.entropy().sum(-1) * weight).sum() / denom
                actor_loss = -(surrogate * weight).sum() / denom - cfg["ent_coef"] * entropy
                self.actor_optimizers[agent].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[agent].parameters(), cfg["max_grad_norm"])
                self.actor_optimizers[agent].step()
                value = self.critics[agent](tensors["state"][idx]).squeeze(-1)
                critic_loss = 0.5 * (((value - tensors["ret"][idx]) ** 2) * weight).sum() / denom
                self.critic_optimizers[agent].zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critics[agent].parameters(), cfg["max_grad_norm"])
                self.critic_optimizers[agent].step()
                totals["actor"] += float(actor_loss.detach())
                totals["critic"] += float(critic_loss.detach())
                totals["entropy"] += float(entropy.detach())
                batches += 1
        self.update_count += 1
        return {k: v / max(batches, 1) for k, v in totals.items()}

    # ---- persistence --------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"algo": "mappo", "format_version": 1, "obs_dims": self.obs_dims,
                      "action_dims": self.action_dims, "state_dim": self.state_dim, "config": self.config,
                      "training_step": self.training_step, "update_count": self.update_count}
        for name in ("actors", "critics", "actor_optimizers", "critic_optimizers"):
            checkpoint[name] = {a: obj.state_dict() for a, obj in getattr(self, name).items()}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        if path.suffix != ".pt" or not path.is_file():
            raise FileNotFoundError(f"MAPPO checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("algo") != "mappo":
            raise ValueError("not a MAPPO checkpoint")
        learner = cls(checkpoint["obs_dims"], checkpoint["action_dims"], checkpoint["state_dim"],
                      checkpoint["config"], device)
        for name in ("actors", "critics", "actor_optimizers", "critic_optimizers"):
            for agent in AGENTS:
                getattr(learner, name)[agent].load_state_dict(checkpoint[name][agent])
        learner.training_step = checkpoint["training_step"]
        learner.update_count = checkpoint["update_count"]
        return learner


def gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised advantage estimation. `dones[t]`=1 cuts the bootstrap after step t."""
    advantages = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(rewards) else last_value
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        advantages[t] = running
    return advantages, advantages + np.asarray(values, dtype=np.float64)
