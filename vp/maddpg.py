import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vp.parallel import AGENTS
from vp.paths import CONFIGS

DEFAULT_CONFIG = CONFIGS / "maddpg_config.json"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"hidden_sizes", "actor_lr", "critic_lr", "batch_size", "replay_capacity",
                "gamma", "tau", "gradient_clip", "warmup_steps", "learn_every",
                "updates_per_step", "noise_start", "noise_end", "noise_decay_steps",
                "checkpoint_every", "torch_threads", "total_steps", "seed", "environment"}
    optional = {"action_reg", "alternate_every", "alternate_pattern", "mix_ppo_evader", "mix_ppo_model", "mix_frozen_seeker",
                                                  "mix_pool_seeker", "mix_pool_evader", "pool_every", "pool_size",
                                                  "actor_lr_seeker", "actor_lr_evader"}
    if not required <= set(config) <= required | optional:
        raise ValueError(f"config keys differ: missing={required - set(config)}, extra={set(config) - required}")
    positive_ints = ["batch_size", "replay_capacity", "learn_every", "updates_per_step",
                     "noise_decay_steps", "checkpoint_every", "torch_threads", "total_steps"]
    for key in positive_ints:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["warmup_steps", "seed"]:
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if config["batch_size"] > config["replay_capacity"]:
        raise ValueError("batch_size cannot exceed replay_capacity")
    if not config["hidden_sizes"] or any(type(n) is not int or n <= 0 for n in config["hidden_sizes"]):
        raise ValueError("hidden_sizes must contain positive integers")
    for key in ["actor_lr", "critic_lr", "gradient_clip"]:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= config["gamma"] <= 1 or not 0 < config["tau"] <= 1:
        raise ValueError("gamma must be in [0,1] and tau in (0,1]")
    if not (np.isfinite(config["noise_start"]) and np.isfinite(config["noise_end"])
            and 0 <= config["noise_end"] <= config["noise_start"]):
        raise ValueError("noise requires 0 <= noise_end <= noise_start")
    reg = config.get("action_reg", 0.0)
    if not isinstance(reg, (int, float)) or isinstance(reg, bool) or not np.isfinite(reg) or reg < 0:
        raise ValueError("action_reg must be a finite non-negative number")
    block = config.get("alternate_every", 0)
    if type(block) is not int or block < 0:
        raise ValueError("alternate_every must be a non-negative integer")
    for key in ("mix_ppo_evader", "mix_frozen_seeker", "mix_pool_seeker", "mix_pool_evader"):
        value = config.get(key, 0.0)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a probability in [0, 1]")
    for key in ("pool_every", "pool_size"):
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError(f"{key} must be a positive integer")
    if sum(config.get(k, 0.0) for k in ("mix_ppo_evader", "mix_frozen_seeker", "mix_pool_seeker", "mix_pool_evader")) > 1:
        raise ValueError("mix probabilities cannot sum above 1")
    if config.get("mix_ppo_evader", 0.0) > 0 and not config.get("mix_ppo_model"):
        raise ValueError("mix_ppo_evader requires mix_ppo_model")
    pattern = config.get("alternate_pattern")
    if pattern is not None and (not isinstance(pattern, list) or not pattern or any(a not in AGENTS for a in pattern)):
        raise ValueError("alternate_pattern must be a non-empty list of agent names")
    env = config["environment"]
    env_keys = {"episode_steps", "randomize_start", "yaw_rate_deg", "pitch_rate_deg", "pitch_limit_deg"}
    if not isinstance(env, dict) or not env_keys <= set(env) <= env_keys | {"boundary_penalty_per_step", "seeker_boundary_penalty_per_step", "obstacles", "seeker_history", "seeker_history_stride", "include_last_known_velocity", "seeker_visible_bonus", "evader_hidden_bonus", "seeker_proximity_coef"}:
        raise ValueError(f"environment must contain exactly {sorted(env_keys)}")
    if type(env["episode_steps"]) is not int or env["episode_steps"] <= 0:
        raise ValueError("environment episode_steps must be a positive integer")
    if type(env["randomize_start"]) is not bool:
        raise ValueError("environment randomize_start must be boolean")
    for key in ["yaw_rate_deg", "pitch_rate_deg", "pitch_limit_deg"]:
        if not isinstance(env[key], (int, float)) or not np.isfinite(env[key]) or env[key] <= 0:
            raise ValueError(f"environment {key} must be finite and positive")
    if env["pitch_limit_deg"] > 90:
        raise ValueError("environment pitch_limit_deg cannot exceed 90")
    return config


def mlp(input_dim, output_dim, hidden_sizes, output_activation=None):
    layers = []
    for width in hidden_sizes:
        layers += [nn.Linear(input_dim, width), nn.ReLU()]
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden_sizes, nn.Tanh)

    def forward(self, observation):
        return self.net(observation)


class Critic(nn.Module):
    def __init__(self, state_dim, joint_action_dim, hidden_sizes):
        super().__init__()
        self.net = mlp(state_dim + joint_action_dim, 1, hidden_sizes)

    def forward(self, state, joint_actions):
        return self.net(torch.cat([state, joint_actions], dim=-1))


class MemBuf:
    def __init__(self, capacity, obs_dims, action_dims, state_dim, seed=0):
        self.capacity = capacity
        self.obs_dims, self.action_dims = dict(obs_dims), dict(action_dims)
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self.position = self.size = 0
        self.observations = {a: np.empty((capacity, obs_dims[a]), np.float32) for a in AGENTS}
        self.next_observations = {a: np.empty_like(self.observations[a]) for a in AGENTS}
        self.actions = {a: np.empty((capacity, action_dims[a]), np.float32) for a in AGENTS}
        self.states = np.empty((capacity, state_dim), np.float32)
        self.next_states = np.empty_like(self.states)
        self.rewards = np.empty((capacity, 2), np.float32)
        self.terminated = np.empty((capacity, 2), np.float32)
        self.truncated = np.empty((capacity, 2), np.float32)
        self.frozen = np.zeros((capacity, 2), np.float32)
        self.frozen_id = np.zeros((capacity, 2), np.int64)

    def __len__(self):
        return self.size

    @staticmethod
    def _check(value, shape, label):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} must have shape {shape} and contain finite values")
        return value

    def add(self, observations, actions, rewards, next_observations, state, next_state,
            terminated, truncated, frozen=(0.0, 0.0), frozen_id=(0, 0)):
        groups = [observations, actions, rewards, next_observations, terminated, truncated]
        if any(set(group) != set(AGENTS) for group in groups):
            raise ValueError("replay transitions must contain both agents")
        obs = {a: self._check(observations[a], (self.obs_dims[a],), a + " observation") for a in AGENTS}
        nxt = {a: self._check(next_observations[a], (self.obs_dims[a],), a + " next observation") for a in AGENTS}
        act = {a: self._check(actions[a], (self.action_dims[a],), a + " action") for a in AGENTS}
        if any(np.any(np.abs(act[a]) > 1) for a in AGENTS):
            raise ValueError("replay actions must be normalized to [-1,1]")
        state = self._check(state, (self.state_dim,), "state")
        next_state = self._check(next_state, (self.state_dim,), "next state")
        reward = self._check([rewards[a] for a in AGENTS], (2,), "rewards")
        term = self._check([terminated[a] for a in AGENTS], (2,), "terminations")
        trunc = self._check([truncated[a] for a in AGENTS], (2,), "truncations")
        if np.any((term != 0) & (term != 1)) or np.any((trunc != 0) & (trunc != 1)):
            raise ValueError("episode flags must be boolean")
        i = self.position
        for a in AGENTS:
            self.observations[a][i], self.next_observations[a][i] = obs[a], nxt[a]
            self.actions[a][i] = act[a]
        self.states[i], self.next_states[i] = state, next_state
        self.rewards[i], self.terminated[i], self.truncated[i] = reward, term, trunc
        self.frozen[i] = self._check(frozen, (2,), "frozen flags")
        self.frozen_id[i] = np.asarray(frozen_id, dtype=np.int64)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if self.size < batch_size:
            raise ValueError("not enough joint transitions for a batch")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        tensor = lambda data: torch.as_tensor(data[indices], device=device)
        return {
            "observations": {a: tensor(self.observations[a]) for a in AGENTS},
            "next_observations": {a: tensor(self.next_observations[a]) for a in AGENTS},
            "actions": {a: tensor(self.actions[a]) for a in AGENTS},
            "state": tensor(self.states), "next_state": tensor(self.next_states),
            "rewards": tensor(self.rewards), "terminated": tensor(self.terminated),
            "truncated": tensor(self.truncated), "frozen": tensor(self.frozen), "frozen_id": torch.as_tensor(self.frozen_id[indices], device=device),
        }


def q_backup(reward, terminated, next_q, gamma):
    return reward + gamma * (1.0 - terminated) * next_q


class MADDPG:
    def __init__(self, obs_dims, action_dims, state_dim, config, device="cpu"):
        self.obs_dims, self.action_dims = dict(obs_dims), dict(action_dims)
        self.state_dim, self.config = state_dim, copy.deepcopy(config)
        if set(obs_dims) != set(AGENTS) or set(action_dims) != set(AGENTS):
            raise ValueError("MADDPG requires seeker and evader spaces")
        self.device = torch.device(device)
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(config["seed"])
        self.rng = np.random.default_rng(config["seed"])
        self.actors = {a: Actor(obs_dims[a], action_dims[a], config["hidden_sizes"]).to(self.device)
                       for a in AGENTS}
        self.critics = {a: Critic(state_dim, sum(action_dims.values()), config["hidden_sizes"]).to(self.device)
                        for a in AGENTS}
        self.target_actors = copy.deepcopy(self.actors)
        self.target_critics = copy.deepcopy(self.critics)
        for net in list(self.target_actors.values()) + list(self.target_critics.values()):
            net.requires_grad_(False)
        actor_lrs = {"seeker": config.get("actor_lr_seeker", config["actor_lr"]),
                     "evader": config.get("actor_lr_evader", config["actor_lr"])}
        self.actor_optimizers = {a: torch.optim.Adam(self.actors[a].parameters(), lr=actor_lrs[a])
                                 for a in AGENTS}
        self.critic_optimizers = {a: torch.optim.Adam(self.critics[a].parameters(), lr=config["critic_lr"])
                                  for a in AGENTS}
        self.update_count = 0
        self.training_step = 0

    def act(self, agent, observation, noise_std=0.0):
        obs = MemBuf._check(observation, (self.obs_dims[agent],), agent + " observation")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")
        with torch.no_grad():
            action = self.actors[agent](torch.as_tensor(obs, device=self.device)).cpu().numpy()
        if noise_std:
            action = action + self.rng.normal(0., noise_std, action.shape)
        action = np.clip(action, -1., 1.).astype(np.float32)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError(f"{agent} actor produced a non-finite action")
        return action

    @torch.no_grad()
    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.lerp_(source_param, self.config["tau"])

    def update(self, replay, active_agents=AGENTS, frozen_next_actions=None, flagged_only=False):
        if not active_agents or any(a not in AGENTS for a in active_agents):
            raise ValueError("active_agents must name at least one valid learner")
        batch = replay.sample(self.config["batch_size"], self.device)
        with torch.no_grad():
            next_actions = {}
            override = {}
            for agent in AGENTS:
                next_actions[agent] = self.target_actors[agent](batch["next_observations"][agent])
                if not (frozen_next_actions and agent in frozen_next_actions):
                    continue
                if flagged_only:
                    rows = batch["frozen"][:, AGENTS.index(agent)] > 0.5
                    if not rows.any():
                        continue
                    next_obs = batch["next_observations"][agent][rows].cpu().numpy()
                    provider = frozen_next_actions[agent]
                    if getattr(provider, "uses_ids", False):
                        ids = batch["frozen_id"][rows, AGENTS.index(agent)].cpu().numpy()
                        prediction = np.asarray(provider.batch(next_obs, ids), dtype=np.float32)
                    else:
                        prediction = np.asarray(frozen_next_actions[agent](next_obs), dtype=np.float32)
                    if prediction.shape != (int(rows.sum()), self.action_dims[agent]):
                        raise ValueError(agent + " frozen next actions have the wrong shape")
                    replaced = next_actions[agent].clone()
                    known = torch.as_tensor(np.all(np.isfinite(prediction), axis=1), device=self.device)
                    target_rows = torch.nonzero(rows).squeeze(-1)[known]
                    replaced[target_rows] = torch.as_tensor(prediction[known.cpu().numpy()], device=self.device)
                else:
                    next_obs = batch["next_observations"][agent].cpu().numpy()
                    prediction = MemBuf._check(
                        frozen_next_actions[agent](next_obs),
                        (self.config["batch_size"], self.action_dims[agent]), agent + " frozen next actions")
                    if np.any(np.abs(prediction) > 1):
                        raise ValueError("frozen opponent actions must be normalized")
                    replaced = torch.as_tensor(prediction, device=self.device)
                override[agent] = replaced

            def joint_next_for(learner):
                return torch.cat([override[a] if a in override and a != learner else next_actions[a]
                                  for a in AGENTS], dim=-1)
        joint_replay = torch.cat([batch["actions"][a] for a in AGENTS], dim=-1)
        metrics = {}
        for agent in active_agents:
            index = AGENTS.index(agent)
            with torch.no_grad():
                next_q = self.target_critics[agent](batch["next_state"], joint_next_for(agent))
                target = q_backup(batch["rewards"][:, index:index+1],
                                        batch["terminated"][:, index:index+1],
                                        next_q, self.config["gamma"])
            prediction = self.critics[agent](batch["state"], joint_replay)
            critic_loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError(f"{agent} critic loss is non-finite")
            self.critic_optimizers[agent].zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[agent].parameters(),
                                     self.config["gradient_clip"], error_if_nonfinite=True)
            self.critic_optimizers[agent].step()

            actor_actions = {a: batch["actions"][a].detach() for a in AGENTS}
            pre_tanh = self.actors[agent].net[:-1](batch["observations"][agent])
            actor_actions[agent] = torch.tanh(pre_tanh)
            self.critics[agent].requires_grad_(False)
            try:
                q_value = self.critics[agent](
                    batch["state"], torch.cat([actor_actions[a] for a in AGENTS], dim=-1))
                actor_loss = -q_value.mean()
                actor_loss = actor_loss + self.config.get("action_reg", 0.0) * (pre_tanh ** 2).mean()
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError(f"{agent} actor loss is non-finite")
                self.actor_optimizers[agent].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[agent].parameters(),
                                         self.config["gradient_clip"], error_if_nonfinite=True)
                self.actor_optimizers[agent].step()
            finally:
                self.critics[agent].requires_grad_(True)
            self._soft_update(self.target_actors[agent], self.actors[agent])
            self._soft_update(self.target_critics[agent], self.critics[agent])
            metrics[agent + "_actor_loss"] = float(actor_loss.detach().cpu())
            metrics[agent + "_critic_loss"] = float(critic_loss.detach().cpu())
        self.update_count += 1
        return metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1, "obs_dims": self.obs_dims, "action_dims": self.action_dims,
            "state_dim": self.state_dim, "config": self.config,
            "training_step": self.training_step, "update_count": self.update_count,
            "noise_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }
        for name in ["actors", "critics", "target_actors", "target_critics",
                     "actor_optimizers", "critic_optimizers"]:
            checkpoint[name] = {a: obj.state_dict() for a, obj in getattr(self, name).items()}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        if path.suffix != ".pt":
            raise ValueError("expected a MADDPG .pt checkpoint; PPO .zip weights are incompatible")
        if not path.is_file():
            raise FileNotFoundError(f"MADDPG checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported MADDPG checkpoint format")
        learner = cls(checkpoint["obs_dims"], checkpoint["action_dims"],
                      checkpoint["state_dim"], checkpoint["config"], device)
        for name in ["actors", "critics", "target_actors", "target_critics",
                     "actor_optimizers", "critic_optimizers"]:
            for agent in AGENTS:
                getattr(learner, name)[agent].load_state_dict(checkpoint[name][agent])
        learner.training_step = checkpoint["training_step"]
        learner.update_count = checkpoint["update_count"]
        learner.rng.bit_generator.state = checkpoint["noise_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        return learner
