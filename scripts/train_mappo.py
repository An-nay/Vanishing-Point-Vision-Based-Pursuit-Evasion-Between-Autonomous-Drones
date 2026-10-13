"""MAPPO training: Phase B (Seeker vs frozen PPO Evader) and joint training with opponent mixing.

    python -m scripts.train_mappo --phase seeker --opponent-model models/ppo_city_plain_s2001.zip --output runs/mappo_B --seeds 2001
    python -m scripts.train_mappo --phase simultaneous --initialize runs/mappo_B/seeker_s2001/final.pt --output runs/mappo_C --seeds 2001

Same environment, observations, opponent mixing, alternating blocks and evaluation as the MADDPG recipe.
Steps where an agent's action came from a frozen opponent are masked out of that agent's PPO loss
(PPO is on-policy, so it may only learn from its own actions).
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from vp.parallel import VanishingPointParallelEnv, AGENTS
from vp.mappo import MAPPO, load_mappo_config, gae, DEFAULT_MAPPO_CONFIG
from vp.opponents import make_evader, CheckpointOpponent
from scripts.train_maddpg import EPISODE_FIELDS, UPDATE_FIELDS, episode_seed, choose_mode
from vp.paths import RUNS


def train(config, phase, output, opponent_model=None, initialize=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    env = VanishingPointParallelEnv(**config["environment"])
    obs_dims = {a: env.observation_space(a).shape[0] for a in AGENTS}
    action_dims = {a: env.action_space(a).shape[0] for a in AGENTS}
    learner = MAPPO(obs_dims, action_dims, env.state_space.shape[0], config)
    if initialize:  # warm start: the pretrained Seeker (actor and critic); the Evader starts fresh
        prior = MAPPO.load(initialize)
        if prior.obs_dims != obs_dims or prior.state_dim != learner.state_dim:
            raise ValueError("initial checkpoint spaces do not match this environment")
        learner.actors["seeker"].load_state_dict(prior.actors["seeker"].state_dict())
        learner.critics["seeker"].load_state_dict(prior.critics["seeker"].state_dict())
    ppo_opponent = None
    if phase == "seeker" or config.get("mix_ppo_evader", 0.0) > 0:
        ppo_opponent = make_evader("ppo", config["seed"], opponent_model or config.get("mix_ppo_model"))
    seeker_opponent = None
    if phase == "simultaneous" and config.get("mix_frozen_seeker", 0.0) > 0:
        if not initialize:
            raise ValueError("mix_frozen_seeker needs --initialize: the pretrained Seeker is the frozen opponent")
        seeker_opponent = CheckpointOpponent(initialize, "seeker")
    active_all = ("seeker",) if phase == "seeker" else AGENTS
    (output / "run.json").write_text(json.dumps(
        {"algo": "mappo", "phase": phase, "config": config, "initialize": str(initialize) if initialize else None},
        indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(config["seed"])
    gamma, lam, scale = config["gamma"], config["gae_lambda"], config["reward_scale"]
    T = config["rollout_steps"]
    episode, step_total = 0, 0
    obs, _ = env.reset(seed=episode_seed(config["seed"], 0))
    mode = "ppo_evader" if phase == "seeker" else choose_mode(config, rng, phase)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file, \
         (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_file:
        episode_log = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
        update_log = csv.DictWriter(update_file, fieldnames=UPDATE_FIELDS)
        episode_log.writeheader()
        update_log.writeheader()
        next_checkpoint = config["checkpoint_every"]
        while step_total < config["total_steps"]:
            store = {a: {k: [] for k in ("obs", "state", "act", "logp", "rew", "val", "done", "valid")}
                     for a in AGENTS}
            for _ in range(T):
                state = env.state().copy()
                actions = {}
                for agent in AGENTS:
                    frozen = (agent == "evader" and mode == "ppo_evader") or \
                             (agent == "seeker" and mode == "frozen_seeker")
                    if agent == "evader" and mode == "ppo_evader":
                        actions[agent], raw, logp = ppo_opponent.act(obs[agent]), np.zeros(action_dims[agent]), 0.0
                    elif agent == "seeker" and mode == "frozen_seeker":
                        actions[agent], raw, logp = seeker_opponent.act(obs[agent]), np.zeros(action_dims[agent]), 0.0
                    else:
                        raw, logp = learner.sample(agent, obs[agent])
                        actions[agent] = np.clip(raw, -1., 1.).astype(np.float32)
                    buf = store[agent]
                    buf["obs"].append(obs[agent]); buf["state"].append(state); buf["act"].append(raw)
                    buf["logp"].append(logp); buf["val"].append(learner.value(agent, state))
                    buf["valid"].append(0.0 if frozen else 1.0)
                obs, rewards, terms, truncs, _ = env.step(actions)
                done = not env.agents  # time limit: bootstrap from the final state, never from the reset state
                final_state = env.state().copy()
                for agent in AGENTS:
                    reward = rewards[agent] * scale
                    if done:
                        reward += gamma * learner.value(agent, final_state)
                    store[agent]["rew"].append(reward)
                    store[agent]["done"].append(1.0 if done else 0.0)
                step_total += 1
                learner.training_step = step_total
                if done:
                    summary = env.sim.episode_summary()
                    episode_log.writerow({"episode": episode, "step": step_total, "partial": False,
                                          "noise_std": 0.0, "mode": mode, **summary})
                    episode_file.flush()
                    print(f"step={step_total} episode={episode} mode={mode} visible={summary['visible_pct']:.1f}% "
                          f"returns=({summary['seeker_reward']:.1f}, {summary['evader_reward']:.1f})", flush=True)
                    episode += 1
                    obs, _ = env.reset(seed=episode_seed(config["seed"], episode))
                    mode = "ppo_evader" if phase == "seeker" else choose_mode(config, rng, phase)
            # ---- learn from this rollout ----
            block = config.get("alternate_every", 0)
            if block and phase == "simultaneous":
                pattern = config.get("alternate_pattern") or list(AGENTS)
                learners = (pattern[(step_total // block) % len(pattern)],)
            else:
                learners = active_all
            last_state = env.state().copy()
            row = {"step": step_total, "update": learner.update_count + 1}
            for agent in learners:
                buf = store[agent]
                values = np.asarray(buf["val"])
                advantages, returns = gae(np.asarray(buf["rew"]), values, np.asarray(buf["done"]),
                                          learner.value(agent, last_state), gamma, lam)
                valid = np.asarray(buf["valid"])
                if valid.sum() < 2:
                    continue
                picked = advantages[valid > 0]
                advantages = (advantages - picked.mean()) / (picked.std() + 1e-8)
                metrics = learner.update(agent, {
                    "obs": np.asarray(buf["obs"]), "state": np.asarray(buf["state"]),
                    "act": np.asarray(buf["act"]), "logp": np.asarray(buf["logp"]),
                    "adv": advantages, "ret": returns, "valid": valid}, rng)
                row[agent + "_actor_loss"], row[agent + "_critic_loss"] = metrics["actor"], metrics["critic"]
            update_log.writerow(row)
            update_file.flush()
            if step_total >= next_checkpoint:
                learner.save(output / f"checkpoint_{step_total:08d}.pt")
                next_checkpoint += config["checkpoint_every"]
    learner.save(output / "final.pt")
    print(f"Saved {output / 'final.pt'} ({learner.update_count} updates)", flush=True)
    return output / "final.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["seeker", "simultaneous"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_MAPPO_CONFIG)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, default=RUNS / "mappo")
    parser.add_argument("--opponent-model", type=Path, help="frozen PPO Evader (phase seeker)")
    parser.add_argument("--initialize", type=Path, help="pretrained MAPPO Seeker to warm-start from")
    args = parser.parse_args()
    try:
        config = load_mappo_config(args.config)
        if args.total_steps:
            config["total_steps"] = args.total_steps
        for seed in args.seeds or [config["seed"]]:
            config["seed"] = seed
            train(dict(config), args.phase, args.output / f"{args.phase}_s{seed}", args.opponent_model,
                  args.initialize)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
