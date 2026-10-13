"""Phase A validation, Phase B frozen-opponent training, Phase C joint learning."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from vanishing_point_parallel import VanishingPointParallelEnv, AGENTS
from vanishing_point_core import OBSTACLE_PRESETS
from maddpg import MADDPG, ReplayBuffer, load_config, DEFAULT_CONFIG
from stage2_opponents import make_evader, CheckpointOpponent, OpponentProvider

EPISODE_FIELDS = [
    "episode", "step", "partial", "noise_std", "steps", "seeker_reward", "evader_reward",
    "undetected_pct", "visible_pct", "initially_visible", "first_loss_time_s",
    "longest_hidden_s", "seeker_wall_contacts", "evader_wall_contacts",
    "seeker_boundary_steps", "evader_boundary_steps", "loss_events", "reacquisitions",
    "reacquisition_mean_s", "unresolved_hidden_s", "first_detection_time_s",
    "drone_collision_steps", "drone_collision_events", "mode",
]
UPDATE_FIELDS = ["step", "update", "seeker_actor_loss", "seeker_critic_loss",
                 "evader_actor_loss", "evader_critic_loss"]


def episode_seed(config_seed, episode):
    """Episode world seeds never collide across training seeds (each seed owns a 1M block)."""
    return config_seed * 1_000_003 + episode


def noise_at(step, config):
    fraction = min(step / config["noise_decay_steps"], 1.)
    return config["noise_start"] + fraction * (config["noise_end"] - config["noise_start"])


def validate_environment(config, seed, episodes=2, render=False):
    from pettingzoo.test import parallel_api_test
    env = VanishingPointParallelEnv(**config["environment"])
    try:
        parallel_api_test(env, num_cycles=config["environment"]["episode_steps"] + 5)
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            while env.agents:
                env.step({a: env.action_space(a).sample() for a in env.agents})
            print(json.dumps(env.sim.episode_summary()))
    finally:
        env.close()
    if render:
        demo = VanishingPointParallelEnv(render_mode="human", **config["environment"])
        try:
            demo.reset(seed=seed)
            while demo.agents:
                demo.step({a: demo.action_space(a).sample() for a in demo.agents})
        finally:
            demo.close()


def choose_mode(config, rng, phase):
    """Per-episode opponent: self-play, a frozen PPO Evader, or a frozen learned Seeker (simultaneous phase only)."""
    if phase != "simultaneous":
        return "selfplay"
    draw = rng.random()
    ppo = config.get("mix_ppo_evader", 0.0)
    if draw < ppo:
        return "ppo_evader"
    edge = ppo + config.get("mix_frozen_seeker", 0.0)
    if draw < edge:
        return "frozen_seeker"
    if draw < edge + config.get("mix_pool_seeker", 0.0):
        return "pool_seeker"  # a past Seeker snapshot plays against the learning Evader
    if draw < edge + config.get("mix_pool_seeker", 0.0) + config.get("mix_pool_evader", 0.0):
        return "pool_evader"  # a past Evader snapshot plays against the learning Seeker
    return "selfplay"


def train(config, phase, output, opponent_kind="random", opponent_model=None,
          initialize=None, render=False):
    """Returns the saved final checkpoint. Replay is fresh for each training run."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    env = VanishingPointParallelEnv(render_mode="human" if render else None,
                                   **config["environment"])
    obs_dims = {a: env.observation_space(a).shape[0] for a in AGENTS}
    action_dims = {a: env.action_space(a).shape[0] for a in AGENTS}
    opponent = make_evader(opponent_kind, config["seed"], opponent_model) if phase == "seeker" else None
    learner = MADDPG(obs_dims, action_dims, env.state_space.shape[0], config)
    if initialize:
        prior = MADDPG.load(initialize)
        if prior.obs_dims != obs_dims or prior.action_dims != action_dims or prior.state_dim != learner.state_dim:
            raise ValueError("initial checkpoint spaces do not match this environment")
        for name in ["actors", "critics", "target_actors", "target_critics"]:
            for agent in AGENTS:
                try:
                    getattr(learner, name)[agent].load_state_dict(getattr(prior, name)[agent].state_dict())
                except RuntimeError as exc:
                    raise ValueError("initial checkpoint network sizes do not match config") from exc
    replay = ReplayBuffer(config["replay_capacity"], obs_dims, action_dims,
                          learner.state_dim, seed=config["seed"])
    active = ("seeker",) if phase == "seeker" else AGENTS
    frozen = {"evader": opponent.act} if opponent else None
    run_metadata = {"phase": phase, "trained_agents": list(active), "config": config,
                    "opponent": opponent_kind if opponent else None,
                    "opponent_model": str(opponent_model) if opponent_model else None,
                    "initialize": str(initialize) if initialize else None}
    (output / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(config["seed"])
    mix_opponent = (make_evader("ppo", config["seed"], config["mix_ppo_model"])
                    if config.get("mix_ppo_evader", 0.0) > 0 else None)
    seeker_opponent = None
    if config.get("mix_frozen_seeker", 0.0) > 0:
        if not initialize:
            raise ValueError("mix_frozen_seeker needs --initialize: the pretrained Seeker is the frozen opponent")
        seeker_opponent = CheckpointOpponent(initialize, "seeker")
    from maddpg import Actor
    pool_on = phase == "simultaneous" and (config.get("mix_pool_seeker", 0.0) > 0 or config.get("mix_pool_evader", 0.0) > 0)
    providers = {a: OpponentProvider(a, action_dims[a], fixed=fixed, capacity=config.get("pool_size", 20),
                                     build_actor=lambda a=a: Actor(obs_dims[a], action_dims[a], config["hidden_sizes"]))
                 for a, fixed in (("evader", mix_opponent), ("seeker", seeker_opponent))}
    if pool_on:  # the starting (warm-started) policies are the first snapshots
        for a in AGENTS:
            providers[a].add_snapshot(learner.actors[a].state_dict())

    def pick(requested):
        """Resolve the episode's opponent: (mode, frozen-policy id). Empty pools fall back to self-play."""
        if requested == "pool_seeker" or requested == "pool_evader":
            ids = providers["seeker" if requested == "pool_seeker" else "evader"].snapshot_ids()
            return (requested, int(rng.choice(ids))) if ids else ("selfplay", 0)
        return (requested, 0) if requested in ("ppo_evader", "frozen_seeker") else ("selfplay", 0)

    mode, opponent_id = pick(choose_mode(config, rng, phase))
    episode = 0
    observations, _ = env.reset(seed=episode_seed(config["seed"], 0))
    try:
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file, \
             (output / "updates.csv").open("w", newline="", encoding="utf-8") as update_file:
            episode_log = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
            update_log = csv.DictWriter(update_file, fieldnames=UPDATE_FIELDS)
            episode_log.writeheader()
            update_log.writeheader()
            for step in range(1, config["total_steps"] + 1):
                sigma = noise_at(step - 1, config)
                actions = {}
                for agent in AGENTS:
                    if agent == "evader" and opponent is not None:
                        actions[agent] = opponent.act(observations[agent])
                    elif agent == "evader" and mode in ("ppo_evader", "pool_evader"):
                        actions[agent] = providers["evader"].act(opponent_id, observations[agent])
                    elif agent == "seeker" and mode in ("frozen_seeker", "pool_seeker"):
                        actions[agent] = providers["seeker"].act(opponent_id, observations[agent])
                    elif step <= config["warmup_steps"]:
                        actions[agent] = rng.uniform(-1., 1., action_dims[agent]).astype(np.float32)
                    else:
                        actions[agent] = learner.act(agent, observations[agent], sigma)
                state = env.state().copy()
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                replay.add(observations, actions, rewards, next_obs, state,
                           env.state().copy(), terms, truncs,
                           frozen=(float(mode in ("frozen_seeker", "pool_seeker")),
                                   float(mode in ("ppo_evader", "pool_evader"))),
                           frozen_id=(opponent_id if mode in ("frozen_seeker", "pool_seeker") else 0,
                                      opponent_id if mode in ("ppo_evader", "pool_evader") else 0))
                if pool_on and step >= config["warmup_steps"] and step % config.get("pool_every", 30000) == 0:
                    for a in AGENTS:  # snapshot both learners into the opponent pool
                        providers[a].add_snapshot(learner.actors[a].state_dict())
                learner.training_step = step
                if (step >= config["warmup_steps"] and len(replay) >= config["batch_size"]
                        and step % config["learn_every"] == 0):
                    learners = active
                    block = config.get("alternate_every", 0)
                    if block and phase == "simultaneous":  # train one agent at a time (less non-stationary)
                        pattern = config.get("alternate_pattern") or list(AGENTS)
                        learners = (pattern[(step // block) % len(pattern)],)
                    for _ in range(config["updates_per_step"]):
                        if phase == "simultaneous" and (mix_opponent or seeker_opponent or pool_on):
                            mixed = {}
                            if mix_opponent or config.get("mix_pool_evader", 0.0) > 0:
                                mixed["evader"] = providers["evader"]
                            if seeker_opponent or config.get("mix_pool_seeker", 0.0) > 0:
                                mixed["seeker"] = providers["seeker"]
                            metrics = learner.update(replay, learners, mixed, flagged_only=True)
                        else:
                            metrics = learner.update(replay, learners, frozen)
                        update_log.writerow({"step": step, "update": learner.update_count, **metrics})
                observations = next_obs
                if not env.agents:
                    summary = env.sim.episode_summary()
                    episode_log.writerow({"episode": episode, "step": step, "partial": False,
                                          "noise_std": sigma, "mode": mode, **summary})
                    episode_file.flush()
                    update_file.flush()
                    print(f"step={step} episode={episode} mode={mode} "
                          f"visible={summary['visible_pct']:.1f}% "
                          f"returns=({summary['seeker_reward']:.1f}, {summary['evader_reward']:.1f}) "
                          f"updates={learner.update_count}", flush=True)
                    episode += 1
                    mode, opponent_id = pick(choose_mode(config, rng, phase))
                    if step < config["total_steps"]:
                        observations, _ = env.reset(seed=episode_seed(config["seed"], episode))
                if step % config["checkpoint_every"] == 0:
                    learner.save(output / f"checkpoint_{step:08d}.pt")
            if env.sim.step_count and env.agents:
                episode_log.writerow({"episode": episode, "step": config["total_steps"],
                                      "partial": True, "noise_std": sigma, "mode": mode,
                                      **env.sim.episode_summary()})
        learner.save(output / "final.pt")
        print(f"Saved {output / 'final.pt'} ({learner.update_count} updates)", flush=True)
        return output / "final.pt"
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validate", "seeker", "simultaneous"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--total-steps", type=int, help="Override config total_steps")
    parser.add_argument("--seeds", type=int, nargs="+", help="Train one run per seed")
    parser.add_argument("--episodes", type=int, default=2, help="Random validation episodes")
    parser.add_argument("--output", type=Path, default=Path("runs/stage2"))
    parser.add_argument("--opponent", choices=["random", "ppo", "maddpg"], default="random")
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument("--initialize", type=Path, help="Warm-start compatible MADDPG networks; fresh replay/optimizers")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--obstacles", choices=sorted(OBSTACLE_PRESETS),
                        help="Obstacle preset; overrides the config (changes observation size)")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.total_steps is not None:
            if args.total_steps <= 0:
                raise ValueError("--total-steps must be positive")
            config["total_steps"] = args.total_steps
        if args.obstacles:
            config["environment"]["obstacles"] = args.obstacles
        seeds = args.seeds or [config["seed"]]
        if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("--seeds must be distinct non-negative integers")
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")
        for seed in seeds:
            config["seed"] = seed
            if args.phase == "validate":
                validate_environment(config, seed, args.episodes, args.render)
            else:
                destination = args.output / f"{args.phase}_s{seed}"
                train(config, args.phase, destination, args.opponent,
                      args.opponent_model, args.initialize, args.render)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
