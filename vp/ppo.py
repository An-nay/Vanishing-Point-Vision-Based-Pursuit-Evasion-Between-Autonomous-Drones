import argparse
import math
import os
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


from vp.core import *
from vp.render import Drawer
from vp.paths import MODELS


class PPOEnv(gym.Env, Simulator, Drawer):

    metadata = {"render_modes": ["human"], "render_fps": int(1 / DT)}

    def __init__(self, render_mode=None, randomize_start=True, obstacles=None):
        gym.Env.__init__(self)
        Simulator.__init__(self, randomize_start=randomize_start, obstacles=obstacles)
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.obs_dim = 21 + int(SHOW_SEEN) + self.obs_extra
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.physics_client = self.seeker_id = self.evader_id = None
        self.seeker_policy = None
        if render_mode == "human":
            self._init_renderer()

    def set_params(self, seeker_speed=None, fov_deg=None, detection_range=None):
        if seeker_speed is not None:
            self.seeker_speed = float(seeker_speed)
        if fov_deg is not None:
            self.fov_deg = float(fov_deg)
        if detection_range is not None:
            self.detection_range = float(detection_range)

    def set_difficulty(self, level):
        level = float(np.clip(level, 0.0, 1.0))
        self.set_params(
            seeker_speed=EASY_SEEKER_SPEED + level * (SEEKER_SPEED - EASY_SEEKER_SPEED),
            fov_deg=EASY_FOV_DEG + level * (FOV_DEG - EASY_FOV_DEG),
            detection_range=EASY_DETECTION_RANGE
            + level * (DETECTION_RANGE - EASY_DETECTION_RANGE),
        )

    def difficulty_summary(self):
        return (
            f"Seeker speed {self.seeker_speed:.2f} m/s | "
            f"FOV {self.fov_deg:.0f} deg | "
            f"range {self.detection_range:.1f} m"
        )

    def _check_visibility(self):
        visible, distance, angle = can_see(
            self.seeker_pos,
            self.seeker_policy.forward,
            self.evader_pos,
            self.fov_deg,
            self.detection_range,
        )
        if visible and ray_hits(self.seeker_pos, self.evader_pos, self.obstacles):
            visible = False
        return visible, distance, angle


    def close(self):
        Drawer.close(self)

    def _sample_initial_state(self):
        return rand_start(self.np_random, self.randomize_start,
                                    self.fov_deg, self.detection_range)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.reset_world(self.np_random, (options or {}).get("initial_state"))
        self.seeker_policy = SeekerBot(
            self.np_random, self.initial_forward, speed=self.seeker_speed,
            obstacles=self.obstacles)
        if self.render_mode == "human":
            self._reset_render_objects()
            self._render_step(self.visible, self.distance, self.angle, reward=0.0)
        return self._get_observation(self.visible), self._get_info(
            self.visible, self.distance, self.angle, False)

    def step(self, action):
        action = np.clip(as_vec(np.asarray(action, dtype=np.float32), 3, "evader action"), -1.0, 1.0)
        visible_before, _, _ = self._check_visibility()
        seeker_velocity = self.seeker_policy.act(
            self.seeker_pos, self.evader_pos, visible_before)
        rewards, contacts, truncated = self.advance(
            seeker_velocity, cap_speed(action, EVADER_SPEED),
            seeker_forward=self.seeker_policy.forward)
        reward = rewards["evader"]
        if not self.visible and self.first_loss_step is None:
            self.first_loss_step = self.step_count
        info = self._get_info(self.visible, self.distance, self.angle, contacts["evader"])
        if truncated:
            info["episode_summary"] = {
                "total_reward": self.total_reward,
                "undetected_ratio": self.undetected_steps / self.step_count,
                "first_loss_time_s": None if self.first_loss_step is None else self.first_loss_step * DT,
                "longest_hidden_s": self.longest_hidden_streak * DT,
                "wall_touches": self.wall_touches,
                "seeker_search_restarts": self.seeker_policy.searches,
            }
        if self.render_mode == "human":
            self._render_step(self.visible, self.distance, self.angle, reward)
            time.sleep(DT)
        return self._get_observation(self.visible), reward, False, truncated, info

    def _get_observation(self, visible):
        self.camera_forward = self.seeker_policy.forward.copy()
        return self.evader_observation(SHOW_SEEN)

    def _get_info(self, visible, distance, angle, boundary_contact):
        return {
            "visible": bool(visible),
            "distance": float(distance),
            "fov_angle_deg": float(angle),
            "seeker_mode": self.seeker_policy.mode,
            "wall_contact": bool(boundary_contact),
            "wall_touches": int(self.wall_touches),
            "searches": int(self.seeker_policy.searches),
        }

def train_ppo(
    total_timesteps,
    seed,
    model_path,
    tensorboard_log=None,
    curriculum=False,
    stages=4,
    obstacles=None,
):
    print("\n=== PPO TRAINING ===")
    print(f"Seed: {seed}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Visibility in observation: {SHOW_SEEN}")
    if curriculum:
        print(f"Curriculum: ON ({stages} stages, easy to full difficulty)")
    else:
        print("Curriculum: OFF (full difficulty throughout)")

    raw_env = PPOEnv(render_mode=None, randomize_start=True, obstacles=obstacles)
    check_env(raw_env, warn=True)
    env = Monitor(raw_env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        device="auto",
    )

    if curriculum:
        levels = np.linspace(0.0, 1.0, stages)
        steps_per_stage = max(1, total_timesteps // stages)
        for i, level in enumerate(levels):
            raw_env.set_difficulty(float(level))
            print(
                f"\n--- Curriculum stage {i + 1}/{stages} "
                f"(difficulty {level:.2f}): {raw_env.difficulty_summary()} ---"
            )
            model.learn(
                total_timesteps=steps_per_stage,
                reset_num_timesteps=(i == 0),
                progress_bar=False,
            )
    else:
        model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    env.close()

    print(f"\nSaved PPO model to: {model_path}.zip")


def run_episode(env, action_fn, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        obs, _, terminated, truncated, info = env.step(action_fn(obs))
        done = terminated or truncated
    return info["episode_summary"]


def print_episode(index, s):
    first_loss = s["first_loss_time_s"]
    first_loss_text = "never" if first_loss is None else f"{first_loss:.1f}s"
    print(
        f"Episode {index + 1:02d}: "
        f"return={s['total_reward']:7.1f} | "
        f"undetected={100 * s['undetected_ratio']:5.1f}% | "
        f"first loss={first_loss_text:>6s} | "
        f"longest hidden={s['longest_hidden_s']:5.1f}s | "
        f"walls={s['wall_touches']} | "
        f"searches={s['seeker_search_restarts']}"
    )


def print_summary(summaries):
    returns = np.array([s["total_reward"] for s in summaries])
    ratios = np.array([s["undetected_ratio"] for s in summaries])
    losses = [
        s["first_loss_time_s"]
        for s in summaries
        if s["first_loss_time_s"] is not None
    ]
    hidden = np.array([s["longest_hidden_s"] for s in summaries])
    walls = np.array([s["wall_touches"] for s in summaries])
    searches = np.array([s["seeker_search_restarts"] for s in summaries])

    print(f"\n--- Summary over {len(summaries)} episodes ---")
    print(f"1. Mean return:                 {returns.mean():.1f} (std {returns.std():.1f})")
    print(f"2. Mean undetected time:        {100 * ratios.mean():.1f}%")
    if losses:
        print(
            f"3. Mean time until first loss:  {np.mean(losses):.1f} s "
            f"(sight lost in {len(losses)}/{len(summaries)} episodes)"
        )
    else:
        print("3. Mean time until first loss:  never lost sight in any episode")
    print(f"4. Mean longest hidden stretch: {hidden.mean():.1f} s")
    print(f"5. Mean wall touches:           {walls.mean():.2f}")
    print(f"6. Mean seeker search restarts: {searches.mean():.2f}")


def evaluate_policy(title, action_fn_factory, episodes, seed, render, overrides, obstacles=None):
    env = PPOEnv(
        render_mode="human" if render else None,
        randomize_start=True,
        obstacles=obstacles,
    )
    env.set_params(**overrides)

    print(f"\n=== {title} ===")
    print(f"Seed: {seed}")
    print(f"Difficulty: {env.difficulty_summary()}")

    action_fn = action_fn_factory(env)
    summaries = []
    for episode in range(episodes):
        summary = run_episode(env, action_fn, seed + episode)
        summaries.append(summary)
        print_episode(episode, summary)

    env.close()
    print_summary(summaries)


def evaluate_ppo(model_path, episodes, seed, render, overrides, obstacles=None):
    print(f"Model: {model_path}")
    model = PPO.load(model_path)

    def factory(env):
        return lambda obs: model.predict(obs, deterministic=True)[0]

    evaluate_policy("PPO EVALUATION", factory, episodes, seed, render, overrides, obstacles)


def evaluate_random(episodes, seed, render, overrides, obstacles=None):

    def factory(env):
        env.action_space.seed(seed)
        return lambda obs: env.action_space.sample()

    evaluate_policy("RANDOM BASELINE", factory, episodes, seed, render, overrides, obstacles)


def add_difficulty_args(parser):
    parser.add_argument("--seeker-speed", type=float, default=None,
                        help="Override Seeker speed in m/s (default 1.8)")
    parser.add_argument("--fov", type=float, default=None,
                        help="Override camera field of view in degrees (default 90)")
    parser.add_argument("--range", dest="detection_range", type=float, default=None,
                        help="Override detection range in metres (default 10)")
    parser.add_argument("--obstacles", default=None, choices=sorted(BOX_MAPS),
                        help="Obstacle preset (default: none). Changes the observation size, so "
                             "a model must be trained and evaluated with the same preset")


def overrides_from_args(args):
    return {
        "seeker_speed": args.seeker_speed,
        "fov_deg": args.fov,
        "detection_range": args.detection_range,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Vanishing Point PPO Evader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train PPO Evader")
    train_parser.add_argument("--timesteps", type=int, default=200_000)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument(
        "--model", type=str, default=str(MODELS / "vanishing_point_ppo")
    )
    train_parser.add_argument(
        "--tensorboard-log", type=str, default=None,
        help="Optional TensorBoard log directory",
    )
    train_parser.add_argument(
        "--curriculum", action="store_true",
        help="Train from an easy Seeker up to full difficulty",
    )
    train_parser.add_argument("--obstacles", default=None, choices=sorted(BOX_MAPS),
                              help="Train inside an obstacle preset (default: none)")
    train_parser.add_argument(
        "--stages", type=int, default=4,
        help="Number of curriculum stages (used with --curriculum)",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate trained PPO")
    eval_parser.add_argument(
        "--model", type=str, default=str(MODELS / "vanishing_point_ppo")
    )
    eval_parser.add_argument("--episodes", type=int, default=5)
    eval_parser.add_argument("--seed", type=int, default=12345)
    eval_parser.add_argument("--render", action="store_true")
    add_difficulty_args(eval_parser)

    random_parser = subparsers.add_parser("random", help="Evaluate random baseline")
    random_parser.add_argument("--episodes", type=int, default=20)
    random_parser.add_argument("--seed", type=int, default=12345)
    random_parser.add_argument("--render", action="store_true")
    add_difficulty_args(random_parser)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        seed = args.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "little")
        print(f"Training seed: {seed}")
        train_ppo(
            total_timesteps=args.timesteps,
            seed=seed,
            model_path=args.model,
            tensorboard_log=args.tensorboard_log,
            curriculum=args.curriculum,
            stages=args.stages,
            obstacles=args.obstacles,
        )

    elif args.command == "eval":
        evaluate_ppo(
            model_path=args.model,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            overrides=overrides_from_args(args),
            obstacles=args.obstacles,
        )

    elif args.command == "random":
        evaluate_random(
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            overrides=overrides_from_args(args),
            obstacles=args.obstacles,
        )


if __name__ == "__main__":
    main()
