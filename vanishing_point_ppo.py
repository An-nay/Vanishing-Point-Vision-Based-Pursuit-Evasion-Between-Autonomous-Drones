"""
Vanishing Point - Phase 1 PPO Evader

- Scripted Seeker with TRACK / SCAN / SEARCH modes
- PPO-controlled Evader
- Bounded 3D arena
- Geometric FOV detection
- Visibility-based reward
- One-time wall-contact penalty
- Randomized, reproducible episode starts

Training is headless and lightweight. PyBullet is used only for human rendering.
"""

import argparse
import math
import os
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


# -----------------------------
# Configuration
# -----------------------------
ARENA_X = 20.0
ARENA_Y = 20.0
ARENA_Z = 10.0
ARENA_SIZE = np.array([ARENA_X, ARENA_Y, ARENA_Z], dtype=np.float32)

DT = 0.1
EPISODE_STEPS = 600  # 60 s at dt=0.1

SEEKER_SPEED = 1.8
EVADER_SPEED = 2.2

FOV_DEG = 90.0
DETECTION_RANGE = 10.0

DRONE_RADIUS = 0.35
BOUNDARY_MARGIN = 0.4

REWARD_HIDDEN = 1.0
REWARD_VISIBLE = -1.0
REWARD_BOUNDARY = -2.0  # first contact only

# Seeker search behaviour
ARRIVE_DIST = 0.5
SCAN_TIMEOUT = 4.0
SCAN_YAW_RATE = 100.0
SCAN_PITCH_MAX = 45.0
SCAN_PITCH_PERIOD = 2.0
WAYPOINT_MARGIN = 1.0

# Policy observation choice.
# False keeps visibility as a reward signal, not an explicit policy input.
# Set True if you want to match an observation specification that includes it.
INCLUDE_VISIBILITY_IN_OBS = False

# Curriculum: difficulty 0.0 is an easy Seeker and 1.0 is the full settings above.
# Speed, FOV and detection range are blended linearly between the two.
EASY_SEEKER_SPEED = 1.0
EASY_FOV_DEG = 60.0
EASY_DETECTION_RANGE = 6.0


# -----------------------------
# Utility functions
# -----------------------------
def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def clamp_position(pos):
    x = np.clip(pos[0], BOUNDARY_MARGIN, ARENA_X - BOUNDARY_MARGIN)
    y = np.clip(pos[1], BOUNDARY_MARGIN, ARENA_Y - BOUNDARY_MARGIN)
    z = np.clip(pos[2], BOUNDARY_MARGIN, ARENA_Z - BOUNDARY_MARGIN)
    return np.array([x, y, z], dtype=np.float64)


def hit_boundary(pos):
    return bool(
        pos[0] <= BOUNDARY_MARGIN
        or pos[0] >= ARENA_X - BOUNDARY_MARGIN
        or pos[1] <= BOUNDARY_MARGIN
        or pos[1] >= ARENA_Y - BOUNDARY_MARGIN
        or pos[2] <= BOUNDARY_MARGIN
        or pos[2] >= ARENA_Z - BOUNDARY_MARGIN
    )


def inside_valid_arena(pos, margin=BOUNDARY_MARGIN):
    return bool(
        margin < pos[0] < ARENA_X - margin
        and margin < pos[1] < ARENA_Y - margin
        and margin < pos[2] < ARENA_Z - margin
    )


def is_visible(
    seeker_pos,
    seeker_forward,
    evader_pos,
    fov_deg=FOV_DEG,
    detection_range=DETECTION_RANGE,
):
    to_evader = np.asarray(evader_pos) - np.asarray(seeker_pos)
    distance = np.linalg.norm(to_evader)

    if distance < 1e-8 or distance > detection_range:
        return False, float(distance), 180.0

    direction_to_evader = to_evader / distance
    seeker_forward = normalize(seeker_forward)
    cos_theta = np.clip(np.dot(seeker_forward, direction_to_evader), -1.0, 1.0)
    angle_deg = math.degrees(math.acos(cos_theta))
    visible = angle_deg <= (fov_deg / 2.0)
    return bool(visible), float(distance), float(angle_deg)


def limit_velocity(action, max_speed):
    """Map PPO action in [-1,1]^3 to a velocity with bounded magnitude."""
    direction = np.asarray(action, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 1.0:
        direction = direction / norm
    return direction * max_speed


# -----------------------------
# Scripted Seeker
# -----------------------------
class ScriptedSeekerPolicy:
    """Rule-based Seeker: TRACK -> SCAN -> SEARCH, reacquiring when visible."""

    def __init__(self, rng, initial_forward=np.array([1.0, 0.0, 0.0]), speed=SEEKER_SPEED):
        self.rng = rng
        self.speed = speed
        self.forward = normalize(initial_forward)
        self.last_known_position = None
        self.waypoint = None
        self.searches = 0
        self._start_scan()

    def _start_scan(self):
        self.mode = "SCAN"
        self.scan_time = 0.0
        self.yaw = math.atan2(self.forward[1], self.forward[0])

    def _set_forward_from_angles(self, yaw, pitch):
        self.forward = normalize(
            np.array(
                [
                    math.cos(pitch) * math.cos(yaw),
                    math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch),
                ]
            )
        )

    def _fly_toward(self, seeker_pos, target):
        vec = target - seeker_pos
        if np.linalg.norm(vec) < ARRIVE_DIST:
            return None
        self.forward = normalize(vec)
        return self.forward * self.speed

    def act(self, seeker_pos, evader_pos, visible):
        if visible:
            self.last_known_position = np.array(evader_pos, dtype=np.float64)
            self.mode = "TRACK"

        if self.mode == "TRACK":
            velocity = self._fly_toward(seeker_pos, self.last_known_position)
            if velocity is not None:
                return velocity
            self._start_scan()
            return np.zeros(3, dtype=np.float64)

        if self.mode == "SEARCH":
            velocity = self._fly_toward(seeker_pos, self.waypoint)
            if velocity is not None:
                return velocity
            self._start_scan()
            return np.zeros(3, dtype=np.float64)

        # SCAN: yaw continuously while pitch oscillates.
        self.scan_time += DT
        self.yaw += math.radians(SCAN_YAW_RATE) * DT
        pitch = math.radians(SCAN_PITCH_MAX) * math.sin(
            2.0 * math.pi * self.scan_time / SCAN_PITCH_PERIOD
        )
        self._set_forward_from_angles(self.yaw, pitch)

        if self.scan_time >= SCAN_TIMEOUT:
            low = np.array([WAYPOINT_MARGIN] * 3, dtype=np.float64)
            high = np.array(
                [
                    ARENA_X - WAYPOINT_MARGIN,
                    ARENA_Y - WAYPOINT_MARGIN,
                    ARENA_Z - WAYPOINT_MARGIN,
                ],
                dtype=np.float64,
            )
            self.waypoint = self.rng.uniform(low, high)
            self.mode = "SEARCH"
            self.searches += 1

        return np.zeros(3, dtype=np.float64)


# -----------------------------
# Gymnasium Environment
# -----------------------------
class VanishingPointEnv(gym.Env):
    """Stage-1 MDP: PPO Evader versus fixed scripted Seeker."""

    metadata = {"render_modes": ["human"], "render_fps": int(1 / DT)}

    def __init__(self, render_mode=None, randomize_start=True):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.render_mode = render_mode
        self.randomize_start = randomize_start

        # Seeker difficulty settings (defaults are the constants at the top).
        self.seeker_speed = SEEKER_SPEED
        self.fov_deg = FOV_DEG
        self.detection_range = DETECTION_RANGE

        # PPO outputs a normalized 3-D control vector. It is scaled to EVADER_SPEED.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation:
        #   evader position normalized around arena center             3
        #   evader velocity / max speed                              3
        #   relative seeker position / arena dimensions              3
        #   relative seeker velocity / combined speed                3
        #   seeker camera forward vector                             3
        #   normalized distances to x-/x+/y-/y+/z-/z+ boundaries    6
        #   optional visibility flag                                 1
        self.obs_dim = 21 + (1 if INCLUDE_VISIBILITY_IN_OBS else 0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.physics_client = None
        self.seeker_id = None
        self.evader_id = None

        if self.render_mode == "human":
            self._init_renderer()

        self.seeker_pos = None
        self.evader_pos = None
        self.seeker_velocity = np.zeros(3, dtype=np.float64)
        self.evader_velocity = np.zeros(3, dtype=np.float64)
        self.seeker_policy = None
        self.evader_was_on_boundary = False
        self.step_count = 0

        self.total_reward = 0.0
        self.undetected_steps = 0
        self.wall_touches = 0
        self.first_loss_step = None
        self.current_hidden_streak = 0
        self.longest_hidden_streak = 0

    # ---------- difficulty control ----------
    def set_params(self, seeker_speed=None, fov_deg=None, detection_range=None):
        """Override Seeker speed, camera FOV and detection range. None keeps the current value."""
        if seeker_speed is not None:
            self.seeker_speed = float(seeker_speed)
        if fov_deg is not None:
            self.fov_deg = float(fov_deg)
        if detection_range is not None:
            self.detection_range = float(detection_range)

    def set_difficulty(self, level):
        """0.0 = easy Seeker, 1.0 = full difficulty (the constants at the top)."""
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
        return is_visible(
            self.seeker_pos,
            self.seeker_policy.forward,
            self.evader_pos,
            self.fov_deg,
            self.detection_range,
        )

    # ---------- reset helpers ----------
    def _sample_initial_state(self):
        """Randomize starts while making the Evader initially visible."""
        if not self.randomize_start:
            return (
                np.array([4.0, 10.0, 5.0], dtype=np.float64),
                np.array([12.0, 10.0, 5.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )

        # Sample from the central arena so a valid visible Evader is easy to place.
        seeker_low = np.array([3.0, 3.0, 2.0])
        seeker_high = np.array([17.0, 17.0, 8.0])

        for _ in range(200):
            seeker_pos = self.np_random.uniform(seeker_low, seeker_high)
            initial_forward = normalize(self.np_random.normal(size=3))
            if np.linalg.norm(initial_forward) < 1e-8:
                continue

            # Keep the initial Evader clearly inside FOV and detection range.
            candidate_dir = normalize(
                initial_forward + self.np_random.normal(0.0, 0.18, size=3)
            )
            distance = self.np_random.uniform(4.0, 8.0)
            evader_pos = seeker_pos + candidate_dir * distance

            if not inside_valid_arena(evader_pos, margin=1.0):
                continue

            visible, _, _ = is_visible(
                seeker_pos, initial_forward, evader_pos,
                self.fov_deg, self.detection_range,
            )
            if visible:
                return seeker_pos, evader_pos, initial_forward

        # Safe fallback if random sampling somehow fails.
        return (
            np.array([4.0, 10.0, 5.0], dtype=np.float64),
            np.array([12.0, 10.0, 5.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.seeker_pos, self.evader_pos, initial_forward = self._sample_initial_state()
        self.seeker_velocity = np.zeros(3, dtype=np.float64)
        self.evader_velocity = np.zeros(3, dtype=np.float64)
        self.seeker_policy = ScriptedSeekerPolicy(
            rng=self.np_random,
            initial_forward=initial_forward,
            speed=self.seeker_speed,
        )

        self.evader_was_on_boundary = False
        self.step_count = 0
        self.total_reward = 0.0
        self.undetected_steps = 0
        self.wall_touches = 0
        self.first_loss_step = None
        self.current_hidden_streak = 0
        self.longest_hidden_streak = 0

        visible, distance, angle = self._check_visibility()

        if self.render_mode == "human":
            self._reset_render_objects()
            self._render_step(visible, distance, angle, reward=0.0)

        observation = self._get_observation(visible)
        info = self._get_info(visible, distance, angle, boundary_contact=False)
        return observation, info

    # ---------- RL step ----------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Visibility before actions determines what the scripted Seeker currently knows.
        visible_before, _, _ = self._check_visibility()

        self.evader_velocity = limit_velocity(action, EVADER_SPEED)
        self.seeker_velocity = self.seeker_policy.act(
            self.seeker_pos, self.evader_pos, visible_before
        )

        proposed_evader_pos = self.evader_pos + self.evader_velocity * DT
        proposed_seeker_pos = self.seeker_pos + self.seeker_velocity * DT

        evader_boundary = hit_boundary(proposed_evader_pos)

        self.evader_pos = clamp_position(proposed_evader_pos)
        self.seeker_pos = clamp_position(proposed_seeker_pos)

        visible, distance, angle = self._check_visibility()

        # Main reward signal: hide from the Seeker.
        reward = REWARD_VISIBLE if visible else REWARD_HIDDEN

        # One-time wall-contact penalty.
        new_wall_contact = evader_boundary and not self.evader_was_on_boundary
        if new_wall_contact:
            reward += REWARD_BOUNDARY
            self.wall_touches += 1
        self.evader_was_on_boundary = evader_boundary

        self.step_count += 1
        self.total_reward += float(reward)

        if not visible:
            self.undetected_steps += 1
            self.current_hidden_streak += 1
            self.longest_hidden_streak = max(
                self.longest_hidden_streak, self.current_hidden_streak
            )
            if self.first_loss_step is None:
                self.first_loss_step = self.step_count
        else:
            self.current_hidden_streak = 0

        # A wall touch is not terminal; time limit ends the episode.
        terminated = False
        truncated = self.step_count >= EPISODE_STEPS

        observation = self._get_observation(visible)
        info = self._get_info(visible, distance, angle, new_wall_contact)

        if truncated:
            info["episode_summary"] = {
                "total_reward": self.total_reward,
                "undetected_ratio": self.undetected_steps / self.step_count,
                "first_loss_time_s": (
                    None
                    if self.first_loss_step is None
                    else self.first_loss_step * DT
                ),
                "longest_hidden_s": self.longest_hidden_streak * DT,
                "wall_touches": self.wall_touches,
                "seeker_search_restarts": self.seeker_policy.searches,
            }

        if self.render_mode == "human":
            self._render_step(visible, distance, angle, reward)
            time.sleep(DT)

        return observation, float(reward), terminated, truncated, info

    # ---------- observation ----------
    def _get_observation(self, visible):
        # Positions centered and scaled roughly to [-1, 1].
        center = ARENA_SIZE / 2.0
        evader_pos_norm = (self.evader_pos.astype(np.float32) - center) / center

        evader_vel_norm = (self.evader_velocity / EVADER_SPEED).astype(np.float32)

        relative_pos = (
            (self.seeker_pos - self.evader_pos) / ARENA_SIZE
        ).astype(np.float32)

        relative_vel = (
            (self.seeker_velocity - self.evader_velocity)
            / (SEEKER_SPEED + EVADER_SPEED)
        ).astype(np.float32)

        seeker_forward = self.seeker_policy.forward.astype(np.float32)

        x, y, z = self.evader_pos
        boundary_distances = np.array(
            [
                x / ARENA_X,
                (ARENA_X - x) / ARENA_X,
                y / ARENA_Y,
                (ARENA_Y - y) / ARENA_Y,
                z / ARENA_Z,
                (ARENA_Z - z) / ARENA_Z,
            ],
            dtype=np.float32,
        )

        parts = [
            evader_pos_norm,
            evader_vel_norm,
            relative_pos,
            relative_vel,
            seeker_forward,
            boundary_distances,
        ]

        if INCLUDE_VISIBILITY_IN_OBS:
            parts.append(np.array([1.0 if visible else 0.0], dtype=np.float32))

        observation = np.concatenate(parts).astype(np.float32)
        return np.clip(observation, -1.0, 1.0)

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

    # ---------- PyBullet rendering ----------
    def _init_renderer(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)
        p.setTimeStep(DT, physicsClientId=self.physics_client)
        p.resetDebugVisualizerCamera(
            cameraDistance=28,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2],
            physicsClientId=self.physics_client,
        )
        self._create_arena()

    def _create_arena(self):
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[ARENA_X / 2, ARENA_Y / 2, 0.05],
            physicsClientId=self.physics_client,
        )
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[ARENA_X / 2, ARENA_Y / 2, 0.05],
            rgbaColor=[0.75, 0.75, 0.75, 1.0],
            physicsClientId=self.physics_client,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_shape,
            baseVisualShapeIndex=floor_visual,
            basePosition=[ARENA_X / 2, ARENA_Y / 2, -0.05],
            physicsClientId=self.physics_client,
        )

        corners = [
            [0, 0, 0], [ARENA_X, 0, 0], [ARENA_X, ARENA_Y, 0], [0, ARENA_Y, 0],
            [0, 0, ARENA_Z], [ARENA_X, 0, ARENA_Z],
            [ARENA_X, ARENA_Y, ARENA_Z], [0, ARENA_Y, ARENA_Z],
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            p.addUserDebugLine(
                corners[a], corners[b], [0.2, 0.2, 0.2],
                lineWidth=1.5, lifeTime=0,
                physicsClientId=self.physics_client,
            )

    def _create_drone(self, position, rgba):
        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=DRONE_RADIUS,
            physicsClientId=self.physics_client,
        )
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=DRONE_RADIUS,
            rgbaColor=rgba,
            physicsClientId=self.physics_client,
        )
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.physics_client,
        )

    def _reset_render_objects(self):
        if self.seeker_id is None:
            self.seeker_id = self._create_drone(
                self.seeker_pos.tolist(), [1.0, 0.2, 0.2, 1.0]
            )
        if self.evader_id is None:
            self.evader_id = self._create_drone(
                self.evader_pos.tolist(), [0.2, 0.4, 1.0, 1.0]
            )

        p.resetBasePositionAndOrientation(
            self.seeker_id, self.seeker_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self.physics_client,
        )
        p.resetBasePositionAndOrientation(
            self.evader_id, self.evader_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self.physics_client,
        )

    def _render_step(self, visible, distance, angle, reward):
        if self.seeker_id is None or self.evader_id is None:
            self._reset_render_objects()

        p.resetBasePositionAndOrientation(
            self.seeker_id, self.seeker_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self.physics_client,
        )
        p.resetBasePositionAndOrientation(
            self.evader_id, self.evader_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self.physics_client,
        )

        forward_end = self.seeker_pos + self.seeker_policy.forward * self.detection_range
        line_colour = [1, 0, 0] if visible else [0, 1, 0]
        p.addUserDebugLine(
            self.seeker_pos.tolist(),
            forward_end.tolist(),
            line_colour,
            lineWidth=3,
            lifeTime=DT,
            physicsClientId=self.physics_client,
        )
        p.addUserDebugText(
            (
                f"Step: {self.step_count} | Mode: {self.seeker_policy.mode} | "
                f"Visible: {visible} | Reward: {reward:.1f} | "
                f"Dist: {distance:.1f}m | Angle: {angle:.1f}deg"
            ),
            [0.5, 0.5, ARENA_Z + 0.7],
            textColorRGB=[0, 0, 0],
            textSize=1.1,
            lifeTime=DT,
            physicsClientId=self.physics_client,
        )
        p.stepSimulation(physicsClientId=self.physics_client)

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except p.error:
                pass
            self.physics_client = None


# -----------------------------
# Training / evaluation
# -----------------------------
def train_ppo(
    total_timesteps,
    seed,
    model_path,
    tensorboard_log=None,
    curriculum=False,
    stages=4,
):
    print("\n=== PPO TRAINING ===")
    print(f"Seed: {seed}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Visibility in observation: {INCLUDE_VISIBILITY_IN_OBS}")
    if curriculum:
        print(f"Curriculum: ON ({stages} stages, easy to full difficulty)")
    else:
        print("Curriculum: OFF (full difficulty throughout)")

    raw_env = VanishingPointEnv(render_mode=None, randomize_start=True)
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
    """Run one full episode and return the environment's episode summary."""
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
    """Print the same six metrics for every evaluation mode."""
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


def evaluate_policy(title, action_fn_factory, episodes, seed, render, overrides):
    env = VanishingPointEnv(
        render_mode="human" if render else None,
        randomize_start=True,
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


def evaluate_ppo(model_path, episodes, seed, render, overrides):
    print(f"Model: {model_path}")
    model = PPO.load(model_path)

    def factory(env):
        return lambda obs: model.predict(obs, deterministic=True)[0]

    evaluate_policy("PPO EVALUATION", factory, episodes, seed, render, overrides)


def evaluate_random(episodes, seed, render, overrides):
    """Baseline using random actions from the same PPO action space."""

    def factory(env):
        env.action_space.seed(seed)
        return lambda obs: env.action_space.sample()

    evaluate_policy("RANDOM BASELINE", factory, episodes, seed, render, overrides)


def add_difficulty_args(parser):
    parser.add_argument("--seeker-speed", type=float, default=None,
                        help="Override Seeker speed in m/s (default 1.8)")
    parser.add_argument("--fov", type=float, default=None,
                        help="Override camera field of view in degrees (default 90)")
    parser.add_argument("--range", dest="detection_range", type=float, default=None,
                        help="Override detection range in metres (default 10)")


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
        "--model", type=str, default="models/vanishing_point_ppo"
    )
    train_parser.add_argument(
        "--tensorboard-log", type=str, default=None,
        help="Optional TensorBoard log directory",
    )
    train_parser.add_argument(
        "--curriculum", action="store_true",
        help="Train from an easy Seeker up to full difficulty",
    )
    train_parser.add_argument(
        "--stages", type=int, default=4,
        help="Number of curriculum stages (used with --curriculum)",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate trained PPO")
    eval_parser.add_argument(
        "--model", type=str, default="models/vanishing_point_ppo"
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
        )

    elif args.command == "eval":
        evaluate_ppo(
            model_path=args.model,
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            overrides=overrides_from_args(args),
        )

    elif args.command == "random":
        evaluate_random(
            episodes=args.episodes,
            seed=args.seed,
            render=args.render,
            overrides=overrides_from_args(args),
        )


if __name__ == "__main__":
    main()
