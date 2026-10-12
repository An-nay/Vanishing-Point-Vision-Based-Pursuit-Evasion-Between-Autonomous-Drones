"""Shared NumPy dynamics and original scripted Seeker. No PyTorch/PyBullet imports."""
import math
import numpy as np

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



def checked_vector(value, size, name):
    """Reject invalid inputs before any world mutation; clip finite controls elsewhere."""
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return value.copy()


def sample_initial_state(rng, randomize_start=True, fov_deg=FOV_DEG,
                         detection_range=DETECTION_RANGE):
    """Exactly the Stage 1 visible-start sampler, including its fallback."""
    if randomize_start:
        for _ in range(200):
            seeker = rng.uniform([3.0, 3.0, 2.0], [17.0, 17.0, 8.0])
            forward = normalize(rng.normal(size=3))
            if np.linalg.norm(forward) < 1e-8:
                continue
            direction = normalize(forward + rng.normal(0.0, 0.18, size=3))
            evader = seeker + direction * rng.uniform(4.0, 8.0)
            if inside_valid_arena(evader, margin=1.0) and is_visible(
                    seeker, forward, evader, fov_deg, detection_range)[0]:
                return seeker, evader, forward
    return (np.array([4.0, 10.0, 5.0]), np.array([12.0, 10.0, 5.0]),
            np.array([1.0, 0.0, 0.0]))


def boundary_distances(position):
    return np.array([position[0] / ARENA_X, (ARENA_X - position[0]) / ARENA_X,
                     position[1] / ARENA_Y, (ARENA_Y - position[1]) / ARENA_Y,
                     position[2] / ARENA_Z, (ARENA_Z - position[2]) / ARENA_Z],
                    dtype=np.float32)


class Simulator:
    """NumPy world dynamics shared by Gymnasium and PettingZoo; no RL/render imports."""

    def __init__(self, episode_steps=EPISODE_STEPS, randomize_start=True,
                 yaw_rate_deg=100.0, pitch_rate_deg=90.0, pitch_limit_deg=89.0):
        if not isinstance(episode_steps, int) or episode_steps <= 0:
            raise ValueError("episode_steps must be a positive integer")
        if not np.all(np.isfinite([yaw_rate_deg, pitch_rate_deg, pitch_limit_deg])):
            raise ValueError("camera parameters must be finite")
        if min(yaw_rate_deg, pitch_rate_deg) <= 0 or not 0 < pitch_limit_deg <= 90:
            raise ValueError("turn rates must be positive and pitch limit in (0, 90]")
        self.episode_steps = episode_steps
        self.randomize_start = randomize_start
        self.yaw_rate = math.radians(yaw_rate_deg)
        self.pitch_rate = math.radians(pitch_rate_deg)
        self.pitch_limit = math.radians(pitch_limit_deg)
        self.seeker_speed = SEEKER_SPEED
        self.fov_deg = FOV_DEG
        self.detection_range = DETECTION_RANGE
        self.seeker_pos = self.evader_pos = None

    def reset_world(self, rng, initial_state=None):
        if initial_state is None:
            seeker, evader, forward = sample_initial_state(
                rng, self.randomize_start, self.fov_deg, self.detection_range)
        else:
            if set(initial_state) != {"seeker_pos", "evader_pos", "seeker_forward"}:
                raise ValueError("initial_state requires seeker_pos, evader_pos, seeker_forward")
            seeker = checked_vector(initial_state["seeker_pos"], 3, "seeker_pos")
            evader = checked_vector(initial_state["evader_pos"], 3, "evader_pos")
            forward = checked_vector(initial_state["seeker_forward"], 3, "seeker_forward")
            if any(not np.array_equal(pos, clamp_position(pos)) for pos in (seeker, evader)):
                raise ValueError("initial positions must be inside the arena boundary margin")
            if np.linalg.norm(forward) < 1e-8:
                raise ValueError("seeker_forward cannot be zero")
            forward = normalize(forward)
        self.seeker_pos, self.evader_pos = seeker.copy(), evader.copy()
        self.seeker_velocity = np.zeros(3)
        self.evader_velocity = np.zeros(3)
        self.initial_forward = forward.copy()
        self.set_camera_forward(forward)
        self.step_count = 0
        self.total_reward = self.seeker_total_reward = 0.0
        self.evader_was_on_boundary = self.seeker_was_on_boundary = False
        self.wall_touches = self.seeker_wall_touches = 0
        self.boundary_steps = {"seeker": 0, "evader": 0}
        self.undetected_steps = self.current_hidden_streak = self.longest_hidden_streak = 0
        self.first_loss_step = None
        self.reacquisition_steps = []
        self.loss_events = 0
        self.last_known_position = None
        self.time_since_detection_steps = 0
        self.visible, self.distance, self.angle = self.check_visibility()
        self.initially_visible = self.visible
        self._update_detection_memory()
        return self.visible, self.distance, self.angle

    def set_camera_forward(self, forward):
        self.camera_forward = normalize(checked_vector(forward, 3, "seeker_forward"))
        if np.linalg.norm(self.camera_forward) < 1e-8:
            raise ValueError("seeker_forward cannot be zero")
        self.camera_yaw = math.atan2(self.camera_forward[1], self.camera_forward[0])
        self.camera_pitch = math.asin(float(np.clip(self.camera_forward[2], -1.0, 1.0)))

    def rotate_camera(self, controls):
        controls = np.clip(checked_vector(controls, 2, "camera action"), -1.0, 1.0)
        self.camera_yaw = (self.camera_yaw + controls[0] * self.yaw_rate * DT
                           + math.pi) % (2 * math.pi) - math.pi
        self.camera_pitch = float(np.clip(
            self.camera_pitch + controls[1] * self.pitch_rate * DT,
            -self.pitch_limit, self.pitch_limit))
        yaw, pitch = self.camera_yaw, self.camera_pitch
        self.camera_forward = np.array([math.cos(pitch) * math.cos(yaw),
                                       math.cos(pitch) * math.sin(yaw), math.sin(pitch)])

    def check_visibility(self):
        return is_visible(self.seeker_pos, self.camera_forward, self.evader_pos,
                          self.fov_deg, self.detection_range)

    def _update_detection_memory(self):
        if self.visible:
            self.last_known_position = self.evader_pos.copy()
            self.time_since_detection_steps = 0

    def advance(self, seeker_velocity, evader_velocity, *, camera_action=None,
                seeker_forward=None):
        """Commit one joint transition. All inputs are checked before mutation.

        Scripted baselines supply an absolute forward vector (their original
        instantaneous turns); learned Seekers supply bounded yaw/pitch rates.
        """
        if self.seeker_pos is None:
            raise RuntimeError("reset the simulator before stepping")
        if self.step_count >= self.episode_steps:
            raise RuntimeError("episode is complete; call reset")
        seeker_velocity = checked_vector(seeker_velocity, 3, "seeker velocity")
        evader_velocity = checked_vector(evader_velocity, 3, "evader velocity")
        if camera_action is not None and seeker_forward is not None:
            raise ValueError("provide camera_action or seeker_forward, not both")
        if camera_action is not None:
            checked_vector(camera_action, 2, "camera action")
        if seeker_forward is not None:
            checked_vector(seeker_forward, 3, "seeker_forward")
            if np.linalg.norm(seeker_forward) < 1e-8:
                raise ValueError("seeker_forward cannot be zero")
        if camera_action is not None:
            self.rotate_camera(camera_action)
        elif seeker_forward is not None:
            self.set_camera_forward(seeker_forward)
            # Scripted policy already normalizes this vector. Preserve its exact
            # value so Stage 1 FOV angles do not acquire re-normalization drift.
            self.camera_forward = np.asarray(seeker_forward, dtype=np.float64).copy()
        self.seeker_velocity, self.evader_velocity = seeker_velocity, evader_velocity
        proposed = {"seeker": self.seeker_pos + seeker_velocity * DT,
                    "evader": self.evader_pos + evader_velocity * DT}
        contacts = {a: hit_boundary(pos) for a, pos in proposed.items()}
        fresh = {"seeker": contacts["seeker"] and not self.seeker_was_on_boundary,
                 "evader": contacts["evader"] and not self.evader_was_on_boundary}
        self.seeker_pos = clamp_position(proposed["seeker"])
        self.evader_pos = clamp_position(proposed["evader"])
        was_visible = self.visible
        self.visible, self.distance, self.angle = self.check_visibility()
        evader_reward = REWARD_VISIBLE if self.visible else REWARD_HIDDEN
        seeker_reward = -evader_reward
        if fresh["evader"]:
            evader_reward += REWARD_BOUNDARY
            self.wall_touches += 1
        if fresh["seeker"]:
            seeker_reward += REWARD_BOUNDARY
            self.seeker_wall_touches += 1
        self.evader_was_on_boundary = contacts["evader"]
        self.seeker_was_on_boundary = contacts["seeker"]
        for agent in contacts:
            self.boundary_steps[agent] += int(contacts[agent])
        self.step_count += 1
        self.total_reward += evader_reward
        self.seeker_total_reward += seeker_reward
        if not self.visible:
            self.time_since_detection_steps += 1
            self.undetected_steps += 1
            self.current_hidden_streak += 1
            self.longest_hidden_streak = max(self.longest_hidden_streak, self.current_hidden_streak)
            if was_visible:
                self.loss_events += 1
                if self.first_loss_step is None:
                    self.first_loss_step = self.step_count
        else:
            if self.current_hidden_streak:
                self.reacquisition_steps.append(self.current_hidden_streak)
            self.current_hidden_streak = 0
        self._update_detection_memory()
        return ({"seeker": float(seeker_reward), "evader": float(evader_reward)},
                fresh, self.step_count >= self.episode_steps)

    def evader_observation(self, include_visibility=False):
        center = ARENA_SIZE / 2.0
        parts = [(self.evader_pos.astype(np.float32) - center) / center,
                 (self.evader_velocity / EVADER_SPEED).astype(np.float32),
                 ((self.seeker_pos - self.evader_pos) / ARENA_SIZE).astype(np.float32),
                 ((self.seeker_velocity - self.evader_velocity)
                  / (SEEKER_SPEED + EVADER_SPEED)).astype(np.float32),
                 self.camera_forward.astype(np.float32), boundary_distances(self.evader_pos)]
        if include_visibility:
            parts.append(np.array([float(self.visible)], dtype=np.float32))
        return self._bounded_observation(parts)

    def seeker_observation(self):
        # Relative current target only when visible; stale last-seen target otherwise.
        # No target velocity, true distance, angle or reward-history input.
        known = self.last_known_position is not None
        relative = ((self.last_known_position - self.seeker_pos) / ARENA_SIZE
                    if known else np.zeros(3))
        center = ARENA_SIZE / 2.0
        return self._bounded_observation([
            (self.seeker_pos - center) / center, self.seeker_velocity / SEEKER_SPEED,
            self.camera_forward, boundary_distances(self.seeker_pos),
            [float(self.visible)], relative, [float(known)],
            [min(self.time_since_detection_steps / self.episode_steps, 1.0)]])

    @staticmethod
    def _bounded_observation(parts):
        obs = np.concatenate(parts).astype(np.float32)
        if not np.all(np.isfinite(obs)):
            raise ValueError("non-finite world state produced an observation")
        return np.clip(obs, -1.0, 1.0)

    def global_state(self):
        """26 normalized values for critics only, including hidden target state."""
        center = ARENA_SIZE / 2.0
        known = self.last_known_position is not None
        last = ((self.last_known_position - center) / center if known else np.zeros(3))
        return self._bounded_observation([
            (self.seeker_pos - center) / center, self.seeker_velocity / SEEKER_SPEED,
            (self.evader_pos - center) / center, self.evader_velocity / EVADER_SPEED,
            self.camera_forward, last,
            [float(known), self.time_since_detection_steps / self.episode_steps,
             float(self.visible), float(self.seeker_was_on_boundary),
             float(self.evader_was_on_boundary), self.step_count / self.episode_steps,
             self.camera_yaw / math.pi, self.camera_pitch / (math.pi / 2)]])

    def episode_summary(self):
        count = max(1, self.step_count)
        return {
            "steps": self.step_count,
            "seeker_reward": self.seeker_total_reward, "evader_reward": self.total_reward,
            "undetected_pct": 100 * self.undetected_steps / count,
            "visible_pct": 100 * (self.step_count - self.undetected_steps) / count,
            "initially_visible": self.initially_visible,
            "first_loss_time_s": None if self.first_loss_step is None else self.first_loss_step * DT,
            "longest_hidden_s": self.longest_hidden_streak * DT,
            "seeker_wall_contacts": self.seeker_wall_touches,
            "evader_wall_contacts": self.wall_touches,
            "seeker_boundary_steps": self.boundary_steps["seeker"],
            "evader_boundary_steps": self.boundary_steps["evader"],
            "loss_events": self.loss_events, "reacquisitions": len(self.reacquisition_steps),
            "reacquisition_mean_s": (float(np.mean(self.reacquisition_steps)) * DT
                                     if self.reacquisition_steps else None),
            "unresolved_hidden_s": self.current_hidden_streak * DT,
        }
