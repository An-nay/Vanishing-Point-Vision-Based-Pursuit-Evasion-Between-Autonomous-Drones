import math
from collections import deque
import numpy as np

ARENA_X = 20.0
ARENA_Y = 20.0
ARENA_Z = 10.0
ARENA_SIZE = np.array([ARENA_X, ARENA_Y, ARENA_Z], dtype=np.float32)

DT = 0.1
EPISODE_STEPS = 600

SEEKER_SPEED = 1.8
EVADER_SPEED = 2.2

FOV_DEG = 90.0
DETECTION_RANGE = 10.0

DRONE_RADIUS = 0.35
BOUNDARY_MARGIN = 0.4

REWARD_HIDDEN = 1.0
REWARD_VISIBLE = -1.0
REWARD_BOUNDARY = -2.0

ARRIVE_DIST = 0.5
SCAN_TIMEOUT = 4.0
SCAN_YAW_RATE = 100.0
SCAN_PITCH_MAX = 45.0
SCAN_PITCH_PERIOD = 2.0
WAYPOINT_MARGIN = 1.0

SHOW_SEEN = False

EASY_SEEKER_SPEED = 1.0
EASY_FOV_DEG = 60.0
EASY_DETECTION_RANGE = 6.0


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


BOX_MAPS = {
    "none": [],
    "city": [
        ((9.0, 3.0, 0.0), (11.0, 12.0, 6.0)),
        ((4.0, 14.0, 0.0), (6.0, 16.0, 10.0)),
        ((14.0, 15.0, 0.0), (16.0, 17.0, 10.0)),
        ((14.0, 4.0, 0.0), (17.0, 7.0, 4.0)),
        ((3.0, 4.0, 0.0), (6.0, 7.0, 5.0)),
        ((8.0, 16.0, 0.0), (12.0, 18.0, 3.0)),
    ],
    "maze": [
        ((1.2, 16.0, 0.0), (3.6, 18.8, 10.0)),
        ((6.8, 15.4, 0.0), (12.8, 18.5, 2.2)),
        ((16.0, 15.2, 0.0), (18.8, 18.6, 10.0)),
        ((17.6, 1.2, 0.0), (19.4, 3.4, 10.0)),
        ((1.6, 1.4, 0.0), (5.8, 3.6, 3.0)),
        ((9.6, 1.4, 0.0), (11.4, 4.6, 6.5)),
        ((1.4, 7.4, 0.0), (4.2, 11.8, 4.6)),
        ((12.4, 5.2, 3.2), (14.6, 6.6, 4.6)),
        ((15.8, 8.0, 0.0), (18.8, 9.2, 6.5)),
        ((15.8, 9.2, 0.0), (16.8, 11.0, 3.5)),
        ((17.8, 9.2, 0.0), (18.8, 11.8, 8.2)),
    ],
    "pillars": [
        ((5.0, 5.0, 0.0), (7.0, 7.0, 10.0)),
        ((13.0, 6.0, 0.0), (15.0, 8.0, 10.0)),
        ((9.0, 13.0, 0.0), (11.0, 15.0, 10.0)),
    ],
}
LAMPS = ((4.8, 4.6), (4.8, 14.4), (15.2, 4.4), (15.2, 14.4))
TREES = ((6.2, 8.8), (15.0, 4.0))

NEAR_K = 3
BOX_SCALE = 10.0


def make_boxes(spec):
    if spec is None:
        spec = "none"
    if isinstance(spec, str):
        if spec not in BOX_MAPS:
            raise ValueError(f"unknown obstacle preset {spec!r}; choose from {sorted(BOX_MAPS)}")
        spec = BOX_MAPS[spec]
    boxes = np.asarray(spec, dtype=np.float64).reshape(-1, 2, 3)
    if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 0] >= boxes[:, 1]):
        raise ValueError("each obstacle needs finite min < max on every axis")
    return boxes


def in_box(point, boxes, margin=0.0):
    if len(boxes) == 0:
        return False
    return bool(np.any(np.all((point > boxes[:, 0] - margin) & (point < boxes[:, 1] + margin), axis=1)))


def ray_hits(start, end, boxes):
    if len(boxes) == 0:
        return False
    start = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - start
    for lo, hi in boxes:
        t_near, t_far, hit = 0.0, 1.0, True
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                if start[axis] < lo[axis] or start[axis] > hi[axis]:
                    hit = False
                    break
            else:
                t1, t2 = (lo[axis] - start[axis]) / delta[axis], (hi[axis] - start[axis]) / delta[axis]
                t_near, t_far = max(t_near, min(t1, t2)), min(t_far, max(t1, t2))
                if t_near > t_far:
                    hit = False
                    break
        if hit:
            return True
    return False


def slide(old, proposed, boxes, margin):
    position, blocked = np.array(old, dtype=np.float64), False
    for axis in range(3):
        candidate = position.copy()
        candidate[axis] = proposed[axis]
        if in_box(candidate, boxes, margin):
            blocked = True
        else:
            position = candidate
    return position, blocked


def close_obs(position, boxes, k=NEAR_K):
    vectors = np.clip(position, boxes[:, 0], boxes[:, 1]) - position
    order = np.argsort(np.linalg.norm(vectors, axis=1))[:k]
    out = np.ones((k, 3))
    out[:len(order)] = vectors[order] / BOX_SCALE
    return np.clip(out, -1.0, 1.0).ravel().astype(np.float32)


def can_see(
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


def cap_speed(action, max_speed):
    direction = np.asarray(action, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 1.0:
        direction = direction / norm
    return direction * max_speed


class SeekerBot:

    def __init__(self, rng, initial_forward=np.array([1.0, 0.0, 0.0]), speed=SEEKER_SPEED,
                 obstacles=None):
        self.rng = rng
        self.obstacles = None if obstacles is None or len(obstacles) == 0 else obstacles
        self._prev_pos, self._stuck = None, 0
        self.speed = speed
        self.forward = normalize(initial_forward)
        self.last_known_position = None
        self.last_known_velocity = np.zeros(3)
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
        if self.obstacles is not None:
            moved = None if self._prev_pos is None else np.linalg.norm(seeker_pos - self._prev_pos)
            self._prev_pos = np.array(seeker_pos, dtype=np.float64)
            if self.mode in ("TRACK", "SEARCH") and moved is not None:
                self._stuck = self._stuck + 1 if moved < 0.3 * self.speed * DT else 0
                if self._stuck >= 15 and not visible:
                    self._stuck = 0
                    self._start_scan()
                    return np.zeros(3, dtype=np.float64)
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
            for _ in range(50):
                if self.obstacles is None or not in_box(self.waypoint, self.obstacles, DRONE_RADIUS):
                    break
                self.waypoint = self.rng.uniform(low, high)
            self.mode = "SEARCH"
            self.searches += 1

        return np.zeros(3, dtype=np.float64)


def as_vec(value, size, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return value.copy()


def rand_start(rng, randomize_start=True, fov_deg=FOV_DEG,
                         detection_range=DETECTION_RANGE, obstacles=None):
    boxes = make_boxes(obstacles)
    if randomize_start:
        for _ in range(200 if len(boxes) == 0 else 2000):
            seeker = rng.uniform([3.0, 3.0, 2.0], [17.0, 17.0, 8.0])
            forward = normalize(rng.normal(size=3))
            if np.linalg.norm(forward) < 1e-8:
                continue
            direction = normalize(forward + rng.normal(0.0, 0.18, size=3))
            evader = seeker + direction * rng.uniform(4.0, 8.0)
            if inside_valid_arena(evader, margin=1.0) and can_see(
                    seeker, forward, evader, fov_deg, detection_range)[0]:
                if len(boxes) and (in_box(seeker, boxes, 1.0) or in_box(evader, boxes, 1.0)
                                   or ray_hits(seeker, evader, boxes)):
                    continue
                return seeker, evader, forward
    if len(boxes):
        return (np.array([4.0, 10.0, 8.0]), np.array([12.0, 10.0, 8.0]),
                np.array([1.0, 0.0, 0.0]))
    return (np.array([4.0, 10.0, 5.0]), np.array([12.0, 10.0, 5.0]),
            np.array([1.0, 0.0, 0.0]))


def wall_dist(position):
    return np.array([position[0] / ARENA_X, (ARENA_X - position[0]) / ARENA_X,
                     position[1] / ARENA_Y, (ARENA_Y - position[1]) / ARENA_Y,
                     position[2] / ARENA_Z, (ARENA_Z - position[2]) / ARENA_Z],
                    dtype=np.float32)


class Simulator:

    def __init__(self, episode_steps=EPISODE_STEPS, randomize_start=True,
                 yaw_rate_deg=100.0, pitch_rate_deg=90.0, pitch_limit_deg=89.0,
                 boundary_penalty_per_step=0.0, obstacles=None,
                 seeker_history=1, seeker_history_stride=5,
                 seeker_boundary_penalty_per_step=None,
                 include_last_known_velocity=False,
                 seeker_visible_bonus=None, evader_hidden_bonus=None,
                 seeker_proximity_coef=0.0):
        if not isinstance(episode_steps, int) or episode_steps <= 0:
            raise ValueError("episode_steps must be a positive integer")
        if not np.all(np.isfinite([yaw_rate_deg, pitch_rate_deg, pitch_limit_deg])):
            raise ValueError("camera parameters must be finite")
        if min(yaw_rate_deg, pitch_rate_deg) <= 0 or not 0 < pitch_limit_deg <= 90:
            raise ValueError("turn rates must be positive and pitch limit in (0, 90]")
        if not np.isfinite(boundary_penalty_per_step) or boundary_penalty_per_step < 0:
            raise ValueError("boundary_penalty_per_step must be finite and non-negative")
        self.boundary_penalty_per_step = boundary_penalty_per_step
        if seeker_boundary_penalty_per_step is None:
            self.seeker_boundary_penalty_per_step = boundary_penalty_per_step
        else:
            if (not np.isfinite(seeker_boundary_penalty_per_step)
                    or seeker_boundary_penalty_per_step < 0):
                raise ValueError("seeker_boundary_penalty_per_step must be finite and non-negative")
            self.seeker_boundary_penalty_per_step = seeker_boundary_penalty_per_step
        self.include_last_known_velocity = bool(include_last_known_velocity)
        if seeker_visible_bonus is not None and (not np.isfinite(seeker_visible_bonus) or seeker_visible_bonus < 0):
            raise ValueError("seeker_visible_bonus must be finite and non-negative")
        if evader_hidden_bonus is not None and (not np.isfinite(evader_hidden_bonus) or evader_hidden_bonus < 0):
            raise ValueError("evader_hidden_bonus must be finite and non-negative")
        if not np.isfinite(seeker_proximity_coef) or seeker_proximity_coef < 0:
            raise ValueError("seeker_proximity_coef must be finite and non-negative")
        self.seeker_visible_bonus = seeker_visible_bonus
        self.evader_hidden_bonus = evader_hidden_bonus
        self.seeker_proximity_coef = float(seeker_proximity_coef)
        if type(seeker_history) is not int or seeker_history < 1 or type(seeker_history_stride) is not int or seeker_history_stride < 1:
            raise ValueError("seeker_history and seeker_history_stride must be positive integers")
        self.seeker_history, self.seeker_history_stride = seeker_history, seeker_history_stride
        self._seeker_frames = deque(maxlen=(seeker_history - 1) * seeker_history_stride + 1)
        self.obstacles = make_boxes(obstacles)
        self.obs_extra = NEAR_K * 3 if len(self.obstacles) else 0
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
            seeker, evader, forward = rand_start(
                rng, self.randomize_start, self.fov_deg, self.detection_range, self.obstacles)
        else:
            if set(initial_state) != {"seeker_pos", "evader_pos", "seeker_forward"}:
                raise ValueError("initial_state requires seeker_pos, evader_pos, seeker_forward")
            seeker = as_vec(initial_state["seeker_pos"], 3, "seeker_pos")
            evader = as_vec(initial_state["evader_pos"], 3, "evader_pos")
            forward = as_vec(initial_state["seeker_forward"], 3, "seeker_forward")
            if any(not np.array_equal(pos, clamp_position(pos)) for pos in (seeker, evader)):
                raise ValueError("initial positions must be inside the arena boundary margin")
            if np.linalg.norm(forward) < 1e-8:
                raise ValueError("seeker_forward cannot be zero")
            if any(in_box(pos, self.obstacles, DRONE_RADIUS) for pos in (seeker, evader)):
                raise ValueError("initial positions must be outside every obstacle")
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
        self.first_detection_step = None
        self.drone_collision_steps = self.drone_collision_events = 0
        self._drones_overlapping = False
        self.reacquisition_steps = []
        self.loss_events = 0
        self.last_known_position = None
        self.last_known_velocity = np.zeros(3)
        self.time_since_detection_steps = 0
        self.visible, self.distance, self.angle = self.check_visibility()
        self.initially_visible = self.visible
        self.first_detection_step = 0 if self.visible else None
        self._update_detection_memory()
        self._seeker_frames.clear()
        self._push_seeker_frame()
        return self.visible, self.distance, self.angle

    def set_camera_forward(self, forward):
        self.camera_forward = normalize(as_vec(forward, 3, "seeker_forward"))
        if np.linalg.norm(self.camera_forward) < 1e-8:
            raise ValueError("seeker_forward cannot be zero")
        self.camera_yaw = math.atan2(self.camera_forward[1], self.camera_forward[0])
        self.camera_pitch = math.asin(float(np.clip(self.camera_forward[2], -1.0, 1.0)))

    def rotate_camera(self, controls):
        controls = np.clip(as_vec(controls, 2, "camera action"), -1.0, 1.0)
        self.camera_yaw = (self.camera_yaw + controls[0] * self.yaw_rate * DT
                           + math.pi) % (2 * math.pi) - math.pi
        self.camera_pitch = float(np.clip(
            self.camera_pitch + controls[1] * self.pitch_rate * DT,
            -self.pitch_limit, self.pitch_limit))
        yaw, pitch = self.camera_yaw, self.camera_pitch
        self.camera_forward = np.array([math.cos(pitch) * math.cos(yaw),
                                       math.cos(pitch) * math.sin(yaw), math.sin(pitch)])

    def check_visibility(self):
        visible, distance, angle = can_see(self.seeker_pos, self.camera_forward, self.evader_pos,
                                              self.fov_deg, self.detection_range)
        if visible and ray_hits(self.seeker_pos, self.evader_pos, self.obstacles):
            visible = False
        return visible, distance, angle

    def _update_detection_memory(self):
        if self.visible:
            self.last_known_position = self.evader_pos.copy()
            self.last_known_velocity = self.evader_velocity.copy()
            self.time_since_detection_steps = 0

    def advance(self, seeker_velocity, evader_velocity, *, camera_action=None,
                seeker_forward=None):
        if self.seeker_pos is None:
            raise RuntimeError("reset the simulator before stepping")
        if self.step_count >= self.episode_steps:
            raise RuntimeError("episode is complete; call reset")
        seeker_velocity = as_vec(seeker_velocity, 3, "seeker velocity")
        evader_velocity = as_vec(evader_velocity, 3, "evader velocity")
        if camera_action is not None and seeker_forward is not None:
            raise ValueError("provide camera_action or seeker_forward, not both")
        if camera_action is not None:
            as_vec(camera_action, 2, "camera action")
        if seeker_forward is not None:
            as_vec(seeker_forward, 3, "seeker_forward")
            if np.linalg.norm(seeker_forward) < 1e-8:
                raise ValueError("seeker_forward cannot be zero")
        if camera_action is not None:
            self.rotate_camera(camera_action)
        elif seeker_forward is not None:
            self.set_camera_forward(seeker_forward)
            self.camera_forward = np.asarray(seeker_forward, dtype=np.float64).copy()
        self.seeker_velocity, self.evader_velocity = seeker_velocity, evader_velocity
        proposed = {"seeker": self.seeker_pos + seeker_velocity * DT,
                    "evader": self.evader_pos + evader_velocity * DT}
        obstacle_hit = {"seeker": False, "evader": False}
        if len(self.obstacles):
            for name, old in (("seeker", self.seeker_pos), ("evader", self.evader_pos)):
                proposed[name], obstacle_hit[name] = slide(
                    old, proposed[name], self.obstacles, DRONE_RADIUS)
        contacts = {a: hit_boundary(pos) or obstacle_hit[a] for a, pos in proposed.items()}
        fresh = {"seeker": contacts["seeker"] and not self.seeker_was_on_boundary,
                 "evader": contacts["evader"] and not self.evader_was_on_boundary}
        self.seeker_pos = clamp_position(proposed["seeker"])
        self.evader_pos = clamp_position(proposed["evader"])
        was_visible = self.visible
        self.visible, self.distance, self.angle = self.check_visibility()
        if self.visible:
            evader_reward = REWARD_VISIBLE
            seeker_reward = (self.seeker_visible_bonus
                             if self.seeker_visible_bonus is not None else -REWARD_VISIBLE)
        else:
            evader_reward = (self.evader_hidden_bonus
                             if self.evader_hidden_bonus is not None else REWARD_HIDDEN)
            seeker_reward = -REWARD_HIDDEN
        if self.seeker_proximity_coef:
            dist = float(np.linalg.norm(self.seeker_pos - self.evader_pos))
            seeker_reward += self.seeker_proximity_coef / (1.0 + dist)
        if fresh["evader"]:
            evader_reward += REWARD_BOUNDARY
            self.wall_touches += 1
        if fresh["seeker"]:
            seeker_reward += REWARD_BOUNDARY
            self.seeker_wall_touches += 1
        if self.boundary_penalty_per_step:
            if contacts["evader"] and not fresh["evader"]:
                evader_reward -= self.boundary_penalty_per_step
        if self.seeker_boundary_penalty_per_step:
            if contacts["seeker"] and not fresh["seeker"]:
                seeker_reward -= self.seeker_boundary_penalty_per_step
        self.evader_was_on_boundary = contacts["evader"]
        self.seeker_was_on_boundary = contacts["seeker"]
        for agent in contacts:
            self.boundary_steps[agent] += int(contacts[agent])
        self.step_count += 1
        if self.visible and self.first_detection_step is None:
            self.first_detection_step = self.step_count
        overlapping = bool(np.linalg.norm(self.seeker_pos - self.evader_pos) < 2 * DRONE_RADIUS)
        self.drone_collision_steps += int(overlapping)
        self.drone_collision_events += int(overlapping and not self._drones_overlapping)
        self._drones_overlapping = overlapping
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
        self._push_seeker_frame()
        return ({"seeker": float(seeker_reward), "evader": float(evader_reward)},
                fresh, self.step_count >= self.episode_steps)

    def evader_observation(self, include_visibility=False):
        center = ARENA_SIZE / 2.0
        parts = [(self.evader_pos.astype(np.float32) - center) / center,
                 (self.evader_velocity / EVADER_SPEED).astype(np.float32),
                 ((self.seeker_pos - self.evader_pos) / ARENA_SIZE).astype(np.float32),
                 ((self.seeker_velocity - self.evader_velocity)
                  / (SEEKER_SPEED + EVADER_SPEED)).astype(np.float32),
                 self.camera_forward.astype(np.float32), wall_dist(self.evader_pos)]
        if include_visibility:
            parts.append(np.array([float(self.visible)], dtype=np.float32))
        if self.obs_extra:
            parts.append(close_obs(self.evader_pos, self.obstacles))
        return self._bounded_observation(parts)

    def _push_seeker_frame(self):
        if self.seeker_history > 1:
            self._seeker_frames.append(self._seeker_base_observation())

    def seeker_observation(self):
        if self.seeker_history == 1:
            return self._seeker_base_observation()
        frames = self._seeker_frames
        stacked = [frames[max(len(frames) - 1 - j * self.seeker_history_stride, 0)]
                   for j in range(self.seeker_history)]
        return np.concatenate(stacked).astype(np.float32)

    def _seeker_base_observation(self):
        known = self.last_known_position is not None
        relative = ((self.last_known_position - self.seeker_pos) / ARENA_SIZE
                    if known else np.zeros(3))
        center = ARENA_SIZE / 2.0
        parts = [
            (self.seeker_pos - center) / center, self.seeker_velocity / SEEKER_SPEED,
            self.camera_forward, wall_dist(self.seeker_pos),
            [float(self.visible)], relative, [float(known)],
            [min(self.time_since_detection_steps / self.episode_steps, 1.0)]]
        if self.include_last_known_velocity:
            last_vel = (self.last_known_velocity / EVADER_SPEED
                        if known else np.zeros(3))
            parts.append(last_vel)
        if self.obs_extra:
            parts.append(close_obs(self.seeker_pos, self.obstacles))
        return self._bounded_observation(parts)

    @property
    def seeker_base_obs_dim(self):
        return 21 + (3 if self.include_last_known_velocity else 0) + self.obs_extra

    @staticmethod
    def _bounded_observation(parts):
        obs = np.concatenate(parts).astype(np.float32)
        if not np.all(np.isfinite(obs)):
            raise ValueError("non-finite world state produced an observation")
        return np.clip(obs, -1.0, 1.0)

    def global_state(self):
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
            "first_detection_time_s": (None if self.first_detection_step is None
                                       else self.first_detection_step * DT),
            "drone_collision_steps": self.drone_collision_steps,
            "drone_collision_events": self.drone_collision_events,
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
