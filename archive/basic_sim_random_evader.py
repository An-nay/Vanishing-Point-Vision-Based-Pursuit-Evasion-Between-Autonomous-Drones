"""
Vanishing Point - Phase 1 Basic 3D Simulation

Stage 1:
- One scripted Seeker (track, 3D scan, random-waypoint search)
- One randomly moving Evader
- Bounded 3D arena
- Geometric field-of-view (FOV) detection
- Visibility-based reward

Later, replace RandomEvaderPolicy with a PPO policy.
"""

import math
import time
import numpy as np
import pybullet as p
import pybullet_data


# -----------------------------
# Configuration
# -----------------------------
ARENA_X = 20.0
ARENA_Y = 20.0
ARENA_Z = 10.0

DT = 0.1
EPISODE_STEPS = 600  # 60 seconds at dt=0.1

SEEKER_SPEED = 1.8
EVADER_SPEED = 2.2

FOV_DEG = 90.0
DETECTION_RANGE = 10.0

DRONE_RADIUS = 0.35
BOUNDARY_MARGIN = 0.4

REWARD_HIDDEN = 1.0
REWARD_VISIBLE = -1.0
# One-time penalty applied when the Evader first touches a wall.
# Kept small so visibility (+1 / -1 per step) stays the main learning signal.
REWARD_BOUNDARY = -2.0

# Seeker search behaviour
ARRIVE_DIST = 0.5          # m, how close counts as "arrived" at a target
SCAN_TIMEOUT = 4.0         # s of scanning before giving up and moving on
SCAN_YAW_RATE = 100.0      # deg/s, spin speed while scanning (unchanged from before)
SCAN_PITCH_MAX = 45.0      # deg, scan tilts up and down by this much
SCAN_PITCH_PERIOD = 2.0    # s, time for one full up-down tilt cycle
WAYPOINT_MARGIN = 1.0      # m, random search points stay this far from walls

# Set to an int to reproduce a run exactly. None picks a fresh seed every run.
SEED = None


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
    """Keep a drone inside the arena."""
    x = np.clip(pos[0], BOUNDARY_MARGIN, ARENA_X - BOUNDARY_MARGIN)
    y = np.clip(pos[1], BOUNDARY_MARGIN, ARENA_Y - BOUNDARY_MARGIN)
    z = np.clip(pos[2], BOUNDARY_MARGIN, ARENA_Z - BOUNDARY_MARGIN)
    return np.array([x, y, z], dtype=np.float64)


def hit_boundary(pos):
    return (
        pos[0] <= BOUNDARY_MARGIN
        or pos[0] >= ARENA_X - BOUNDARY_MARGIN
        or pos[1] <= BOUNDARY_MARGIN
        or pos[1] >= ARENA_Y - BOUNDARY_MARGIN
        or pos[2] <= BOUNDARY_MARGIN
        or pos[2] >= ARENA_Z - BOUNDARY_MARGIN
    )


def is_visible(seeker_pos, seeker_forward, evader_pos):
    """
    Geometric FOV test in 3D.

    Visible when:
    1. Evader is within detection range
    2. Angle between seeker-facing vector and seeker->evader vector
       is within half the camera FOV
    """
    to_evader = np.asarray(evader_pos) - np.asarray(seeker_pos)
    distance = np.linalg.norm(to_evader)

    if distance < 1e-8 or distance > DETECTION_RANGE:
        return False, distance, 180.0

    direction_to_evader = to_evader / distance
    seeker_forward = normalize(seeker_forward)

    cos_theta = np.clip(
        np.dot(seeker_forward, direction_to_evader),
        -1.0,
        1.0,
    )

    angle_deg = math.degrees(math.acos(cos_theta))
    visible = angle_deg <= (FOV_DEG / 2.0)

    return visible, distance, angle_deg


# -----------------------------
# Drone creation / rendering
# -----------------------------
def create_drone(position, rgba):
    collision = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=DRONE_RADIUS,
    )

    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=DRONE_RADIUS,
        rgbaColor=rgba,
    )

    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

    return body_id


def create_arena():
    # Floor
    floor_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[ARENA_X / 2, ARENA_Y / 2, 0.05],
    )

    floor_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[ARENA_X / 2, ARENA_Y / 2, 0.05],
        rgbaColor=[0.75, 0.75, 0.75, 1.0],
    )

    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=floor_shape,
        baseVisualShapeIndex=floor_visual,
        basePosition=[ARENA_X / 2, ARENA_Y / 2, -0.05],
    )

    # Draw upper arena boundaries as debug lines
    corners = [
        [0, 0, 0],
        [ARENA_X, 0, 0],
        [ARENA_X, ARENA_Y, 0],
        [0, ARENA_Y, 0],
        [0, 0, ARENA_Z],
        [ARENA_X, 0, ARENA_Z],
        [ARENA_X, ARENA_Y, ARENA_Z],
        [0, ARENA_Y, ARENA_Z],
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for a, b in edges:
        p.addUserDebugLine(
            corners[a],
            corners[b],
            [0.2, 0.2, 0.2],
            lineWidth=1.5,
            lifeTime=0,
        )


# -----------------------------
# Policies
# -----------------------------
class RandomEvaderPolicy:
    """
    Simple baseline policy.
    Randomly changes target direction every few simulation steps.
    """

    def __init__(self, change_every=12):
        self.change_every = change_every
        self.counter = 0
        self.direction = normalize(np.random.uniform(-1, 1, size=3))

    def act(self):
        if self.counter % self.change_every == 0:
            self.direction = normalize(np.random.uniform(-1, 1, size=3))

        self.counter += 1
        return self.direction * EVADER_SPEED


class ScriptedSeekerPolicy:
    """
    Rule-based seeker with three modes:

    TRACK  - Evader visible (or just lost): fly toward its last-known position.
    SCAN   - Arrived and nothing in view: spin AND tilt up/down so the camera
             cone sweeps the full 3D space. Gives up after SCAN_TIMEOUT.
    SEARCH - Scan failed: fly to a random point in the arena, then scan again.

    Seeing the Evader at any time switches straight back to TRACK.
    """

    def __init__(self, initial_forward=np.array([1.0, 0.0, 0.0])):
        self.forward = normalize(initial_forward)
        self.last_known_position = None
        self.waypoint = None
        self.searches = 0  # how many times a scan failed and a new search began
        self._start_scan()

    def _start_scan(self):
        self.mode = "SCAN"
        self.scan_time = 0.0
        self.yaw = math.atan2(self.forward[1], self.forward[0])

    def _set_forward_from_angles(self, yaw, pitch):
        self.forward = normalize(
            np.array([
                math.cos(pitch) * math.cos(yaw),
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
            ])
        )

    def _fly_toward(self, seeker_pos, target):
        """Face the target and fly at it. Returns None if already there."""
        vec = target - seeker_pos
        if np.linalg.norm(vec) < ARRIVE_DIST:
            return None
        self.forward = normalize(vec)
        return self.forward * SEEKER_SPEED

    def act(self, seeker_pos, evader_pos, visible):
        if visible:
            self.last_known_position = np.array(evader_pos, dtype=np.float64)
            self.mode = "TRACK"

        if self.mode == "TRACK":
            velocity = self._fly_toward(seeker_pos, self.last_known_position)
            if velocity is not None:
                return velocity
            self._start_scan()  # arrived and evader not in view
            return np.zeros(3)

        if self.mode == "SEARCH":
            velocity = self._fly_toward(seeker_pos, self.waypoint)
            if velocity is not None:
                return velocity
            self._start_scan()  # reached the waypoint, scan again
            return np.zeros(3)

        # SCAN: spin around vertical axis while tilting up and down
        self.scan_time += DT
        self.yaw += math.radians(SCAN_YAW_RATE) * DT
        pitch = math.radians(SCAN_PITCH_MAX) * math.sin(
            2.0 * math.pi * self.scan_time / SCAN_PITCH_PERIOD
        )
        self._set_forward_from_angles(self.yaw, pitch)

        if self.scan_time >= SCAN_TIMEOUT:
            low = np.array([WAYPOINT_MARGIN] * 3)
            high = np.array([
                ARENA_X - WAYPOINT_MARGIN,
                ARENA_Y - WAYPOINT_MARGIN,
                ARENA_Z - WAYPOINT_MARGIN,
            ])
            self.waypoint = np.random.uniform(low, high)
            self.mode = "SEARCH"
            self.searches += 1

        return np.zeros(3)


# -----------------------------
# Main simulation
# -----------------------------
def main():
    seed = SEED if SEED is not None else int(time.time() * 1000) % (2**32)
    np.random.seed(seed)

    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.setTimeStep(DT)

    p.resetDebugVisualizerCamera(
        cameraDistance=28,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2],
    )

    create_arena()

    seeker_pos = np.array([4.0, 10.0, 5.0], dtype=np.float64)
    evader_pos = np.array([12.0, 10.0, 5.0], dtype=np.float64)

    seeker_id = create_drone(
        seeker_pos,
        rgba=[1.0, 0.2, 0.2, 1.0],   # red
    )

    evader_id = create_drone(
        evader_pos,
        rgba=[0.2, 0.4, 1.0, 1.0],   # blue
    )

    seeker_policy = ScriptedSeekerPolicy()
    evader_policy = RandomEvaderPolicy()

    total_reward = 0.0
    undetected_steps = 0
    evader_was_on_boundary = False
    wall_touches = 0
    first_loss_step = None      # first step the Evader is out of sight
    current_hidden_streak = 0   # consecutive hidden steps right now
    longest_hidden_streak = 0   # longest run of consecutive hidden steps

    print("\nVanishing Point - Basic 3D Simulation")
    print("Red = Seeker | Blue = Evader")
    print(f"Seed: {seed}\n")

    for step in range(EPISODE_STEPS):

        # 1. Check visibility using current state
        visible, distance, angle = is_visible(
            seeker_pos,
            seeker_policy.forward,
            evader_pos,
        )

        # 2. Select actions
        evader_velocity = evader_policy.act()

        seeker_velocity = seeker_policy.act(
            seeker_pos,
            evader_pos,
            visible,
        )

        # 3. Transition dynamics
        proposed_evader_pos = evader_pos + evader_velocity * DT
        proposed_seeker_pos = seeker_pos + seeker_velocity * DT

        evader_boundary = hit_boundary(proposed_evader_pos)
        seeker_boundary = hit_boundary(proposed_seeker_pos)

        evader_pos = clamp_position(proposed_evader_pos)
        seeker_pos = clamp_position(proposed_seeker_pos)

        # 4. Recalculate visibility after movement
        visible, distance, angle = is_visible(
            seeker_pos,
            seeker_policy.forward,
            evader_pos,
        )

        # 5. Reward
        reward = REWARD_VISIBLE if visible else REWARD_HIDDEN

        # Penalize only the moment of contact, not every step spent against the wall
        if evader_boundary and not evader_was_on_boundary:
            reward += REWARD_BOUNDARY
            wall_touches += 1
        evader_was_on_boundary = evader_boundary

        total_reward += reward

        if not visible:
            undetected_steps += 1
            current_hidden_streak += 1
            longest_hidden_streak = max(longest_hidden_streak, current_hidden_streak)
            if first_loss_step is None:
                first_loss_step = step
        else:
            current_hidden_streak = 0

        # 6. Update PyBullet positions
        p.resetBasePositionAndOrientation(
            seeker_id,
            seeker_pos.tolist(),
            [0, 0, 0, 1],
        )

        p.resetBasePositionAndOrientation(
            evader_id,
            evader_pos.tolist(),
            [0, 0, 0, 1],
        )

        # 7. Draw seeker facing/FOV reference lines
        forward_end = seeker_pos + seeker_policy.forward * DETECTION_RANGE

        line_colour = [1, 0, 0] if visible else [0, 1, 0]

        p.addUserDebugLine(
            seeker_pos.tolist(),
            forward_end.tolist(),
            line_colour,
            lineWidth=3,
            lifeTime=DT,
        )

        # Display current status in GUI
        p.addUserDebugText(
            f"Step: {step} | Mode: {seeker_policy.mode} | Visible: {visible} | Reward: {reward:.1f}",
            [0.5, 0.5, ARENA_Z + 0.7],
            textColorRGB=[0, 0, 0],
            textSize=1.2,
            lifeTime=DT,
        )

        if step % 20 == 0:
            print(
                f"Step={step:03d} | "
                f"Mode={seeker_policy.mode:6s} | "
                f"Visible={str(visible):5s} | "
                f"Distance={distance:5.2f} m | "
                f"Angle={angle:5.1f} deg | "
                f"Reward={reward:5.1f}"
            )

        p.stepSimulation()
        time.sleep(DT)

    undetected_ratio = undetected_steps / EPISODE_STEPS

    if first_loss_step is None:
        first_loss_time = "Never lost sight"
    else:
        first_loss_time = f"{first_loss_step * DT:.1f} s"

    print("\n--- Episode Summary ---")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Undetected time: {100 * undetected_ratio:.1f}%")
    print(f"Time until first loss of sight: {first_loss_time}")
    print(f"Longest undetected stretch: {longest_hidden_streak * DT:.1f} s")
    print(f"Evader wall touches: {wall_touches}")
    print(f"Seeker search restarts: {seeker_policy.searches}")
    print(f"Seed: {seed}")

    input("\nPress Enter to close...")
    p.disconnect(physics_client)


if __name__ == "__main__":
    main()
