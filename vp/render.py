"""Lazy PyBullet visualization shared by both RL interfaces."""
import math
import numpy as np
from vp.core import ARENA_X, ARENA_Y, ARENA_Z, DT

SEEKER_COLOUR = [0.90, 0.15, 0.15, 1.0]
EVADER_COLOUR = [0.15, 0.45, 0.95, 1.0]
OBSTACLE_COLOURS = [[0.55, 0.50, 0.45, 1.0], [0.45, 0.48, 0.55, 1.0], [0.60, 0.55, 0.40, 1.0]]
DRONE_SCALE = 1.8  # drawn larger than the collision radius so they read as drones

class RendererMixin:
    def _init_renderer(self):
        global p, pybullet_data
        import pybullet as p
        import pybullet_data
        # Tests/screenshots can set render_backend = p.DIRECT to render without a window.
        self.physics_client = p.connect(getattr(self, 'render_backend', p.GUI))
        if self.physics_client < 0:
            raise RuntimeError('PyBullet could not connect to the renderer')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)
        p.setTimeStep(DT, physicsClientId=self.physics_client)
        p.resetDebugVisualizerCamera(
            cameraDistance=28,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2],
            physicsClientId=self.physics_client,
        )
        self._trail_prev = {}
        self._heading = {'evader': 0.0}
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
        for tick in range(2, int(ARENA_X), 2):  # 2 m floor grid for a sense of scale
            p.addUserDebugLine([tick, 0, 0.01], [tick, ARENA_Y, 0.01], [0.6, 0.6, 0.6],
                               lineWidth=1, lifeTime=0, physicsClientId=self.physics_client)
            p.addUserDebugLine([0, tick, 0.01], [ARENA_X, tick, 0.01], [0.6, 0.6, 0.6],
                               lineWidth=1, lifeTime=0, physicsClientId=self.physics_client)
        self._create_obstacles()

    def _create_obstacles(self):
        for index, (low, high) in enumerate(self.obstacles):
            half, centre = ((high - low) / 2).tolist(), ((high + low) / 2).tolist()
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                         rgbaColor=OBSTACLE_COLOURS[index % len(OBSTACLE_COLOURS)],
                                         physicsClientId=self.physics_client)
            p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=centre,
                              physicsClientId=self.physics_client)

    def _create_drone(self, position, rgba):
        """Quadcopter: body, two crossed arms, four rotor discs and a nose marker (+x)."""
        k = DRONE_SCALE
        arm = 0.42 * k
        dark = [0.12, 0.12, 0.12, 1.0]
        shapes, extents, radii, lengths, offsets, orients, colours = [], [], [], [], [], [], []

        def add(kind, half=(0, 0, 0), radius=0, length=0, offset=(0, 0, 0), yaw=0.0, colour=dark):
            shapes.append(kind)
            extents.append(list(half))
            radii.append(radius)
            lengths.append(length)
            offsets.append(list(offset))
            orients.append(p.getQuaternionFromEuler([0, 0, yaw]))
            colours.append(colour)

        add(p.GEOM_BOX, half=(0.17 * k, 0.17 * k, 0.06 * k), colour=rgba)
        for yaw in (math.pi / 4, -math.pi / 4):
            add(p.GEOM_BOX, half=(arm, 0.025 * k, 0.015 * k), yaw=yaw)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            reach = arm / math.sqrt(2)
            add(p.GEOM_CYLINDER, radius=0.22 * k, length=0.015 * k,
                offset=(sx * reach, sy * reach, 0.05 * k), colour=[rgba[0], rgba[1], rgba[2], 0.55])
        add(p.GEOM_SPHERE, radius=0.09 * k, offset=(0.19 * k, 0, 0), colour=[1, 0.9, 0.1, 1.0])
        visual = p.createVisualShapeArray(
            shapeTypes=shapes, halfExtents=extents, radii=radii, lengths=lengths,
            visualFramePositions=offsets, visualFrameOrientations=orients, rgbaColors=colours,
            physicsClientId=self.physics_client)
        return p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=position,
                                 physicsClientId=self.physics_client)

    def _reset_render_objects(self):
        if self.seeker_id is None:
            self.seeker_id = self._create_drone(self.seeker_pos.tolist(), SEEKER_COLOUR)
        if self.evader_id is None:
            self.evader_id = self._create_drone(self.evader_pos.tolist(), EVADER_COLOUR)
        self._trail_prev = {}
        self._place_drones()

    def _place_drones(self):
        """Seeker faces its camera; the Evader faces its direction of travel."""
        forward = np.asarray(self.camera_forward, dtype=float)
        seeker_yaw = math.atan2(forward[1], forward[0])
        seeker_pitch = -math.asin(float(np.clip(forward[2], -1, 1)))  # +pitch tilts the nose down in PyBullet
        velocity = np.asarray(self.evader_velocity, dtype=float)
        if np.linalg.norm(velocity[:2]) > 0.1:
            self._heading['evader'] = math.atan2(velocity[1], velocity[0])
        for body, pos, orient in (
                (self.seeker_id, self.seeker_pos, p.getQuaternionFromEuler([0, seeker_pitch, seeker_yaw])),
                (self.evader_id, self.evader_pos, p.getQuaternionFromEuler([0, 0, self._heading['evader']]))):
            p.resetBasePositionAndOrientation(body, pos.tolist(), orient,
                                              physicsClientId=self.physics_client)

    def _draw_view_cone(self, visible):
        """Seeker field of view as a wireframe cone clipped at the detection range."""
        forward = np.asarray(self.camera_forward, dtype=float)
        helper = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, helper)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        radius = self.detection_range * math.tan(math.radians(self.fov_deg / 2))
        colour = [1.0, 0.25, 0.25] if visible else [1.0, 0.8, 0.2]
        origin = np.asarray(self.seeker_pos, dtype=float)
        centre = origin + forward * self.detection_range
        ring = [centre + radius * (math.cos(a) * right + math.sin(a) * up)
                for a in np.linspace(0, 2 * math.pi, 17)]
        for index, point in enumerate(ring[:-1]):
            if index % 2 == 0:
                p.addUserDebugLine(origin.tolist(), point.tolist(), colour, lineWidth=1,
                                   lifeTime=DT, physicsClientId=self.physics_client)
            p.addUserDebugLine(point.tolist(), ring[index + 1].tolist(), colour, lineWidth=1,
                               lifeTime=DT, physicsClientId=self.physics_client)

    def _draw_trails(self):
        for name, pos, colour in (("seeker", self.seeker_pos, SEEKER_COLOUR[:3]),
                                  ("evader", self.evader_pos, EVADER_COLOUR[:3])):
            previous = self._trail_prev.get(name)
            if previous is not None:
                p.addUserDebugLine(previous, pos.tolist(), colour, lineWidth=2, lifeTime=4.0,
                                   physicsClientId=self.physics_client)
            self._trail_prev[name] = pos.tolist()

    def _render_step(self, visible, distance, angle, reward):
        if self.seeker_id is None or self.evader_id is None:
            self._reset_render_objects()
        self._place_drones()
        self._draw_view_cone(visible)
        self._draw_trails()

        # Sight line: green = seen, orange = in range and in FOV but hidden behind an obstacle.
        blocked = (not visible) and distance <= self.detection_range and angle <= self.fov_deg / 2
        if visible or blocked:
            p.addUserDebugLine(self.seeker_pos.tolist(), self.evader_pos.tolist(),
                               [0.1, 0.9, 0.1] if visible else [1.0, 0.5, 0.0],
                               lineWidth=2 if visible else 3, lifeTime=DT,
                               physicsClientId=self.physics_client)
        status = "VISIBLE" if visible else ("BLOCKED by obstacle" if blocked else "hidden")
        p.addUserDebugText(
            (
                f"Step: {self.step_count} | Mode: {getattr(self, 'seeker_mode', getattr(getattr(self, 'seeker_policy', None), 'mode', 'LEARNED'))} | "
                f"{status} | Reward: {reward:.1f} | "
                f"Dist: {distance:.1f}m | Angle: {angle:.1f}deg"
            ),
            [0.5, 0.5, ARENA_Z + 0.7],
            textColorRGB=[0, 0, 0],
            textSize=1.1,
            lifeTime=DT,
            physicsClientId=self.physics_client,
        )
        p.stepSimulation(physicsClientId=self.physics_client)

    def snapshot(self, path, width=960, height=720):
        """Save the current scene to a PNG (works with render_backend = p.DIRECT)."""
        import matplotlib.image as mpimg
        view = p.computeViewMatrixFromYawPitchRoll([ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 3], 30, 45, -28, 0, 2,
                                                   physicsClientId=self.physics_client)
        proj = p.computeProjectionMatrixFOV(60, width / height, 0.1, 100, physicsClientId=self.physics_client)
        image = p.getCameraImage(width, height, view, proj, physicsClientId=self.physics_client)[2]
        mpimg.imsave(path, np.reshape(image, (height, width, 4))[:, :, :3].astype(np.uint8))

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except p.error:
                pass
            self.physics_client = None
