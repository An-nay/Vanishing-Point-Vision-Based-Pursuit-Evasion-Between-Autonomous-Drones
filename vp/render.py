import math
import numpy as np
from vp.core import ARENA_X, ARENA_Y, ARENA_Z, DT, LAMPS, TREES

SEEKER_COLOUR = [0.90, 0.15, 0.15, 1.0]
EVADER_COLOUR = [0.15, 0.45, 0.95, 1.0]
DRONE_SCALE = 1.8
CAMERA_VIEWS = (
    (26, 40, -62),
    (26, 130, -58),
    (28, 220, -64),
    (26, 310, -56),
)

class Drawer:
    def _init_renderer(self):
        global p, pybullet_data
        import pybullet as p
        import pybullet_data
        self.physics_client = p.connect(getattr(self, 'render_backend', p.GUI))
        if self.physics_client < 0:
            raise RuntimeError('PyBullet could not connect to the renderer')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, 0, physicsClientId=self.physics_client)
        p.setTimeStep(DT, physicsClientId=self.physics_client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        try:
            p.configureDebugVisualizer(rgbBackground=[0.0, 0.0, 0.02], physicsClientId=self.physics_client)
        except TypeError:
            pass
        self._camera_frame = 0
        self._aim_camera()
        self._trail_prev = {}
        self._heading = {'evader': 0.0}
        self._create_arena()

    def _aim_camera(self):
        distance, yaw, pitch = CAMERA_VIEWS[(self._camera_frame // 50) % len(CAMERA_VIEWS)]
        self._camera_frame += 1
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[ARENA_X / 2, ARENA_Y / 2, 3.0],
            physicsClientId=self.physics_client,
        )

    def _create_arena(self):
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[ARENA_X / 2, ARENA_Y / 2, 0.05],
            physicsClientId=self.physics_client,
        )
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[ARENA_X / 2, ARENA_Y / 2, 0.05],
            rgbaColor=[0.02, 0.02, 0.06, 1.0],
            physicsClientId=self.physics_client,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=floor_shape,
            baseVisualShapeIndex=floor_visual,
            basePosition=[ARENA_X / 2, ARENA_Y / 2, -0.05],
            physicsClientId=self.physics_client,
        )

        floor = [
            [0, 0, 0.02], [ARENA_X, 0, 0.02], [ARENA_X, ARENA_Y, 0.02], [0, ARENA_Y, 0.02],
        ]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            p.addUserDebugLine(floor[a], floor[b], [0.25, 0.28, 0.85], lineWidth=2,
                               lifeTime=0, physicsClientId=self.physics_client)
        self._create_streets()
        self._create_details()
        self._create_pellets()
        self._create_obstacles()

    def _create_details(self):
        client = self.physics_client
        for x, y in LAMPS:
            p.addUserDebugLine([x, y, 0.05], [x, y, 1.15], [0.55, 0.52, 0.4], lineWidth=1.2,
                               lifeTime=0, physicsClientId=client)
            p.addUserDebugPoints([[x, y, 1.25]], [[1.0, 0.86, 0.45]], pointSize=8,
                                 lifeTime=0, physicsClientId=client)
        for x, y in TREES:
            p.addUserDebugLine([x, y, 0.05], [x, y, 1.2], [0.42, 0.3, 0.18], lineWidth=2.0,
                               lifeTime=0, physicsClientId=client)
            ring = []
            for step in range(7):
                angle = 2 * math.pi * step / 7
                ring.append([x + 0.45 * math.cos(angle), y + 0.45 * math.sin(angle), 1.55])
            p.addUserDebugPoints(ring, [[0.12, 0.48, 0.28]] * len(ring), pointSize=10,
                                 lifeTime=0, physicsClientId=client)
        for low, high in self.obstacles:
            if high[2] < 9.5:
                continue
            centre = (low + high) / 2
            p.addUserDebugLine([centre[0], centre[1], high[2]], [centre[0], centre[1], high[2] + 0.7],
                               [0.75, 0.78, 0.9], lineWidth=1.2, lifeTime=0, physicsClientId=client)

    def _create_streets(self):
        def open_line(start, end):
            start, end = np.asarray(start, float), np.asarray(end, float)
            for step in range(8):
                a = start + (end - start) * step / 8
                b = start + (end - start) * (step + 1) / 8
                mid = (a + b) / 2
                if len(self.obstacles) and np.any(np.all((mid >= self.obstacles[:, 0]) & (mid <= self.obstacles[:, 1]), axis=1)):
                    continue
                p.addUserDebugLine(a, b, [0.16, 0.16, 0.26], lineWidth=1.4,
                                   lifeTime=0, physicsClientId=self.physics_client)

        for y in (5.2, 13.8):
            open_line([0.3, y, 0.04], [ARENA_X - 0.3, y, 0.04])
        for x in (5.6, 14.2):
            open_line([x, 0.3, 0.04], [x, ARENA_Y - 0.3, 0.04])

    def _create_pellets(self):
        spots, colours = [], []
        for x in np.arange(1.0, ARENA_X, 1.0):
            for y in np.arange(1.0, ARENA_Y, 1.0):
                point = np.array([x, y, 1.0])
                if len(self.obstacles) and np.any(np.all((point >= self.obstacles[:, 0]) & (point <= self.obstacles[:, 1]), axis=1)):
                    continue
                spots.append([x, y, 0.12])
                colours.append([0.42, 0.36, 0.22])
        if spots:
            p.addUserDebugPoints(spots, colours, pointSize=6, lifeTime=0, physicsClientId=self.physics_client)

    def _create_obstacles(self):
        identity = p.getQuaternionFromEuler([0, 0, 0])
        for low, high in self.obstacles:
            half, centre = (high - low) / 2, (high + low) / 2
            cap = [half[0] * 0.92, half[1] * 0.92, min(0.12, float(half[2]) * 0.3)]
            visual = p.createVisualShapeArray(
                shapeTypes=[p.GEOM_BOX, p.GEOM_BOX],
                halfExtents=[half.tolist(), cap],
                radii=[0, 0],
                lengths=[0, 0],
                visualFramePositions=[[0, 0, 0], [0, 0, float(half[2] - cap[2])]],
                visualFrameOrientations=[identity, identity],
                rgbaColors=[[0.10, 0.12, 0.62, 1], [0.55, 0.64, 1.0, 1]],
                physicsClientId=self.physics_client,
            )
            p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=centre.tolist(),
                              physicsClientId=self.physics_client)

    def _create_drone(self, position, rgba):
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
        forward = np.asarray(self.camera_forward, dtype=float)
        seeker_yaw = math.atan2(forward[1], forward[0])
        seeker_pitch = -math.asin(float(np.clip(forward[2], -1, 1)))
        velocity = np.asarray(self.evader_velocity, dtype=float)
        if np.linalg.norm(velocity[:2]) > 0.1:
            self._heading['evader'] = math.atan2(velocity[1], velocity[0])
        for body, pos, orient in (
                (self.seeker_id, self.seeker_pos, p.getQuaternionFromEuler([0, seeker_pitch, seeker_yaw])),
                (self.evader_id, self.evader_pos, p.getQuaternionFromEuler([0, 0, self._heading['evader']]))):
            p.resetBasePositionAndOrientation(body, pos.tolist(), orient,
                                              physicsClientId=self.physics_client)

    def _draw_view_cone(self, visible):
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

        blocked = (not visible) and distance <= self.detection_range and angle <= self.fov_deg / 2
        if visible or blocked:
            p.addUserDebugLine(self.seeker_pos.tolist(), self.evader_pos.tolist(),
                               [0.1, 0.9, 0.1] if visible else [1.0, 0.5, 0.0],
                               lineWidth=2 if visible else 3, lifeTime=DT,
                               physicsClientId=self.physics_client)
        status = "VISIBLE" if visible else ("BLOCKED" if blocked else "HIDDEN")
        mode = getattr(self, "seeker_mode", None) or getattr(getattr(self, "seeker_policy", None), "mode", "LEARNED")
        p.addUserDebugText(
            f"{status}    {mode}",
            [1.0, ARENA_Y - 1.0, ARENA_Z + 0.6],
            textColorRGB=[0.92, 0.94, 1.0],
            textSize=1.3,
            lifeTime=DT,
            physicsClientId=self.physics_client,
        )
        self._aim_camera()
        p.stepSimulation(physicsClientId=self.physics_client)

    def snapshot(self, path, width=960, height=720):
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
