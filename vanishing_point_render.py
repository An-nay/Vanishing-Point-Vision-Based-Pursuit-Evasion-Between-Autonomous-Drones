"""Lazy PyBullet visualization shared by both RL interfaces."""
from vanishing_point_core import ARENA_X, ARENA_Y, ARENA_Z, DT, DRONE_RADIUS

class RendererMixin:
    def _init_renderer(self):
        global p, pybullet_data
        import pybullet as p
        import pybullet_data
        self.physics_client = p.connect(p.GUI)
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

        forward_end = self.seeker_pos + self.camera_forward * self.detection_range
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
                f"Step: {self.step_count} | Mode: {getattr(self, 'seeker_mode', getattr(getattr(self, 'seeker_policy', None), 'mode', 'LEARNED'))} | "
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
