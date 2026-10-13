"""PettingZoo Parallel interface for the shared Vanishing Point world."""
import time
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from vp.core import (
    Simulator, ScriptedSeekerPolicy, checked_vector, limit_velocity,
    SEEKER_SPEED, EVADER_SPEED, DT,
)
from vp.render import RendererMixin

AGENTS = ("seeker", "evader")


class VanishingPointParallelEnv(ParallelEnv, RendererMixin):
    metadata = {"name": "vanishing_point_v2", "render_modes": ["human"],
                "render_fps": int(1 / DT), "is_parallelizable": True}

    def __init__(self, render_mode=None, **world_options):
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.agents = []
        self.sim = Simulator(**world_options)
        obs_dim = 21 + self.sim.obs_extra  # +9 when obstacles are present
        dims = {"seeker": obs_dim * self.sim.seeker_history, "evader": obs_dim}
        self.observation_spaces = {a: spaces.Box(-1., 1., (dims[a],), np.float32) for a in AGENTS}
        self.action_spaces = {a: spaces.Box(-1., 1., (5 if a == "seeker" else 3,), np.float32)
                              for a in AGENTS}
        self.state_space = spaces.Box(-1., 1., (26,), np.float32)
        self.physics_client = self.seeker_id = self.evader_id = None
        self.np_random, _ = seeding.np_random()
        self.seeker_policy = None

    def __getattr__(self, name):
        # RendererMixin reads world positions/constants through the interface.
        sim = self.__dict__.get("sim")
        if sim is not None and hasattr(sim, name):
            return getattr(sim, name)
        raise AttributeError(name)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            for index, agent in enumerate(AGENTS):
                self.action_space(agent).seed(seed + index)
        self.sim.reset_world(self.np_random, (options or {}).get("initial_state"))
        self.seeker_policy = ScriptedSeekerPolicy(self.np_random, self.sim.initial_forward,
                                                 obstacles=self.sim.obstacles)
        self.seeker_mode = "LEARNED"
        self.agents = self.possible_agents.copy()
        if self.render_mode == "human":
            if self.physics_client is None:
                self._init_renderer()
            self._reset_render_objects()
            self.render()
        return self._observations(), self._infos({a: False for a in AGENTS})

    def _observations(self):
        return {"seeker": self.sim.seeker_observation(),
                "evader": self.sim.evader_observation()}

    def state(self):
        if self.sim.seeker_pos is None:
            raise RuntimeError("call reset before requesting state")
        return self.sim.global_state()

    def _infos(self, contacts):
        # The Seeker never receives true hidden distance, bearing or target velocity.
        return {a: {"visible": self.sim.visible, "wall_contact": bool(contacts[a]),
                    "wall_touches": (self.sim.seeker_wall_touches if a == "seeker"
                                     else self.sim.wall_touches)}
                for a in AGENTS}

    def _finish_transition(self, rewards, contacts, truncated):
        observations = self._observations()  # retain final observations for replay
        terminations = {a: False for a in AGENTS}
        truncations = {a: truncated for a in AGENTS}
        infos = self._infos(contacts)
        if truncated:
            summary = self.sim.episode_summary()
            for agent in AGENTS:
                infos[agent]["episode_summary"] = summary.copy()
            self.agents = []
        if self.render_mode == "human":
            self.render()
            time.sleep(DT)
        return observations, rewards, terminations, truncations, infos

    def step(self, actions):
        if not self.agents:
            if actions:
                raise RuntimeError("no live agents; call reset before stepping")
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError("a joint step requires actions for both seeker and evader")
        # Validate BOTH actions before rotating the camera or moving either drone.
        seeker = np.clip(checked_vector(actions["seeker"], 5, "seeker action"), -1., 1.)
        evader = np.clip(checked_vector(actions["evader"], 3, "evader action"), -1., 1.)
        self.seeker_mode = "LEARNED"
        result = self.sim.advance(limit_velocity(seeker[:3], SEEKER_SPEED),
                                  limit_velocity(evader, EVADER_SPEED),
                                  camera_action=seeker[3:])
        return self._finish_transition(*result)

    def step_scripted(self, evader_action):
        """Baseline-only adapter preserving the original instantaneous camera turns.

        Learning always uses step(joint_actions). This adapter advances the very
        same world, but uses the original TRACK/SCAN/SEARCH controller.
        """
        if not self.agents:
            raise RuntimeError("call reset before stepping")
        evader = np.clip(checked_vector(evader_action, 3, "evader action"), -1., 1.)
        velocity = self.seeker_policy.act(
            self.sim.seeker_pos, self.sim.evader_pos, self.sim.visible)
        self.seeker_mode = self.seeker_policy.mode
        result = self.sim.advance(velocity, limit_velocity(evader, EVADER_SPEED),
                                  seeker_forward=self.seeker_policy.forward)
        return self._finish_transition(*result)

    def render(self):
        if self.render_mode == "human":
            self._render_step(self.sim.visible, self.sim.distance, self.sim.angle,
                              self.sim.total_reward)

    def close(self):
        RendererMixin.close(self)


def parallel_env(**kwargs):
    return VanishingPointParallelEnv(**kwargs)
