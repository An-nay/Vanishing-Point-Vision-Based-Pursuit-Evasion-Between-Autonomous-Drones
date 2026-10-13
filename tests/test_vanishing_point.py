"""
Independent tests for vp/ppo.py.

Run from the project root with the project's virtual environment active:

    python -m pytest tests/test_vanishing_point.py -v
"""

import math
import numpy as np
import pytest

from vp import ppo as vp


# ---------------------------------------------------------------------
# 1. Field-of-view tests
# ---------------------------------------------------------------------
def test_visible_straight_ahead():
    visible, dist, angle = vp.is_visible(
        seeker_pos=np.array([0.0, 0.0, 0.0]),
        seeker_forward=np.array([1.0, 0.0, 0.0]),
        evader_pos=np.array([5.0, 0.0, 0.0]),
        fov_deg=90.0,
        detection_range=10.0,
    )
    assert visible
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_not_visible_just_outside_fov_edge():
    # Half-FOV is 45 degrees for a 90-degree FOV. Put the target at 46 degrees.
    seeker_pos = np.array([0.0, 0.0, 0.0])
    seeker_forward = np.array([1.0, 0.0, 0.0])
    angle_rad = math.radians(46.0)
    evader_pos = seeker_pos + 5.0 * np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])
    visible, _, angle = vp.is_visible(seeker_pos, seeker_forward, evader_pos, fov_deg=90.0, detection_range=10.0)
    assert not visible
    assert angle == pytest.approx(46.0, abs=0.5)


def test_visible_just_inside_fov_edge():
    seeker_pos = np.array([0.0, 0.0, 0.0])
    seeker_forward = np.array([1.0, 0.0, 0.0])
    angle_rad = math.radians(44.0)
    evader_pos = seeker_pos + 5.0 * np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])
    visible, _, angle = vp.is_visible(seeker_pos, seeker_forward, evader_pos, fov_deg=90.0, detection_range=10.0)
    assert visible


def test_not_visible_beyond_detection_range():
    visible, dist, _ = vp.is_visible(
        seeker_pos=np.array([0.0, 0.0, 0.0]),
        seeker_forward=np.array([1.0, 0.0, 0.0]),
        evader_pos=np.array([15.0, 0.0, 0.0]),
        fov_deg=90.0,
        detection_range=10.0,
    )
    assert not visible
    assert dist == pytest.approx(15.0)


# ---------------------------------------------------------------------
# 2. Seeker mode transition tests
# ---------------------------------------------------------------------
def test_seeker_switches_to_track_when_visible():
    rng = np.random.default_rng(0)
    seeker = vp.ScriptedSeekerPolicy(rng, initial_forward=np.array([1.0, 0.0, 0.0]))
    assert seeker.mode == "SCAN"
    seeker.act(
        seeker_pos=np.array([0.0, 0.0, 0.0]),
        evader_pos=np.array([2.0, 0.0, 0.0]),
        visible=True,
    )
    assert seeker.mode == "TRACK"


def test_seeker_switches_from_scan_to_search_after_timeout():
    rng = np.random.default_rng(0)
    seeker = vp.ScriptedSeekerPolicy(rng, initial_forward=np.array([1.0, 0.0, 0.0]))
    seeker_pos = np.array([5.0, 5.0, 5.0])
    evader_pos = np.array([100.0, 100.0, 100.0])  # far away and never visible
    steps_needed = int(vp.SCAN_TIMEOUT / vp.DT) + 2
    for _ in range(steps_needed):
        seeker.act(seeker_pos, evader_pos, visible=False)
    assert seeker.mode == "SEARCH"
    assert seeker.searches >= 1


# ---------------------------------------------------------------------
# 3. Wall / boundary penalty tests
# ---------------------------------------------------------------------
def test_wall_penalty_applies_once_per_contact_streak():
    env = vp.VanishingPointEnv(render_mode=None, randomize_start=False)
    env.reset(seed=0)
    # Place the Evader right next to the +x wall so one step drives it into contact.
    env.evader_pos = np.array([19.5, 10.0, 5.0], dtype=np.float64)

    into_wall = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(5):
        env.step(into_wall)

    # wall_touches should have incremented only once across the whole streak,
    # since it's a one-time penalty per contact, not per step spent in contact.
    assert env.wall_touches == 1
    env.close()


def test_wall_penalty_increments_again_after_leaving_wall():
    env = vp.VanishingPointEnv(render_mode=None, randomize_start=False)
    env.reset(seed=0)
    env.evader_pos = np.array([19.5, 10.0, 5.0], dtype=np.float64)

    into_wall = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(3):
        env.step(into_wall)
    assert env.wall_touches == 1

    # Move it clearly away from the wall, then place it back at the wall
    # directly rather than relying on stepping speed math, so the test
    # doesn't depend on exact velocity or timestep values.
    env.evader_pos = np.array([15.0, 10.0, 5.0], dtype=np.float64)
    env.evader_was_on_boundary = False

    env.evader_pos = np.array([19.5, 10.0, 5.0], dtype=np.float64)
    for _ in range(3):
        env.step(into_wall)
    assert env.wall_touches == 2
    env.close()


# ---------------------------------------------------------------------
# 4. Reset / observation tests
# ---------------------------------------------------------------------
def test_reset_returns_correct_observation_shape_and_bounds():
    env = vp.VanishingPointEnv(render_mode=None, randomize_start=True)
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.obs_dim,)
    assert np.all(np.isfinite(obs))
    assert np.all(obs >= -1.0 - 1e-5)
    assert np.all(obs <= 1.0 + 1e-5)
    env.close()


def test_episode_truncates_at_expected_length():
    env = vp.VanishingPointEnv(render_mode=None, randomize_start=True)
    env.reset(seed=1)
    zero_action = np.zeros(3, dtype=np.float32)
    steps = 0
    truncated = False
    while not truncated and steps < vp.EPISODE_STEPS + 5:
        _, _, terminated, truncated, info = env.step(zero_action)
        steps += 1
    assert truncated
    assert steps == vp.EPISODE_STEPS
    assert "episode_summary" in info
    env.close()


# ---------------------------------------------------------------------
# 5. NaN / infinite value checks over a random rollout
# ---------------------------------------------------------------------
def test_no_nan_or_inf_over_random_episodes():
    env = vp.VanishingPointEnv(render_mode=None, randomize_start=True)
    env.action_space.seed(123)
    for episode in range(3):
        obs, _ = env.reset(seed=100 + episode)
        assert np.all(np.isfinite(obs))
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs)), "Non-finite value found in observation"
            assert np.isfinite(reward), "Non-finite reward"
            done = terminated or truncated
    env.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
