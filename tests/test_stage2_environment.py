"""Phase A: world correctness, information restrictions and API contract."""
import math
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from vanishing_point_core import Simulator, DT, SEEKER_SPEED, EVADER_SPEED, is_visible
from vanishing_point_parallel import VanishingPointParallelEnv

START = {"seeker_pos": [4., 10., 5.], "evader_pos": [12., 10., 5.],
         "seeker_forward": [1., 0., 0.]}


def make_env(**kwargs):
    env = VanishingPointParallelEnv(randomize_start=False, **kwargs)
    env.reset(seed=7)
    return env


def actions(seeker=None, evader=None):
    return {"seeker": np.zeros(5) if seeker is None else np.array(seeker),
            "evader": np.zeros(3) if evader is None else np.array(evader)}


def test_parallel_api():
    parallel_api_test(VanishingPointParallelEnv(episode_steps=15), num_cycles=35)


def test_joint_motion_and_speed_limits():
    env = make_env()
    env.step(actions([1, 1, 1, 0, 0], [-1, 0, 0]))
    np.testing.assert_allclose(env.sim.seeker_pos, np.array(START["seeker_pos"])
                               + np.ones(3) / math.sqrt(3) * SEEKER_SPEED * DT)
    np.testing.assert_allclose(env.sim.evader_pos, [12 - EVADER_SPEED * DT, 10, 5])
    assert np.linalg.norm(env.sim.seeker_velocity) == pytest.approx(SEEKER_SPEED)


def test_camera_rates_wrapping_and_pitch_limit():
    env = make_env()
    env.step(actions([0, 0, 0, 1, 1]))
    assert env.sim.camera_yaw == pytest.approx(math.radians(100) * DT)
    assert env.sim.camera_pitch == pytest.approx(math.radians(90) * DT)
    for _ in range(40):
        env.step(actions([0, 0, 0, 1, 1]))
    assert -math.pi <= env.sim.camera_yaw < math.pi
    assert env.sim.camera_pitch == pytest.approx(math.radians(89))
    assert np.linalg.norm(env.sim.camera_forward) == pytest.approx(1.)


def test_shared_fov_range_and_vertical_geometry():
    assert is_visible([0, 0, 0], [0, 0, 1], [0, 0, 5])[0]
    assert not is_visible([0, 0, 0], [1, 0, 0], [0, 0, 5])[0]
    assert not is_visible([0, 0, 0], [1, 0, 0], [10.01, 0, 0])[0]
    direction = [math.cos(math.radians(44)), math.sin(math.radians(44)), 0]
    assert is_visible([0, 0, 0], [1, 0, 0], np.array(direction) * 5)[0]


def test_hidden_evader_cannot_change_seeker_observation_or_info():
    env = make_env()
    env.sim.set_camera_forward([-1., 0., 0.])
    env.step(actions())  # lose sight; memory still contains the visible reset position
    observation = env.sim.seeker_observation().copy()
    info = env._infos({"seeker": False, "evader": False})["seeker"]
    np.testing.assert_allclose(observation[16:19], [8 / 20, 0, 0])
    env.sim.evader_pos = np.array([17., 14., 8.])
    env.sim.evader_velocity = np.array([-2., .3, .7])
    env.sim.visible, env.sim.distance, env.sim.angle = env.sim.check_visibility()
    assert not env.sim.visible
    np.testing.assert_array_equal(observation, env.sim.seeker_observation())
    assert info == env._infos({"seeker": False, "evader": False})["seeker"]
    assert env.sim.time_since_detection_steps == 1


def test_never_seen_hidden_start_does_not_seed_true_target_memory():
    env = make_env()
    hidden = dict(START, seeker_forward=[-1., 0., 0.])
    obs, _ = env.reset(seed=7, options={"initial_state": hidden})
    assert not env.sim.visible and env.sim.last_known_position is None
    np.testing.assert_array_equal(obs["seeker"][16:20], np.zeros(4))
    env.step(actions())
    assert env.sim.first_loss_step is None


def test_independent_wall_penalty_and_reentry():
    env = make_env()
    env.sim.evader_pos = np.array([19.5, 10., 5.])
    _, rewards, _, _, _ = env.step(actions(evader=[1, 0, 0]))
    assert rewards == {"seeker": -1., "evader": -1.}
    _, rewards, _, _, _ = env.step(actions(evader=[1, 0, 0]))
    assert rewards["evader"] == 1. and env.sim.wall_touches == 1
    env.step(actions(evader=[-1, 0, 0]))  # actual leave and re-entry, no flag edits
    env.step(actions(evader=[1, 0, 0]))
    assert env.sim.wall_touches == 2 and env.sim.seeker_wall_touches == 0


def test_seeker_penalty_does_not_affect_evader_reward():
    env = make_env()
    env.reset(options={"initial_state": {"seeker_pos": [19.5, 10, 5],
                                       "evader_pos": [12, 10, 5],
                                       "seeker_forward": [-1, 0, 0]}})
    _, rewards, _, _, info = env.step(actions([1, 0, 0, 0, 0]))
    assert rewards == {"seeker": -1., "evader": -1.}
    assert info["seeker"]["wall_contact"]
    assert not info["evader"]["wall_contact"]


def test_detection_does_not_terminate_and_final_observations_are_returned():
    env = make_env(episode_steps=3)
    for step in range(3):
        obs, rewards, terminated, truncated, info = env.step(actions())
        assert not any(terminated.values())
        assert all(truncated.values()) == (step == 2)
        assert set(obs) == {"seeker", "evader"}
        assert rewards == {"seeker": 1., "evader": -1.}
    assert env.agents == [] and info["seeker"]["episode_summary"]["steps"] == 3
    assert env.step({}) == ({}, {}, {}, {}, {})


def test_reset_reproducibility_clears_statistics_and_memory():
    env = VanishingPointParallelEnv()
    first, _ = env.reset(seed=32)
    state = env.state().copy()
    env.step(actions(evader=[1, 0, 0]))
    second, _ = env.reset(seed=32)
    for agent in first:
        np.testing.assert_array_equal(first[agent], second[agent])
    np.testing.assert_array_equal(state, env.state())
    assert env.sim.step_count == env.sim.wall_touches == env.sim.seeker_wall_touches == 0
    assert env.sim.reacquisition_steps == []


@pytest.mark.parametrize("bad", [
    {"seeker": np.zeros(5)}, actions(seeker=[0, 0, 0]),
    actions(evader=[float("nan"), 0, 0]), actions(seeker=[0, 0, 0, float("inf"), 0])
])
def test_invalid_joint_actions_are_atomic(bad):
    env = make_env()
    state = env.state().copy()
    with pytest.raises(ValueError):
        env.step(bad)
    np.testing.assert_array_equal(state, env.state())


def test_observation_dimensions_spaces_and_random_rollout():
    env = VanishingPointParallelEnv()
    for seed in [20, 21]:
        obs, _ = env.reset(seed=seed)
        while env.agents:
            for agent in env.agents:
                assert env.observation_space(agent).contains(obs[agent])
            assert env.state_space.contains(env.state())
            obs, rewards, _, _, _ = env.step(
                {agent: env.action_space(agent).sample() for agent in env.agents})
            assert all(np.isfinite(r) for r in rewards.values())


def test_reacquisition_tracks_hidden_streaks():
    env = make_env()
    env.sim.set_camera_forward([-1, 0, 0])
    env.step(actions())
    env.step(actions())
    env.sim.set_camera_forward([1, 0, 0])
    env.step(actions())
    summary = env.sim.episode_summary()
    assert summary["longest_hidden_s"] == pytest.approx(.2)
    assert summary["reacquisition_mean_s"] == pytest.approx(.2)
    assert summary["reacquisitions"] == summary["loss_events"] == 1


def test_stage1_and_parallel_scripted_rollouts_match():
    from vanishing_point_ppo import VanishingPointEnv
    stage1 = VanishingPointEnv()
    stage2 = VanishingPointParallelEnv()
    rng = np.random.default_rng(18)
    for seed in [18, 29]:
        old_obs, _ = stage1.reset(seed=seed)
        new_obs, _ = stage2.reset(seed=seed)
        np.testing.assert_array_equal(old_obs, new_obs["evader"])
        for _ in range(600):
            control = rng.uniform(-1., 1., 3).astype(np.float32)
            old_obs, old_reward, old_term, old_trunc, _ = stage1.step(control)
            new_obs, rewards, terms, truncs, _ = stage2.step_scripted(control)
            np.testing.assert_array_equal(old_obs, new_obs["evader"])
            assert old_reward == rewards["evader"]
            assert old_term == terms["evader"] and old_trunc == truncs["evader"]
    stage1.close()
    stage2.close()

def test_renderer_uses_shared_world_and_releases_clients(monkeypatch):
    import pybullet as bullet
    from vanishing_point_ppo import VanishingPointEnv
    original_connect = bullet.connect
    monkeypatch.setattr(bullet, "connect", lambda mode: original_connect(bullet.DIRECT))
    # Exercise the actual renderer without opening GUI windows or waiting for playback.
    monkeypatch.setattr("vanishing_point_parallel.time.sleep", lambda seconds: None)
    stage1 = VanishingPointEnv(render_mode="human", randomize_start=False)
    stage2 = VanishingPointParallelEnv(render_mode="human", randomize_start=False)
    try:
        stage1.reset(seed=4)
        stage2.reset(seed=4)
        stage2.step(actions(seeker=[1, 0, 0, 1, 0]))
        assert stage1.physics_client != stage2.physics_client
        position, _ = bullet.getBasePositionAndOrientation(
            stage2.seeker_id, physicsClientId=stage2.physics_client)
        np.testing.assert_allclose(position, stage2.sim.seeker_pos)
        assert bullet.getNumBodies(stage1.physics_client) == 3
        assert bullet.getNumBodies(stage2.physics_client) == 3
    finally:
        stage1.close()
        stage2.close()
    assert stage1.physics_client is stage2.physics_client is None
