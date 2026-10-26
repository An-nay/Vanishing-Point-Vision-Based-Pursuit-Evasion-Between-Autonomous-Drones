import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from vp.core import (Simulator, make_boxes, ray_hits, in_box,
                                  slide, rand_start, DRONE_RADIUS)
from vp.parallel import DroneEnv

WALL = [[[9.0, 3.0, 0.0], [11.0, 12.0, 6.0]]]


def world(obstacles=WALL, **kw):
    sim = Simulator(obstacles=obstacles, **kw)
    return sim


def start(sim, seeker, evader, forward):
    sim.reset_world(np.random.default_rng(0), {"seeker_pos": seeker, "evader_pos": evader,
                                               "seeker_forward": forward})


def test_segment_blocked_through_and_over():
    boxes = make_boxes(WALL)
    assert ray_hits([5, 6, 3], [15, 6, 3], boxes)
    assert not ray_hits([5, 6, 8], [15, 6, 8], boxes)
    assert not ray_hits([5, 1, 3], [15, 1, 3], boxes)
    assert not ray_hits([5, 6, 3], [8, 6, 3], boxes)
    assert not ray_hits([0, 0, 0], [1, 1, 1], make_boxes(None))


def test_wall_hides_an_evader_that_is_in_range_and_in_fov():
    sim = world()
    start(sim, [6.0, 6.0, 3.0], [14.0, 6.0, 3.0], [1.0, 0.0, 0.0])
    assert not sim.visible
    open_sim = world(obstacles=None)
    start(open_sim, [6.0, 6.0, 3.0], [14.0, 6.0, 3.0], [1.0, 0.0, 0.0])
    assert open_sim.visible


def test_clear_line_of_sight_is_still_visible():
    sim = world()
    start(sim, [6.0, 6.0, 8.0], [14.0, 6.0, 8.0], [1.0, 0.0, 0.0])
    assert sim.visible


def test_drones_cannot_enter_obstacles_and_contact_is_penalised():
    sim = world()
    start(sim, [6.0, 6.0, 3.0], [5.0, 5.0, 3.0], [1.0, 0.0, 0.0])
    rewards = None
    for _ in range(80):
        rewards, fresh, _ = sim.advance(np.array([1.8, 0, 0]), np.zeros(3), camera_action=[0, 0])
        assert not in_box(sim.seeker_pos, make_boxes(WALL))
    assert sim.seeker_pos[0] <= 9.0 - DRONE_RADIUS + 1e-9
    assert sim.seeker_wall_touches == 1
    assert sim.boundary_steps["seeker"] > 50


def test_motion_slides_along_a_face():
    boxes = make_boxes(WALL)
    position, blocked = slide([8.6, 6.0, 3.0], [9.0, 6.5, 3.0], boxes, DRONE_RADIUS)
    assert blocked and position[1] == pytest.approx(6.5) and position[0] == pytest.approx(8.6)


def test_observation_sizes_and_api():
    assert DroneEnv().observation_space("seeker").shape == (21,)
    env = DroneEnv(obstacles="city")
    assert env.observation_space("seeker").shape == (30,) and env.observation_space("evader").shape == (30,)
    obs, _ = env.reset(seed=1)
    assert obs["seeker"].shape == (30,) and np.all(np.abs(obs["evader"]) <= 1)
    parallel_api_test(DroneEnv(obstacles="city", episode_steps=40), num_cycles=45)


@pytest.mark.parametrize("preset", ["city", "pillars", "maze"])
def test_sampled_starts_are_clear_and_visible(preset):
    boxes = make_boxes(preset)
    for seed in range(40):
        seeker, evader, _ = rand_start(np.random.default_rng(seed), obstacles=boxes)
        assert not in_box(seeker, boxes, 0.5) and not in_box(evader, boxes, 0.5)
        assert not ray_hits(seeker, evader, boxes)


def test_explicit_start_inside_an_obstacle_is_rejected():
    with pytest.raises(ValueError):
        start(world(), [10.0, 6.0, 3.0], [5.0, 5.0, 3.0], [1.0, 0.0, 0.0])


def test_scripted_seeker_does_not_freeze_against_obstacles():
    for seed in (3, 4, 7):
        env = DroneEnv(obstacles="city", episode_steps=600)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        previous, run, longest = env.sim.seeker_pos.copy(), 0, 0
        while env.agents:
            env.step_scripted(rng.uniform(-1, 1, 3))
            moved = np.linalg.norm(env.sim.seeker_pos - previous)
            previous = env.sim.seeker_pos.copy()
            run = run + 1 if env.seeker_mode in ("TRACK", "SEARCH") and moved < 0.05 else 0
            longest = max(longest, run)
        assert longest < 20


def test_unknown_preset_and_bad_boxes_are_rejected():
    with pytest.raises(ValueError):
        make_boxes("nope")
    with pytest.raises(ValueError):
        make_boxes([[[1, 1, 1], [0, 2, 2]]])


def test_obstacle_free_world_is_unchanged():
    sim = Simulator()
    assert len(sim.obstacles) == 0 and sim.obs_extra == 0


def test_seeker_history_stacks_newest_first_and_pads_at_reset():
    env = DroneEnv(obstacles="city", seeker_history=3, seeker_history_stride=2, episode_steps=20)
    assert env.observation_space("seeker").shape == (90,) and env.observation_space("evader").shape == (30,)
    obs, _ = env.reset(seed=4)
    first = obs["seeker"]
    assert np.array_equal(first[:30], first[30:60]) and np.array_equal(first[:30], first[60:])
    for _ in range(4):
        obs, *_ = env.step({"seeker": np.array([1, 0, 0, 0, 0], np.float32), "evader": np.zeros(3, np.float32)})
    now = obs["seeker"]
    assert not np.array_equal(now[:30], now[30:60])
    assert np.array_equal(now[:30], env.sim._seeker_base_observation())
    assert DroneEnv(episode_steps=20).observation_space("seeker").shape == (21,)


def run_steps(sim, n, seeker_velocity=(0, 0, 0), camera=(0, 0)):
    for _ in range(n):
        sim.advance(np.array(seeker_velocity, float), np.zeros(3), camera_action=list(camera))


def test_first_detection_time_is_zero_when_visible_and_measures_search_when_hidden():
    visible = world(obstacles=None)
    start(visible, [5.0, 5.0, 5.0], [10.0, 5.0, 5.0], [1.0, 0.0, 0.0])
    run_steps(visible, 5)
    assert visible.episode_summary()["first_detection_time_s"] == 0.0

    hidden = world(obstacles=None)
    start(hidden, [5.0, 5.0, 5.0], [10.0, 5.0, 5.0], [-1.0, 0.0, 0.0])
    run_steps(hidden, 5)
    assert hidden.episode_summary()["first_detection_time_s"] is None
    run_steps(hidden, 40, camera=(1.0, 0.0))
    assert 1.0 < hidden.episode_summary()["first_detection_time_s"] < 2.5


def test_drone_collisions_are_counted_as_steps_and_events():
    sim = world(obstacles=None)
    start(sim, [5.0, 5.0, 5.0], [5.5, 5.0, 5.0], [1.0, 0.0, 0.0])
    run_steps(sim, 6)
    summary = sim.episode_summary()
    assert summary["drone_collision_steps"] == 6 and summary["drone_collision_events"] == 1
    far = world(obstacles=None)
    start(far, [5.0, 5.0, 5.0], [9.0, 5.0, 5.0], [1.0, 0.0, 0.0])
    run_steps(far, 6)
    assert far.episode_summary()["drone_collision_steps"] == 0
