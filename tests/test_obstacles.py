"""Obstacle world: occlusion, collisions, observations and start states."""
import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from vp.core import (Simulator, make_obstacles, segment_blocked, point_in_boxes,
                                  resolve_motion, sample_initial_state, DRONE_RADIUS)
from vp.parallel import VanishingPointParallelEnv

WALL = [[[9.0, 3.0, 0.0], [11.0, 12.0, 6.0]]]  # the 'city' central wall


def world(obstacles=WALL, **kw):
    sim = Simulator(obstacles=obstacles, **kw)
    return sim


def start(sim, seeker, evader, forward):
    sim.reset_world(np.random.default_rng(0), {"seeker_pos": seeker, "evader_pos": evader,
                                               "seeker_forward": forward})


def test_segment_blocked_through_and_over():
    boxes = make_obstacles(WALL)
    assert segment_blocked([5, 6, 3], [15, 6, 3], boxes)          # through the wall
    assert not segment_blocked([5, 6, 8], [15, 6, 8], boxes)      # above it
    assert not segment_blocked([5, 1, 3], [15, 1, 3], boxes)      # around its end
    assert not segment_blocked([5, 6, 3], [8, 6, 3], boxes)       # stops short
    assert not segment_blocked([0, 0, 0], [1, 1, 1], make_obstacles(None))


def test_wall_hides_an_evader_that_is_in_range_and_in_fov():
    sim = world()
    start(sim, [6.0, 6.0, 3.0], [14.0, 6.0, 3.0], [1.0, 0.0, 0.0])  # 8 m away, dead ahead
    assert not sim.visible
    open_sim = world(obstacles=None)
    start(open_sim, [6.0, 6.0, 3.0], [14.0, 6.0, 3.0], [1.0, 0.0, 0.0])
    assert open_sim.visible  # same geometry, no wall -> seen


def test_clear_line_of_sight_is_still_visible():
    sim = world()
    start(sim, [6.0, 6.0, 8.0], [14.0, 6.0, 8.0], [1.0, 0.0, 0.0])
    assert sim.visible


def test_drones_cannot_enter_obstacles_and_contact_is_penalised():
    sim = world()
    start(sim, [6.0, 6.0, 3.0], [5.0, 5.0, 3.0], [1.0, 0.0, 0.0])
    rewards = None
    for _ in range(80):  # Seeker flies straight at the wall at 1.8 m/s
        rewards, fresh, _ = sim.advance(np.array([1.8, 0, 0]), np.zeros(3), camera_action=[0, 0])
        assert not point_in_boxes(sim.seeker_pos, make_obstacles(WALL))
    assert sim.seeker_pos[0] <= 9.0 - DRONE_RADIUS + 1e-9
    assert sim.seeker_wall_touches == 1          # one streak, scored like a wall touch
    assert sim.boundary_steps["seeker"] > 50


def test_motion_slides_along_a_face():
    boxes = make_obstacles(WALL)
    position, blocked = resolve_motion([8.6, 6.0, 3.0], [9.0, 6.5, 3.0], boxes, DRONE_RADIUS)
    assert blocked and position[1] == pytest.approx(6.5) and position[0] == pytest.approx(8.6)


def test_observation_sizes_and_api():
    assert VanishingPointParallelEnv().observation_space("seeker").shape == (21,)
    env = VanishingPointParallelEnv(obstacles="city")
    assert env.observation_space("seeker").shape == (30,) and env.observation_space("evader").shape == (30,)
    obs, _ = env.reset(seed=1)
    assert obs["seeker"].shape == (30,) and np.all(np.abs(obs["evader"]) <= 1)
    parallel_api_test(VanishingPointParallelEnv(obstacles="city", episode_steps=40), num_cycles=45)


@pytest.mark.parametrize("preset", ["city", "pillars"])
def test_sampled_starts_are_clear_and_visible(preset):
    boxes = make_obstacles(preset)
    for seed in range(40):
        seeker, evader, _ = sample_initial_state(np.random.default_rng(seed), obstacles=boxes)
        assert not point_in_boxes(seeker, boxes, 0.5) and not point_in_boxes(evader, boxes, 0.5)
        assert not segment_blocked(seeker, evader, boxes)


def test_explicit_start_inside_an_obstacle_is_rejected():
    with pytest.raises(ValueError):
        start(world(), [10.0, 6.0, 3.0], [5.0, 5.0, 3.0], [1.0, 0.0, 0.0])


def test_scripted_seeker_does_not_freeze_against_obstacles():
    """Never stays pressed against an obstacle in TRACK/SEARCH for more than ~2 s."""
    for seed in (3, 4, 7):
        env = VanishingPointParallelEnv(obstacles="city", episode_steps=600)
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
        make_obstacles("nope")
    with pytest.raises(ValueError):
        make_obstacles([[[1, 1, 1], [0, 2, 2]]])


def test_obstacle_free_world_is_unchanged():
    sim = Simulator()
    assert len(sim.obstacles) == 0 and sim.obs_extra == 0


def test_seeker_history_stacks_newest_first_and_pads_at_reset():
    env = VanishingPointParallelEnv(obstacles="city", seeker_history=3, seeker_history_stride=2, episode_steps=20)
    assert env.observation_space("seeker").shape == (90,) and env.observation_space("evader").shape == (30,)
    obs, _ = env.reset(seed=4)
    first = obs["seeker"]
    assert np.array_equal(first[:30], first[30:60]) and np.array_equal(first[:30], first[60:])  # padded
    for _ in range(4):
        obs, *_ = env.step({"seeker": np.array([1, 0, 0, 0, 0], np.float32), "evader": np.zeros(3, np.float32)})
    now = obs["seeker"]
    assert not np.array_equal(now[:30], now[30:60])           # newest differs from 2 steps ago
    assert np.array_equal(now[:30], env.sim._seeker_base_observation())
    assert VanishingPointParallelEnv(episode_steps=20).observation_space("seeker").shape == (21,)


def run_steps(sim, n, seeker_velocity=(0, 0, 0), camera=(0, 0)):
    for _ in range(n):
        sim.advance(np.array(seeker_velocity, float), np.zeros(3), camera_action=list(camera))


def test_first_detection_time_is_zero_when_visible_and_measures_search_when_hidden():
    visible = world(obstacles=None)
    start(visible, [5.0, 5.0, 5.0], [10.0, 5.0, 5.0], [1.0, 0.0, 0.0])
    run_steps(visible, 5)
    assert visible.episode_summary()["first_detection_time_s"] == 0.0

    hidden = world(obstacles=None)
    start(hidden, [5.0, 5.0, 5.0], [10.0, 5.0, 5.0], [-1.0, 0.0, 0.0])  # camera faces away
    run_steps(hidden, 5)
    assert hidden.episode_summary()["first_detection_time_s"] is None   # not detected yet
    run_steps(hidden, 40, camera=(1.0, 0.0))                             # turn at 100 deg/s until it sees the Evader
    assert 1.0 < hidden.episode_summary()["first_detection_time_s"] < 2.5


def test_drone_collisions_are_counted_as_steps_and_events():
    sim = world(obstacles=None)
    start(sim, [5.0, 5.0, 5.0], [5.5, 5.0, 5.0], [1.0, 0.0, 0.0])  # 0.5 m apart < 0.7 m
    run_steps(sim, 6)
    summary = sim.episode_summary()
    assert summary["drone_collision_steps"] == 6 and summary["drone_collision_events"] == 1
    far = world(obstacles=None)
    start(far, [5.0, 5.0, 5.0], [9.0, 5.0, 5.0], [1.0, 0.0, 0.0])
    run_steps(far, 6)
    assert far.episode_summary()["drone_collision_steps"] == 0
