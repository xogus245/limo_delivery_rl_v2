"""Curriculum stages: waypoint subsets, plane-crossing capture, policy transfer."""

import numpy as np
import pytest

from limo_delivery_rl_v2.delivery_env import LimoWaypointRLEnv
from limo_delivery_rl_v2.geometry import Pose2D
from limo_delivery_rl_v2.state import (
    WAYPOINTS,
    DeliveryEnvConfig,
    EpisodeConfig,
    StopReason,
    observation_dim,
    stage_config,
)
from limo_delivery_rl_v2.train_ppo import build_parser, build_stage_config
from limo_delivery_rl_v2.waypoint_manager import WaypointManager

# --------------------------------------------------------------- stage config


def test_default_stage_config_is_unchanged():
    config = stage_config()
    base = DeliveryEnvConfig()

    assert config.waypoints == base.waypoints
    assert config.episode.waypoint_radius == base.episode.waypoint_radius
    assert config.episode.waypoint_capture_width == 0.0


@pytest.mark.parametrize("count", [1, 2, 3])
def test_stage_config_takes_the_first_n_waypoints_in_order(count):
    config = stage_config(waypoint_count=count)

    assert config.waypoints == WAYPOINTS[:count]


def test_stage_config_rejects_an_out_of_range_waypoint_count():
    for bad in (0, -1, 4):
        with pytest.raises(ValueError, match="waypoint_count"):
            stage_config(waypoint_count=bad)


def test_stage_config_can_disable_obstacles_for_a_warmup_stage():
    assert stage_config(obstacles_enabled=False).obstacles.enabled is False
    assert stage_config().obstacles.enabled is True


def test_training_arguments_map_onto_the_stage_config():
    args = build_parser().parse_args(
        [
            "--waypoints", "1",
            "--waypoint-radius", "0.9",
            "--waypoint-hold-steps", "1",
            "--waypoint-capture-width", "1.0",
        ]
    )

    config = build_stage_config(args)

    assert config.waypoints == WAYPOINTS[:1]
    assert config.episode.waypoint_radius == pytest.approx(0.9)
    assert config.episode.waypoint_hold_steps == 1
    assert config.episode.waypoint_capture_width == pytest.approx(1.0)


def test_training_defaults_leave_every_stage_knob_at_the_specification():
    config = build_stage_config(build_parser().parse_args([]))

    assert config == DeliveryEnvConfig()
    assert build_parser().parse_args([]).resume is None


# ------------------------------------------------------- plane-crossing capture


def manager(width: float, count: int = 1) -> WaypointManager:
    """A manager over the first ``count`` waypoints with the given capture width."""
    instance = WaypointManager(
        WAYPOINTS[:count], EpisodeConfig(waypoint_capture_width=width)
    )
    instance.reset(start_xy=(0.0, 0.0))
    return instance


def test_capture_is_disabled_by_default_so_the_specification_rule_stands():
    instance = manager(width=0.0)

    # 1.5 m past the waypoint, well outside the 0.60 m radius.
    assert not instance.update(Pose2D(4.5, 0.0, 0.0)).reached
    assert not instance.completed


def test_an_off_centre_pass_is_missed_without_capture_and_caught_with_it():
    #  Drives past x=3.0 at y=0.8: the 0.60 m radius is never entered.
    fly_by = Pose2D(3.2, 0.8, 0.0)

    assert not manager(width=0.0).update(fly_by).reached
    assert manager(width=1.0).update(fly_by).reached


def test_capture_requires_actually_crossing_the_waypoint_plane():
    instance = manager(width=1.0)

    # Still short of the waypoint, off to the side: not reached.
    assert not instance.update(Pose2D(2.5, 0.8, 0.0)).reached
    # Now past it.
    assert instance.update(Pose2D(3.1, 0.8, 0.0)).reached


def test_capture_band_bounds_how_far_off_course_still_counts():
    assert manager(width=1.0).update(Pose2D(3.2, 0.9, 0.0)).reached
    assert not manager(width=1.0).update(Pose2D(3.2, 1.5, 0.0)).reached


def test_capture_grants_the_bonus_exactly_once():
    instance = manager(width=1.0, count=1)
    bonuses = sum(int(instance.update(Pose2D(3.2, 0.8, 0.0)).bonus_granted) for _ in range(20))

    assert bonuses == 1


def test_capture_direction_follows_the_previous_waypoint():
    instance = manager(width=1.0, count=2)
    for _ in range(EpisodeConfig().waypoint_hold_steps):
        instance.update(Pose2D(3.0, 0.0, 0.0))
    assert instance.index == 1

    # Between waypoint 1 and 2 the approach direction is 3.0 -> 6.0, so a pose
    # at x=4.0 has not crossed waypoint 2's plane.
    assert not instance.update(Pose2D(4.0, 0.7, 0.0)).reached
    assert instance.update(Pose2D(6.2, 0.7, 0.0)).reached


def test_start_pose_anchors_the_first_waypoint_direction():
    instance = WaypointManager(WAYPOINTS[:1], EpisodeConfig(waypoint_capture_width=1.0))
    instance.reset(start_xy=(0.0, 0.0))

    assert instance.approach_origin() == (0.0, 0.0)


# -------------------------------------------------------------- stage transfer


def test_every_stage_keeps_the_same_observation_and_action_shape():
    """A policy can only carry across stages if the spaces never change."""
    shapes = set()
    for count in (1, 2, 3):
        env = LimoWaypointRLEnv(config=stage_config(waypoint_count=count), enable_ros=False)
        shapes.add((env.observation_space.shape, env.action_space.shape))
        assert env.observation_space.shape == (observation_dim(env.config.observation),)
        env.close()

    assert shapes == {((112,), (2,))}


def test_a_single_waypoint_stage_finishes_at_the_first_waypoint():
    env = LimoWaypointRLEnv(config=stage_config(waypoint_count=1), enable_ros=False)
    env.reset(seed=42)

    info = {}
    for _ in range(2000):
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break

    assert info["reason"] == StopReason.SUCCESS.value
    summary = info["episode_summary"]
    assert summary["waypoints_reached"] == 1.0
    # One waypoint means one bonus and no switch discontinuity at all.
    assert summary["reward_waypoint"] == pytest.approx(20.0)
    assert summary["waypoint_switch_progress_sum"] == pytest.approx(0.0)
    env.close()


def test_a_shorter_stage_reaches_success_far_sooner():
    """The point of stage 1: the +100 lands 3 m out instead of 9.5 m out."""
    lengths = {}
    for count in (1, 3):
        env = LimoWaypointRLEnv(config=stage_config(waypoint_count=count), enable_ros=False)
        env.reset(seed=42)
        steps = 0
        for _ in range(4000):
            _o, _r, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            steps += 1
            if terminated or truncated:
                break
        assert info["reason"] == StopReason.SUCCESS.value
        lengths[count] = steps
        env.close()

    assert lengths[1] < lengths[3] / 2


# ------------------------------------------------------------- dwell removal


def test_a_hold_of_one_arrives_on_the_first_in_radius_step():
    """No dwell: entering the radius is arrival, so the robot never parks."""
    instance = WaypointManager(
        WAYPOINTS[:1], EpisodeConfig(waypoint_hold_steps=1)
    )
    instance.reset(start_xy=(0.0, 0.0))

    assert not instance.update(Pose2D(2.0, 0.0, 0.0)).reached      # outside 0.60 m
    update = instance.update(Pose2D(2.5, 0.0, 0.0))                 # inside
    assert update.reached
    assert update.bonus_granted


def test_a_hold_of_one_still_requires_being_inside_the_radius():
    instance = WaypointManager(WAYPOINTS[:1], EpisodeConfig(waypoint_hold_steps=1))
    instance.reset(start_xy=(0.0, 0.0))

    for x in (0.0, 1.0, 2.0, 2.39):
        assert not instance.update(Pose2D(x, 0.0, 0.0)).reached
    assert not instance.completed


def test_a_zero_hold_is_rejected_rather_than_arriving_from_anywhere():
    with pytest.raises(ValueError, match="waypoint_hold_steps"):
        WaypointManager(WAYPOINTS[:1], EpisodeConfig(waypoint_hold_steps=0))


def test_the_specification_default_still_requires_five_steps():
    assert EpisodeConfig().waypoint_hold_steps == 5


def test_no_dwell_stage_finishes_sooner_than_the_dwelling_default():
    lengths = {}
    for hold in (1, 5):
        env = LimoWaypointRLEnv(
            config=stage_config(waypoint_count=3, waypoint_hold_steps=hold), enable_ros=False
        )
        env.reset(seed=42)
        steps = 0
        for _ in range(4000):
            _o, _r, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            steps += 1
            if terminated or truncated:
                break
        assert info["reason"] == StopReason.SUCCESS.value
        lengths[hold] = steps
        env.close()

    assert lengths[1] <= lengths[5]


# ------------------------------------------------------- obstacle randomisation

OBSTACLE_HALF = 0.125
ROBOT_HALF_WIDTH = 0.0875   # base_y_size/2 + wheel_length/2
CORRIDOR = (-1.4, 1.05)     # measured free-space band in map.pgm for x in [0, 12]


def sampled_obstacles(seeds=range(60), **stage):
    """Collect the obstacle poses the env draws across many episodes."""
    env = LimoWaypointRLEnv(config=stage_config(**stage), enable_ros=False)
    poses = []
    for seed in seeds:
        env.reset(seed=seed)
        poses.extend((spec.x, spec.y) for spec in env.episode_obstacles)
    env.close()
    return poses


def test_obstacles_are_fixed_unless_randomisation_is_requested():
    poses = set(sampled_obstacles(waypoint_count=1))

    assert poses == {(2.07, -0.18)}
    assert DeliveryEnvConfig().obstacles.randomize is False


def test_randomised_poses_stay_inside_the_configured_ranges():
    obstacles = DeliveryEnvConfig().obstacles
    poses = sampled_obstacles(waypoint_count=1, obstacles_randomized=True)

    assert len(poses) >= 60
    for x, y in poses:
        assert obstacles.x_range[0] <= x <= obstacles.x_range[1]
        assert obstacles.y_range[0] <= y <= obstacles.y_range[1]


def test_randomised_poses_stay_clear_of_the_corridor_walls():
    for _x, y in sampled_obstacles(waypoint_count=1, obstacles_randomized=True):
        assert CORRIDOR[0] < y - OBSTACLE_HALF
        assert y + OBSTACLE_HALF < CORRIDOR[1]


def test_every_sampled_pose_still_blocks_the_centreline():
    """Otherwise the robot could drive straight through and learn nothing."""
    for _x, y in sampled_obstacles(waypoint_count=1, obstacles_randomized=True):
        assert abs(y) - OBSTACLE_HALF < ROBOT_HALF_WIDTH


def test_randomisation_produces_obstacles_on_both_sides_of_the_path():
    ys = [y for _x, y in sampled_obstacles(waypoint_count=1, obstacles_randomized=True)]

    assert sum(1 for y in ys if y < 0) >= 10, "no left-hand avoidance episodes"
    assert sum(1 for y in ys if y > 0) >= 10, "no right-hand avoidance episodes"


def test_randomisation_is_reproducible_for_a_fixed_seed():
    first = sampled_obstacles(seeds=[7], waypoint_count=1, obstacles_randomized=True)
    second = sampled_obstacles(seeds=[7], waypoint_count=1, obstacles_randomized=True)

    assert first == second


def test_disabled_obstacles_beat_randomisation():
    assert sampled_obstacles(
        waypoint_count=1, obstacles_enabled=False, obstacles_randomized=True
    ) == []


def test_the_randomise_flag_reaches_the_stage_config():
    args = build_parser().parse_args(["--randomize-obstacles"])

    assert build_stage_config(args).obstacles.randomize is True
    assert build_stage_config(build_parser().parse_args([])).obstacles.randomize is False
