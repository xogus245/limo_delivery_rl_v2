"""Waypoint arrival: 5-step hold requirement, single bonus, ordered progression."""

import math

import pytest

from limo_delivery_rl_v2.geometry import Pose2D
from limo_delivery_rl_v2.state import EpisodeConfig, WAYPOINTS
from limo_delivery_rl_v2.waypoint_manager import WaypointManager


def manager(config: EpisodeConfig | None = None) -> WaypointManager:
    """A manager over the three fixed experiment waypoints."""
    return WaypointManager(WAYPOINTS, config or EpisodeConfig())


def test_default_configuration_matches_the_experiment_specification():
    config = EpisodeConfig()

    assert config.waypoint_radius == pytest.approx(0.60)
    assert config.waypoint_hold_steps == 5
    assert WAYPOINTS == ((3.0, 0.0, 0.0), (6.0, 0.0, 0.0), (9.5, 0.0, 0.0))


def test_arrival_requires_five_consecutive_steps_inside_the_radius():
    instance = manager()
    inside = Pose2D(3.0, 0.0, 0.0)

    for step in range(4):
        update = instance.update(inside)
        assert not update.reached, f"reached too early at step {step}"
        assert instance.index == 0

    update = instance.update(inside)
    assert update.reached
    assert update.bonus_granted
    assert instance.reached_count == 1


def test_leaving_the_radius_resets_the_hold_counter():
    instance = manager()
    inside = Pose2D(3.0, 0.0, 0.0)
    outside = Pose2D(1.0, 0.0, 0.0)

    for _ in range(4):
        instance.update(inside)
    instance.update(outside)
    assert instance.hold_steps == 0

    for _ in range(4):
        assert not instance.update(inside).reached
    assert instance.update(inside).reached


def test_bonus_is_granted_once_per_waypoint():
    instance = manager()
    inside = Pose2D(3.0, 0.0, 0.0)
    bonuses = 0

    for _ in range(40):
        update = instance.update(inside)
        bonuses += int(update.bonus_granted)

    assert bonuses == 1
    assert instance.reached_count == 1


def test_waypoints_advance_in_order_and_the_last_one_completes_the_run():
    instance = manager()
    switches = 0
    bonuses = 0

    for waypoint in WAYPOINTS:
        pose = Pose2D(waypoint[0], waypoint[1], 0.0)
        for _ in range(EpisodeConfig().waypoint_hold_steps):
            update = instance.update(pose)
        switches += int(update.switched)
        bonuses += int(update.bonus_granted)

    assert bonuses == 3
    assert switches == 2, "only the two intermediate waypoints switch"
    assert instance.completed
    assert instance.reached_count == 3


def test_a_completed_run_stops_granting_bonuses():
    instance = manager()
    for waypoint in WAYPOINTS:
        for _ in range(5):
            instance.update(Pose2D(waypoint[0], waypoint[1], 0.0))

    update = instance.update(Pose2D(9.5, 0.0, 0.0))

    assert update.completed
    assert not update.bonus_granted
    assert instance.reached_count == 3


def test_switching_reports_the_distance_to_the_new_waypoint():
    instance = manager()
    pose = Pose2D(3.0, 0.0, 0.0)

    for _ in range(5):
        update = instance.update(pose)

    assert update.switched
    # 3.0 -> 6.0 means the reported distance jumps to the next target.
    assert update.distance == pytest.approx(3.0)


def test_measure_returns_distance_and_signed_heading_error():
    instance = manager()

    distance, heading = instance.measure(Pose2D(0.0, 0.0, 0.0))
    assert distance == pytest.approx(3.0)
    assert heading == pytest.approx(0.0)

    _, heading = instance.measure(Pose2D(0.0, 0.0, math.pi / 2.0))
    assert heading == pytest.approx(-math.pi / 2.0)


def test_reset_restores_the_first_waypoint():
    instance = manager()
    for _ in range(5):
        instance.update(Pose2D(3.0, 0.0, 0.0))

    instance.reset()

    assert instance.index == 0
    assert instance.reached_count == 0
    assert not instance.completed


def test_empty_waypoint_list_is_rejected():
    with pytest.raises(ValueError, match="at least one waypoint"):
        WaypointManager((), EpisodeConfig())
