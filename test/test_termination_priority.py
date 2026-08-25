"""Termination conditions and their priority order."""

import pytest

from limo_delivery_rl_v2.state import EpisodeConfig, SafetyConfig, StopReason
from limo_delivery_rl_v2.termination import TerminationContext, termination_status

EPISODE = EpisodeConfig()
SAFETY = SafetyConfig()


def status(**changes):
    """Evaluate termination for a healthy, mid-episode context plus overrides."""
    values = {
        "min_obstacle_distance": 8.0,
        "waypoints_completed": False,
        "path_available": True,
        "sensor_lost": False,
        "cross_track_error": 0.0,
        "steps": 1,
        "stuck_steps": 0,
    }
    values.update(changes)
    return termination_status(TerminationContext(**values), EPISODE, SAFETY)


def reason(**changes) -> StopReason:
    """Return only the stop reason."""
    return status(**changes)[2]


def test_healthy_step_continues():
    assert status() == (False, False, StopReason.NONE)


def test_each_condition_is_detected():
    assert reason(min_obstacle_distance=0.25) is StopReason.COLLISION
    assert reason(waypoints_completed=True) is StopReason.SUCCESS
    assert reason(path_available=False) is StopReason.PATH_FAILED
    assert reason(sensor_lost=True) is StopReason.SENSOR_TIMEOUT
    assert reason(cross_track_error=2.51) is StopReason.PATH_DEVIATION
    assert reason(cross_track_error=-2.51) is StopReason.PATH_DEVIATION
    assert reason(stuck_steps=300) is StopReason.STUCK
    assert reason(steps=5000) is StopReason.TIMEOUT


def test_only_collision_and_success_are_mdp_terminations():
    assert status(min_obstacle_distance=0.25)[:2] == (True, False)
    assert status(waypoints_completed=True)[:2] == (True, False)
    for truncating in (
        {"path_available": False},
        {"sensor_lost": True},
        {"cross_track_error": 3.0},
        {"stuck_steps": 300},
        {"steps": 5000},
    ):
        assert status(**truncating)[:2] == (False, True), truncating


def test_collision_outranks_every_other_condition():
    assert reason(
        min_obstacle_distance=0.2,
        waypoints_completed=True,
        path_available=False,
        sensor_lost=True,
        cross_track_error=9.0,
        stuck_steps=999,
        steps=99999,
    ) is StopReason.COLLISION


def test_success_outranks_every_failure():
    assert reason(
        waypoints_completed=True,
        path_available=False,
        sensor_lost=True,
        cross_track_error=9.0,
        stuck_steps=999,
        steps=99999,
    ) is StopReason.SUCCESS


def test_priority_order_is_exact():
    ladder = [
        ({"min_obstacle_distance": 0.2}, StopReason.COLLISION),
        ({"waypoints_completed": True}, StopReason.SUCCESS),
        ({"path_available": False}, StopReason.PATH_FAILED),
        ({"sensor_lost": True}, StopReason.SENSOR_TIMEOUT),
        ({"cross_track_error": 9.0}, StopReason.PATH_DEVIATION),
        ({"stuck_steps": 999}, StopReason.STUCK),
        ({"steps": 99999}, StopReason.TIMEOUT),
    ]
    for index in range(len(ladder)):
        combined: dict = {}
        for changes, _ in ladder[index:]:
            combined.update(changes)
        assert reason(**combined) is ladder[index][1]


def test_thresholds_match_the_specification():
    assert SAFETY.collision_distance == pytest.approx(0.25)
    assert SAFETY.max_path_deviation == pytest.approx(2.50)
    assert EPISODE.max_steps == 5000                 # 250 s at 20 Hz
    assert EPISODE.stuck_step_limit == 300           # 15 s at 20 Hz
    assert EPISODE.stuck_speed_threshold == pytest.approx(0.016)


def test_boundaries_are_inclusive_where_the_specification_says_so():
    assert reason(min_obstacle_distance=0.2500001) is StopReason.NONE
    assert reason(cross_track_error=2.50) is StopReason.NONE
    assert reason(stuck_steps=299) is StopReason.NONE
    assert reason(steps=4999) is StopReason.NONE
