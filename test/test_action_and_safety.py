"""Action scaling, acceleration limiting and the hard safety gate."""

import numpy as np
import pytest

from limo_delivery_rl_v2.action_scaler import (
    ACTION_DIM,
    ActionScaler,
    RateLimiter,
    sanitize_action,
)
from limo_delivery_rl_v2.safety_controller import (
    apply_safety_limits,
    find_forbidden_publishers,
)
from limo_delivery_rl_v2.state import ControlConfig, SafetyConfig, SafetyMode

CONTROL = ControlConfig()
SAFETY = SafetyConfig()


def gate(**changes):
    """Call the safety gate with nominal, fully healthy defaults."""
    values = {
        "linear_target": 0.2,
        "angular_target": 0.0,
        "previous_linear": 0.2,
        "previous_angular": 0.0,
        "min_obstacle_distance": 8.0,
        "scan_age": 0.0,
        "odom_age": 0.0,
        "tf_age": 0.0,
        "action_valid": True,
        "control": CONTROL,
        "safety": SAFETY,
    }
    values.update(changes)
    return apply_safety_limits(**values)


# ------------------------------------------------------------------ scaling


def test_action_space_is_two_dimensional():
    assert ACTION_DIM == 2


@pytest.mark.parametrize(
    ("action", "expected_v", "expected_w"),
    [
        ((-1.0, -1.0), 0.0, -0.9),
        ((0.0, 0.0), 0.21, 0.0),
        ((1.0, 1.0), 0.42, 0.9),
    ],
)
def test_scaling_maps_the_unit_box_onto_the_command_envelope(action, expected_v, expected_w):
    linear, angular = ActionScaler(CONTROL).scale(np.array(action, dtype=np.float32))

    assert linear == pytest.approx(expected_v)
    assert angular == pytest.approx(expected_w)


def test_scaling_never_commands_reverse_or_exceeds_the_training_speed():
    scaler = ActionScaler(CONTROL)

    for value in np.linspace(-1.0, 1.0, 41):
        linear, angular = scaler.scale(np.array([value, value], dtype=np.float32))
        assert 0.0 <= linear <= CONTROL.max_train_linear_speed + 1e-9
        assert abs(angular) <= CONTROL.max_angular_speed + 1e-9


def test_sanitize_rejects_nan_inf_and_wrong_shapes():
    for bad in ([np.nan, 0.0], [0.0, np.inf], [0.0], [0.0, 0.0, 0.0], []):
        value, valid = sanitize_action(np.array(bad, dtype=np.float32))
        assert not valid
        assert value.tolist() == [0.0, 0.0]


def test_sanitize_clips_out_of_range_actions():
    value, valid = sanitize_action(np.array([5.0, -5.0], dtype=np.float32))

    assert valid
    assert value.tolist() == [1.0, -1.0]


# ------------------------------------------------------- acceleration limits


def test_step_limits_match_the_specified_accelerations():
    limiter = RateLimiter(CONTROL)

    assert limiter.max_linear_step == pytest.approx(0.05)   # 1.0 m/s^2 * 0.05 s
    assert limiter.max_angular_step == pytest.approx(0.09)  # 1.8 rad/s^2 * 0.05 s


def test_rate_limiter_caps_acceleration_and_deceleration():
    limiter = RateLimiter(CONTROL)

    assert limiter.limit(0.42, 0.9, 0.0, 0.0) == pytest.approx((0.05, 0.09))
    assert limiter.limit(0.0, -0.9, 0.42, 0.0) == pytest.approx((0.37, -0.09))


def test_command_ramp_never_exceeds_the_step_limits_over_a_trajectory():
    limiter = RateLimiter(CONTROL)
    linear, angular = 0.0, 0.0
    targets = [(0.42, 0.9), (0.0, -0.9), (0.42, 0.0), (0.1, 0.5)]

    for target in targets:
        for _ in range(30):
            next_linear, next_angular = limiter.limit(*target, linear, angular)
            assert abs(next_linear - linear) <= limiter.max_linear_step + 1e-9
            assert abs(next_angular - angular) <= limiter.max_angular_step + 1e-9
            linear, angular = next_linear, next_angular


# ---------------------------------------------------------------- safety gate


def test_invalid_action_forces_a_full_stop():
    assert gate(action_valid=False) == (0.0, 0.0, SafetyMode.INVALID_ACTION)
    assert gate(linear_target=float("nan")) == (0.0, 0.0, SafetyMode.INVALID_ACTION)


def test_stale_scan_or_odom_forces_a_full_stop():
    assert gate(scan_age=0.51)[2] is SafetyMode.SENSOR_TIMEOUT
    assert gate(odom_age=0.51)[2] is SafetyMode.SENSOR_TIMEOUT
    assert gate(scan_age=0.50)[2] is SafetyMode.OK


def test_stale_tf_forces_a_full_stop():
    assert gate(tf_age=0.51)[2] is SafetyMode.TF_TIMEOUT
    assert gate(tf_age=float("inf"))[2] is SafetyMode.TF_TIMEOUT
    assert gate(tf_age=0.50)[2] is SafetyMode.OK


def test_imminent_collision_forces_a_full_stop():
    linear, angular, mode = gate(min_obstacle_distance=0.25)

    assert (linear, angular) == (0.0, 0.0)
    assert mode is SafetyMode.IMMINENT_COLLISION


def test_collision_threshold_catches_the_simulated_lidar_floor():
    # Gazebo publishes its 0.20 m floor as the next float above 0.20.
    assert gate(min_obstacle_distance=0.20000000298023224)[2] is SafetyMode.IMMINENT_COLLISION


def test_speed_is_not_reduced_by_nearby_obstacles():
    # Distance-proportional slowdown is explicitly forbidden: identical commands
    # must survive the gate identically at 8 m and just above the collision line.
    far = gate(linear_target=0.42, previous_linear=0.42, min_obstacle_distance=8.0)
    near = gate(linear_target=0.42, previous_linear=0.42, min_obstacle_distance=0.26)

    assert far[0] == pytest.approx(near[0])
    assert near[0] == pytest.approx(0.42)
    assert near[2] is SafetyMode.OK


def test_absolute_speed_ceiling_is_enforced():
    linear, angular, mode = gate(
        linear_target=5.0, angular_target=5.0, previous_linear=0.55, previous_angular=0.9
    )

    assert linear <= CONTROL.max_linear_speed + 1e-9
    assert angular <= CONTROL.max_angular_speed + 1e-9
    assert mode is SafetyMode.OK


def test_hard_stop_takes_priority_over_the_deceleration_ramp():
    linear, angular, mode = gate(min_obstacle_distance=0.1, previous_linear=0.42)

    assert (linear, angular) == (0.0, 0.0)
    assert mode.is_hard_stop


def test_forbidden_cmd_vel_publishers_are_detected():
    names = ["planner_server", "controller_server", "velocity_smoother", "limo_waypoint_rl_bridge"]

    assert find_forbidden_publishers(names) == ["controller_server", "velocity_smoother"]
    assert find_forbidden_publishers(["planner_server", "map_server"]) == []
