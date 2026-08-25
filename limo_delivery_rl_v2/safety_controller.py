"""Hard safety gate applied between the policy action and ``/cmd_vel``.

Deliberately absent: any distance-proportional slowdown.  The policy is
responsible for choosing a safe speed; the gate only enforces bounds that must
hold regardless of what the policy asked for.
"""

from __future__ import annotations

import math

import numpy as np

from limo_delivery_rl_v2.action_scaler import RateLimiter
from limo_delivery_rl_v2.state import ControlConfig, SafetyConfig, SafetyMode

#: Nodes that must never own ``/cmd_vel`` while the RL bridge is running.
FORBIDDEN_CMD_VEL_PUBLISHERS: frozenset[str] = frozenset(
    {
        "behavior_server",
        "bt_navigator",
        "controller_server",
        "velocity_smoother",
        "waypoint_follower",
    }
)


def sensor_data_is_fresh(scan_age: float, odom_age: float, config: SafetyConfig) -> bool:
    """Whether both ``/scan`` and ``/odom`` are within the timeout window."""
    return scan_age <= config.sensor_timeout_sec and odom_age <= config.sensor_timeout_sec


def pose_is_fresh(tf_age: float, config: SafetyConfig) -> bool:
    """Whether the ``map->base_link`` lookup succeeded recently enough."""
    return tf_age <= config.tf_timeout_sec


def apply_safety_limits(
    *,
    linear_target: float,
    angular_target: float,
    previous_linear: float,
    previous_angular: float,
    min_obstacle_distance: float,
    scan_age: float,
    odom_age: float,
    tf_age: float,
    action_valid: bool,
    control: ControlConfig,
    safety: SafetyConfig,
    rate_limiter: RateLimiter | None = None,
) -> tuple[float, float, SafetyMode]:
    """Clamp a policy command, returning ``(linear, angular, mode)``.

    Hard-stop conditions return exactly ``(0.0, 0.0)`` with no acceleration ramp,
    because every one of them means the command cannot be trusted at all.  In the
    nominal path the command is clipped to the absolute speed envelope and then
    acceleration limited.
    """
    if not action_valid or not math.isfinite(linear_target) or not math.isfinite(angular_target):
        return 0.0, 0.0, SafetyMode.INVALID_ACTION
    if not sensor_data_is_fresh(scan_age, odom_age, safety):
        return 0.0, 0.0, SafetyMode.SENSOR_TIMEOUT
    if not pose_is_fresh(tf_age, safety):
        return 0.0, 0.0, SafetyMode.TF_TIMEOUT
    if min_obstacle_distance <= safety.collision_distance:
        return 0.0, 0.0, SafetyMode.IMMINENT_COLLISION

    linear = float(np.clip(linear_target, 0.0, control.max_linear_speed))
    angular = float(np.clip(angular_target, -control.max_angular_speed, control.max_angular_speed))
    limiter = rate_limiter or RateLimiter(control)
    linear, angular = limiter.limit(linear, angular, previous_linear, previous_angular)
    return linear, angular, SafetyMode.OK


def find_forbidden_publishers(publisher_names: list[str]) -> list[str]:
    """Return the sorted subset of ``publisher_names`` that must not own ``/cmd_vel``."""
    return sorted(set(publisher_names) & FORBIDDEN_CMD_VEL_PUBLISHERS)
