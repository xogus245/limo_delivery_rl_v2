"""Angle helpers and ``map`` <-> ``base_link`` coordinate transforms.

Every relative coordinate fed to the policy is produced here so that a single
pose snapshot is used consistently within one control step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose2D:
    """A planar pose in a named frame (the frame is tracked by the caller)."""

    x: float
    y: float
    yaw: float


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` into ``[-pi, pi)``."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract the yaw component of a quaternion."""
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_to_quaternion_z_w(yaw: float) -> tuple[float, float]:
    """Return the ``(z, w)`` quaternion components for a planar ``yaw``."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def map_to_base_link(point: tuple[float, float], pose: Pose2D) -> tuple[float, float]:
    """Express a ``map``-frame ``point`` in ``base_link`` coordinates.

    This is the inverse of the ``map->base_link`` transform whose translation is
    ``(pose.x, pose.y)`` and whose rotation is ``pose.yaw``.
    """
    dx = point[0] - pose.x
    dy = point[1] - pose.y
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return (cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy)


def base_link_to_map(point: tuple[float, float], pose: Pose2D) -> tuple[float, float]:
    """Express a ``base_link``-frame ``point`` in ``map`` coordinates."""
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return (
        pose.x + cos_yaw * point[0] - sin_yaw * point[1],
        pose.y + sin_yaw * point[0] + cos_yaw * point[1],
    )


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Planar distance between two points."""
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def heading_error_to(point: tuple[float, float], pose: Pose2D) -> float:
    """Signed bearing of ``point`` relative to the robot heading, wrapped to +-pi."""
    return wrap_angle(math.atan2(point[1] - pose.y, point[0] - pose.x) - pose.yaw)
