"""Backend abstraction separating the Gym environment from ROS 2 I/O.

``LimoWaypointRLEnv`` talks to a backend that supplies a global path plus, once
per control step, a :class:`BackendFrame`.  :class:`OfflineBackend` implements
that contract with a deterministic unicycle model so the reward, observation and
termination logic can be unit tested without a running simulator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from limo_delivery_rl_v2.geometry import Pose2D
from limo_delivery_rl_v2.state import DeliveryEnvConfig, ObstacleSpec


@dataclass(frozen=True, slots=True)
class BackendFrame:
    """One control step's worth of sensing.

    ``pose`` is ``None`` when the ``map->base_link`` lookup is currently failing;
    ``tf_age`` then reports how long it has been unavailable.
    """

    pose: Pose2D | None
    linear_velocity: float
    angular_velocity: float
    lidar_metres: NDArray[np.float32]
    scan_age: float
    odom_age: float
    tf_age: float


class EnvBackend(Protocol):
    """Interface the environment requires from a simulation or ROS backend."""

    def reset_episode(
        self,
        start_pose: tuple[float, float, float],
        waypoints: Sequence[tuple[float, float, float]],
        obstacles: Sequence[ObstacleSpec],
    ) -> tuple[tuple[tuple[float, float], ...], BackendFrame]:
        """Reset the robot, plan the path and spawn ``obstacles`` afterwards."""

    def apply_command(self, linear: float, angular: float) -> BackendFrame:
        """Publish one command and return the resulting frame one control step later."""

    def close(self) -> None:
        """Release any resources held by the backend."""


def straight_path_through(
    start: tuple[float, float],
    waypoints: Sequence[tuple[float, float, float]],
    spacing: float = 0.05,
) -> tuple[tuple[float, float], ...]:
    """Densely sample the straight polyline from ``start`` through every waypoint."""
    corners = [start] + [(x, y) for x, y, _ in waypoints]
    points: list[tuple[float, float]] = [corners[0]]
    for begin, end in zip(corners, corners[1:]):
        distance = math.hypot(end[0] - begin[0], end[1] - begin[1])
        steps = max(1, int(math.ceil(distance / max(spacing, 1e-6))))
        for index in range(1, steps + 1):
            ratio = index / steps
            points.append(
                (
                    begin[0] + (end[0] - begin[0]) * ratio,
                    begin[1] + (end[1] - begin[1]) * ratio,
                )
            )
    return tuple(points)


class OfflineBackend:
    """Deterministic unicycle simulator used for tests and ``--no-ros`` smoke runs.

    Sensors are perfect and always fresh, and the LiDAR reports free space
    everywhere, so an offline episode exercises path following and waypoint
    bookkeeping without any obstacle interaction.
    """

    def __init__(self, config: DeliveryEnvConfig) -> None:
        """Bind the simulator to the shared environment configuration."""
        self._config = config
        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._linear = 0.0
        self._angular = 0.0

    @property
    def pose(self) -> Pose2D:
        """Current simulated ``map->base_link`` pose."""
        return self._pose

    def reset_episode(
        self,
        start_pose: tuple[float, float, float],
        waypoints: Sequence[tuple[float, float, float]],
        obstacles: Sequence[ObstacleSpec] = (),
    ) -> tuple[tuple[tuple[float, float], ...], BackendFrame]:
        """Teleport the robot and return a straight-line stand-in for the Nav2 path.

        Obstacles are accepted for interface parity; the offline LiDAR always
        reports free space.
        """
        self._obstacles = tuple(obstacles)
        self._pose = Pose2D(*start_pose)
        self._linear = 0.0
        self._angular = 0.0
        path = straight_path_through((start_pose[0], start_pose[1]), waypoints)
        return path, self._frame()

    def apply_command(self, linear: float, angular: float) -> BackendFrame:
        """Integrate the unicycle model for one control period."""
        dt = self._config.control.control_dt
        yaw = self._pose.yaw + angular * dt
        self._pose = Pose2D(
            self._pose.x + linear * math.cos(self._pose.yaw) * dt,
            self._pose.y + linear * math.sin(self._pose.yaw) * dt,
            yaw,
        )
        self._linear = linear
        self._angular = angular
        return self._frame()

    def close(self) -> None:
        """No resources to release."""

    def _frame(self) -> BackendFrame:
        """Build a frame reporting free space and perfectly fresh sensors."""
        return BackendFrame(
            pose=self._pose,
            linear_velocity=self._linear,
            angular_velocity=self._angular,
            lidar_metres=np.full(
                self._config.observation.lidar_bins,
                self._config.observation.lidar_max_range,
                dtype=np.float32,
            ),
            scan_age=0.0,
            odom_age=0.0,
            tf_age=0.0,
        )
