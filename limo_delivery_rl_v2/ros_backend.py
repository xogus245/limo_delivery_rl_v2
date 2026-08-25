"""ROS 2 implementation of :class:`~limo_delivery_rl_v2.env_backend.EnvBackend`.

Implements the full episode reset order and the per-step command/sense cycle.
The control loop is paced by ``/scan`` (20 Hz), so one environment step consumes
exactly one control period of simulation time regardless of the real-time factor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from limo_delivery_rl_v2.env_backend import BackendFrame
from limo_delivery_rl_v2.gazebo_reset import GazeboResetManager, ObstacleManager
from limo_delivery_rl_v2.ros_bridge import Nav2PlanningError, RosBridge
from limo_delivery_rl_v2.state import DeliveryEnvConfig


class RosBackend:
    """Drives Gazebo, Nav2 and the sensors for one training environment."""

    def __init__(self, config: DeliveryEnvConfig) -> None:
        """Start the bridge and run the start-up contract checks."""
        self._config = config
        self._bridge = RosBridge(config)
        self._node = self._bridge.node
        self._obstacles = ObstacleManager(self._node, config)
        self._reset_manager = GazeboResetManager(self._node, config, self._obstacles)
        self._node.validate_cmd_vel_ownership()
        self._node.validate_node_graph()
        if not config.localization.use_amcl:
            self._node.validate_map_to_odom(config.gazebo.service_timeout_sec)
        self._node.pose_provider.wait_for_pose(config.gazebo.service_timeout_sec)

    @property
    def node(self):
        """The underlying bridge node (exposed for diagnostics and tests)."""
        return self._node

    def reset_episode(
        self,
        start_pose: tuple[float, float, float],
        waypoints: Sequence[tuple[float, float, float]],
    ) -> tuple[tuple[tuple[float, float], ...], BackendFrame]:
        """Run the full reset order and return the planned path and first frame."""
        self._reset_manager.reset_robot(start_pose)

        # Steps 11-12: drop everything sensed before the teleport, then require
        # one fresh scan and one fresh odom before planning.
        boundary = self._node.now_sim_sec()
        self._node.discard_before(boundary)
        scan, odom = self._node.wait_for_sensors(
            0, 0, self._config.gazebo.sensor_refresh_timeout_sec
        )
        if scan is None or odom is None:
            raise Nav2PlanningError(
                "no /scan or /odom received after the reset; cannot plan a path"
            )

        # Steps 13-14: plan through every waypoint from the settled pose. The
        # frame contract is validated inside the path provider.
        snapshot = self._node.pose_provider.snapshot()
        planning_start = (
            (snapshot.pose.x, snapshot.pose.y, snapshot.pose.yaw)
            if snapshot is not None
            else start_pose
        )
        path = self._node.path_provider.request_path(
            waypoints, planning_start, self._config.gazebo.path_request_timeout_sec
        )

        # Steps 15-16: obstacles appear only now, and the first observation must
        # already show them.
        self._obstacles.spawn_all()
        before_scan, before_odom = self._node.sensor_sequences()
        self._node.wait_for_sensors(
            before_scan, before_odom, self._config.gazebo.sensor_refresh_timeout_sec
        )
        return path, self._frame(fresh_required=False)

    def apply_command(self, linear: float, angular: float) -> BackendFrame:
        """Publish one command and wait for the next scan/odom pair."""
        before_scan, before_odom = self._node.sensor_sequences()
        self._node.publish_cmd_vel(linear, angular)
        scan, odom = self._node.wait_for_sensors(
            before_scan, before_odom, self._config.gazebo.sensor_refresh_timeout_sec
        )
        stalled = (
            scan is None
            or odom is None
            or scan.seq <= before_scan
            or odom.seq <= before_odom
        )
        return self._frame(fresh_required=stalled)

    def close(self) -> None:
        """Stop the robot and shut the bridge down."""
        try:
            self._node.publish_cmd_vel(0.0, 0.0)
        finally:
            self._bridge.close()

    def _frame(self, *, fresh_required: bool) -> BackendFrame:
        """Assemble a :class:`BackendFrame` from the current snapshots.

        ``fresh_required`` marks a step whose sensor wait timed out on wall time;
        the ages are then forced to infinity so a frozen ``/clock`` cannot hide a
        dead sensor behind a simulation time that never advances.
        """
        now = self._node.now_sim_sec()
        scan = self._node.latest_scan()
        odom = self._node.latest_odom()
        pose_snapshot = self._node.pose_provider.snapshot()
        bins = (
            scan.bins_metres
            if scan is not None
            else np.full(
                self._config.observation.lidar_bins,
                self._config.observation.lidar_max_range,
                dtype=np.float32,
            )
        )
        infinite = float("inf")
        return BackendFrame(
            pose=pose_snapshot.pose if pose_snapshot is not None else None,
            linear_velocity=odom.linear_velocity if odom is not None else 0.0,
            angular_velocity=odom.angular_velocity if odom is not None else 0.0,
            lidar_metres=bins,
            scan_age=infinite if (fresh_required or scan is None) else max(0.0, now - scan.stamp_sec),
            odom_age=infinite if (fresh_required or odom is None) else max(0.0, now - odom.stamp_sec),
            tf_age=self._node.pose_provider.age(now),
        )
