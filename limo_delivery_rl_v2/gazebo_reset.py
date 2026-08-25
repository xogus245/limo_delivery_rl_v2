"""Gazebo reset sequencing and training-obstacle lifecycle.

``/reset_world`` is deliberately not used as the default reset: rewinding the
simulation clock makes tf2 reject every buffered transform.  ``/set_entity_state``
teleports the robot without a time jump.
"""

from __future__ import annotations

import time

from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from nav2_msgs.srv import ClearEntireCostmap
from std_srvs.srv import Empty

from limo_delivery_rl_v2.geometry import yaw_to_quaternion_z_w
from limo_delivery_rl_v2.ros_bridge import RosBridgeNode
from limo_delivery_rl_v2.state import DeliveryEnvConfig, ObstacleSpec


class GazeboResetError(RuntimeError):
    """Raised when the simulator cannot be returned to a valid episode start."""


def obstacle_sdf(spec: ObstacleSpec) -> str:
    """Return the SDF for one static box obstacle."""
    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{spec.name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>{spec.size_x} {spec.size_y} {spec.height}</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{spec.size_x} {spec.size_y} {spec.height}</size></box>
        </geometry>
        <material>
          <ambient>0.9 0.1 0.1 1</ambient>
          <diffuse>0.9 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def populate_set_entity_state_request(
    request, name: str, reference_frame: str, pose: tuple[float, float, float]
) -> None:
    """Fill a ``SetEntityState`` request with a pose and a fully zeroed twist."""
    request.state.name = name
    request.state.reference_frame = reference_frame
    request.state.pose.position.x = float(pose[0])
    request.state.pose.position.y = float(pose[1])
    request.state.pose.position.z = 0.0
    request.state.pose.orientation.x = 0.0
    request.state.pose.orientation.y = 0.0
    (
        request.state.pose.orientation.z,
        request.state.pose.orientation.w,
    ) = yaw_to_quaternion_z_w(float(pose[2]))
    for axis in ("x", "y", "z"):
        setattr(request.state.twist.linear, axis, 0.0)
        setattr(request.state.twist.angular, axis, 0.0)


class ObstacleManager:
    """Spawns and deletes training obstacles.

    Obstacles are added only *after* the global path has been planned, so the
    static-map path never routes around them; avoidance has to come from
    ``/scan``.
    """

    def __init__(self, node: RosBridgeNode, config: DeliveryEnvConfig) -> None:
        """Bind the manager to the bridge node and obstacle configuration."""
        self._node = node
        self._config = config
        self._timeout = config.gazebo.service_timeout_sec

    @property
    def specs(self) -> tuple[ObstacleSpec, ...]:
        """Obstacles that this manager owns."""
        return self._config.obstacles.specs if self._config.obstacles.enabled else ()

    def spawn_all(self) -> None:
        """Spawn every configured obstacle, raising on failure."""
        for spec in self.specs:
            request = SpawnEntity.Request()
            request.name = spec.name
            request.xml = obstacle_sdf(spec)
            request.robot_namespace = ""
            request.initial_pose.position.x = float(spec.x)
            request.initial_pose.position.y = float(spec.y)
            request.initial_pose.position.z = float(spec.height) / 2.0
            (
                request.initial_pose.orientation.z,
                request.initial_pose.orientation.w,
            ) = yaw_to_quaternion_z_w(float(spec.yaw))
            request.reference_frame = self._config.gazebo.reference_frame
            response = self._node.call_service(
                self._node.spawn_entity_client, request, self._timeout
            )
            if response is None or not response.success:
                raise GazeboResetError(f"failed to spawn obstacle '{spec.name}'")

    def remove_all(self) -> None:
        """Delete every configured obstacle, tolerating ones that do not exist."""
        for spec in self.specs:
            request = DeleteEntity.Request()
            request.name = spec.name
            response = self._node.call_service(
                self._node.delete_entity_client, request, self._timeout
            )
            if response is None:
                raise GazeboResetError(f"delete_entity timed out for '{spec.name}'")
            if response.success:
                continue
            status = str(getattr(response, "status_message", "")).lower()
            if not any(token in status for token in ("not exist", "does not exist", "not found")):
                raise GazeboResetError(
                    f"failed to delete obstacle '{spec.name}': {response.status_message}"
                )


class GazeboResetManager:
    """Executes the deterministic reset sequence for one episode start."""

    def __init__(
        self,
        node: RosBridgeNode,
        config: DeliveryEnvConfig,
        obstacles: ObstacleManager,
    ) -> None:
        """Bind the manager to the bridge node, config and obstacle manager."""
        self._node = node
        self._config = config
        self._obstacles = obstacles
        self._timeout = config.gazebo.service_timeout_sec

    def hold_zero_command(self, duration_sec: float) -> None:
        """Publish ``(0, 0)`` repeatedly so the drive plugin cannot latch an old command."""
        deadline = time.monotonic() + duration_sec
        while True:
            self._node.publish_cmd_vel(0.0, 0.0)
            if time.monotonic() >= deadline:
                return
            time.sleep(0.02)

    def pause(self) -> None:
        """Pause Gazebo physics."""
        if self._node.call_service(self._node.pause_client, Empty.Request(), self._timeout) is None:
            raise GazeboResetError("failed to pause Gazebo physics")

    def unpause(self) -> None:
        """Resume Gazebo physics."""
        if (
            self._node.call_service(self._node.unpause_client, Empty.Request(), self._timeout)
            is None
        ):
            raise GazeboResetError("failed to unpause Gazebo physics")

    def teleport_robot(self, pose: tuple[float, float, float]) -> None:
        """Set the robot pose and zero its twist in one ``/set_entity_state`` call."""
        request = SetEntityState.Request()
        populate_set_entity_state_request(
            request,
            self._config.gazebo.entity_name,
            self._config.gazebo.reference_frame,
            pose,
        )
        response = self._node.call_service(
            self._node.set_entity_state_client, request, self._timeout
        )
        if response is None or not response.success:
            raise GazeboResetError("failed to reset the robot pose via /set_entity_state")

    def clear_global_costmap(self) -> None:
        """Clear the global costmap if that service is actually running.

        In a planner-only launch the local costmap does not exist, so it is never
        waited on.
        """
        self._node.call_service(
            self._node.global_costmap_client,
            ClearEntireCostmap.Request(),
            self._timeout,
            required=False,
        )

    def wait_for_pose_match(self, pose: tuple[float, float, float]) -> None:
        """Block until ``map->base_link`` agrees with the commanded reset pose."""
        tolerance = self._config.gazebo.pose_settle_tolerance
        deadline = time.monotonic() + self._config.gazebo.pose_settle_timeout_sec
        best = float("inf")
        while time.monotonic() < deadline:
            snapshot = self._node.pose_provider.snapshot()
            if snapshot is not None:
                error = float(
                    ((snapshot.pose.x - pose[0]) ** 2 + (snapshot.pose.y - pose[1]) ** 2) ** 0.5
                )
                best = min(best, error)
                if error <= tolerance:
                    return
            time.sleep(0.02)
        raise GazeboResetError(
            f"map->base_link never matched the reset pose (best error {best:.3f} m > {tolerance} m)"
        )

    def reset_robot(self, pose: tuple[float, float, float]) -> None:
        """Run reset steps 1-10: stop, teleport, settle and clear stale Nav2 state."""
        hold = self._config.gazebo.zero_command_hold_sec
        self.hold_zero_command(hold)
        self.pause()
        self._obstacles.remove_all()
        self.teleport_robot(pose)
        self.unpause()
        self.hold_zero_command(hold)
        self.pause()
        self.teleport_robot(pose)
        self.unpause()
        if self._config.localization.use_amcl:
            self._reset_amcl(pose)
        else:
            self._node.validate_map_to_odom(self._timeout)
        self.clear_global_costmap()
        self.wait_for_pose_match(pose)

    def _reset_amcl(self, pose: tuple[float, float, float]) -> None:
        """Hardware-only branch: seed AMCL and force a no-motion update.

        Never reached during training, where ``use_amcl`` is ``False``.
        """
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav2_msgs.srv import SetInitialPose

        request = SetInitialPose.Request()
        message = PoseWithCovarianceStamped()
        message.header.frame_id = self._config.frames.map_frame
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.pose.pose.position.x = float(pose[0])
        message.pose.pose.position.y = float(pose[1])
        (
            message.pose.pose.orientation.z,
            message.pose.pose.orientation.w,
        ) = yaw_to_quaternion_z_w(float(pose[2]))
        message.pose.covariance[0] = 0.25
        message.pose.covariance[7] = 0.25
        message.pose.covariance[35] = 0.06853892326654787
        request.pose = message
        if self._node.call_service(self._node.initial_pose_client, request, self._timeout) is None:
            raise GazeboResetError("failed to seed AMCL via /set_initial_pose")
        if (
            self._node.call_service(
                self._node.nomotion_update_client, Empty.Request(), self._timeout
            )
            is None
        ):
            raise GazeboResetError("failed to trigger /request_nomotion_update")
