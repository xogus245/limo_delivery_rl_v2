"""ROS 2 bridge node: sensing, tf2 pose lookup, ``/cmd_vel`` and Nav2 planning.

All subscriptions, service clients and action clients share a
``ReentrantCallbackGroup`` and are serviced by a ``MultiThreadedExecutor`` that
spins on its own thread.  Nothing here calls ``spin_once`` or
``spin_until_future_complete``; the environment blocks on ``threading`` events
instead, which keeps a stalled callback from deadlocking the control loop.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import rclpy
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathThroughPoses
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry
from numpy.typing import NDArray
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from limo_delivery_rl_v2.geometry import Pose2D, yaw_from_quaternion, yaw_to_quaternion_z_w
from limo_delivery_rl_v2.lidar import LidarBinner, geometry_from_scan
from limo_delivery_rl_v2.safety_controller import find_forbidden_publishers
from limo_delivery_rl_v2.state import DeliveryEnvConfig


class Nav2PlanningError(RuntimeError):
    """Raised when Nav2 cannot deliver a usable global path."""


class FrameContractError(RuntimeError):
    """Raised when a message arrives in an unexpected frame."""


def stamp_to_seconds(stamp) -> float:
    """Convert a ``builtin_interfaces/Time`` message to float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass(frozen=True, slots=True)
class ScanSnapshot:
    """Latest ``/scan``, already reduced to per-bin minima in metres."""

    seq: int
    stamp_sec: float
    bins_metres: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class OdomSnapshot:
    """Latest ``/odom`` twist. The pose field of the message is never read."""

    seq: int
    stamp_sec: float
    linear_velocity: float
    angular_velocity: float


@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    """Latest successful ``map->base_link`` tf2 lookup."""

    stamp_sec: float
    pose: Pose2D


def scan_qos() -> QoSProfile:
    """Sensor-data QoS for ``/scan`` (BEST_EFFORT, VOLATILE, depth 5)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


def odom_qos() -> QoSProfile:
    """Sensor-data QoS for ``/odom`` (BEST_EFFORT, VOLATILE, depth 10)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


def cmd_vel_qos() -> QoSProfile:
    """Command QoS for ``/cmd_vel`` (RELIABLE, VOLATILE, depth 10)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class TfPoseProvider:
    """Thread-safe snapshot of the ``map->base_link`` transform.

    The robot pose used by observations, rewards and termination comes only from
    here.  ``/odom`` poses are never substituted, and a failed lookup is reported
    as such rather than silently reusing the previous value.
    """

    def __init__(self, node: Node, buffer: Buffer, map_frame: str, base_frame: str) -> None:
        """Bind the provider to a tf2 buffer and the two frames of interest."""
        self._node = node
        self._buffer = buffer
        self._map_frame = map_frame
        self._base_frame = base_frame
        self._lock = threading.Lock()
        self._snapshot: PoseSnapshot | None = None
        self._event = threading.Event()

    def poll(self) -> None:
        """Refresh the snapshot; called from a timer inside the executor.

        The lookup uses ``rclpy.time.Time()`` (latest available) evaluated on the
        simulation clock; wall time is never used.
        """
        try:
            transform = self._buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except TransformException:
            return
        rotation = transform.transform.rotation
        snapshot = PoseSnapshot(
            stamp_sec=stamp_to_seconds(transform.header.stamp),
            pose=Pose2D(
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w),
            ),
        )
        with self._lock:
            self._snapshot = snapshot
        self._event.set()

    def snapshot(self) -> PoseSnapshot | None:
        """Return the latest successful lookup, or ``None`` if there never was one."""
        with self._lock:
            return self._snapshot

    def age(self, now_sec: float) -> float:
        """Simulation-time age of the latest lookup; ``inf`` if none succeeded."""
        snapshot = self.snapshot()
        if snapshot is None:
            return float("inf")
        return max(0.0, now_sec - snapshot.stamp_sec)

    def wait_for_pose(self, timeout_sec: float) -> PoseSnapshot | None:
        """Block (on wall time) until at least one lookup has succeeded."""
        if self.snapshot() is None:
            self._event.wait(timeout_sec)
        return self.snapshot()


class Nav2ThroughPosesPathProvider:
    """Wraps the ``ComputePathThroughPoses`` action client.

    Only the planner is used; the controller server stays offline so nothing but
    this process publishes ``/cmd_vel``.
    """

    def __init__(
        self,
        node: Node,
        action_client: ActionClient,
        map_frame: str,
        planner_id: str,
    ) -> None:
        """Bind the provider to an action client and the required frame contract."""
        self._node = node
        self._client = action_client
        self._map_frame = map_frame
        self._planner_id = planner_id

    def request_path(
        self,
        goals: Sequence[tuple[float, float, float]],
        start: tuple[float, float, float] | None,
        timeout_sec: float,
    ) -> tuple[tuple[float, float], ...]:
        """Plan one path through every goal and return its ``map``-frame points.

        Raises :class:`Nav2PlanningError` on any failure and
        :class:`FrameContractError` if the returned path is not in ``map``.
        """
        if not self._client.wait_for_server(timeout_sec=timeout_sec):
            raise Nav2PlanningError("ComputePathThroughPoses action server is unavailable")
        goal_msg = ComputePathThroughPoses.Goal()
        goal_msg.goals = [self._pose_stamped(goal) for goal in goals]
        goal_msg.planner_id = self._planner_id
        goal_msg.use_start = start is not None
        if start is not None:
            goal_msg.start = self._pose_stamped(start)

        handle = _await_future(self._client.send_goal_async(goal_msg), timeout_sec)
        if handle is None or not handle.accepted:
            raise Nav2PlanningError("Nav2 rejected the ComputePathThroughPoses goal")
        wrapped = _await_future(handle.get_result_async(), timeout_sec)
        if wrapped is None:
            raise Nav2PlanningError("Nav2 did not return a path before the timeout")
        path = wrapped.result.path
        if not path.poses:
            raise Nav2PlanningError("Nav2 returned an empty path")
        frame_id = path.header.frame_id.lstrip("/")
        if frame_id != self._map_frame:
            raise FrameContractError(
                f"Nav2 path frame_id is '{path.header.frame_id}', expected '{self._map_frame}'"
            )
        return tuple(
            (float(pose.pose.position.x), float(pose.pose.position.y)) for pose in path.poses
        )

    def _pose_stamped(self, pose: tuple[float, float, float]) -> PoseStamped:
        """Build a ``map``-frame ``PoseStamped`` stamped with the simulation clock."""
        msg = PoseStamped()
        msg.header.frame_id = self._map_frame
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.orientation.z, msg.pose.orientation.w = yaw_to_quaternion_z_w(float(pose[2]))
        return msg


def _await_future(future, timeout_sec: float):
    """Wait for an rclpy future without spinning the caller's thread."""
    done = threading.Event()
    future.add_done_callback(lambda _future: done.set())
    if not done.wait(timeout_sec):
        future.cancel()
        return None
    return future.result()


class RosBridgeNode(Node):
    """Single ROS node owning every topic, service and action used for training."""

    def __init__(self, config: DeliveryEnvConfig) -> None:
        """Create publishers, subscriptions, tf2 plumbing and all clients."""
        super().__init__(
            config.node_name,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        self._config = config
        self._group = ReentrantCallbackGroup()
        self._binner = LidarBinner(
            config.observation.lidar_bins, config.observation.lidar_max_range
        )

        self._condition = threading.Condition()
        self._scan: ScanSnapshot | None = None
        self._odom: OdomSnapshot | None = None
        self._scan_seq = 0
        self._odom_seq = 0
        self._stale_threshold = float("-inf")

        self.cmd_vel_publisher = self.create_publisher(
            Twist, config.topics.cmd_vel, cmd_vel_qos()
        )
        self.create_subscription(
            LaserScan, config.topics.scan, self._on_scan, scan_qos(), callback_group=self._group
        )
        self.create_subscription(
            Odometry, config.topics.odom, self._on_odom, odom_qos(), callback_group=self._group
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.pose_provider = TfPoseProvider(
            self, self.tf_buffer, config.frames.map_frame, config.frames.base_frame
        )
        self.create_timer(0.02, self.pose_provider.poll, callback_group=self._group)

        self.set_entity_state_client = self._client(
            SetEntityState, config.gazebo.set_entity_state_service
        )
        self.spawn_entity_client = self._client(SpawnEntity, config.gazebo.spawn_entity_service)
        self.delete_entity_client = self._client(DeleteEntity, config.gazebo.delete_entity_service)
        self.pause_client = self._client(Empty, config.gazebo.pause_physics_service)
        self.unpause_client = self._client(Empty, config.gazebo.unpause_physics_service)
        self.global_costmap_client = self._client(
            ClearEntireCostmap, config.gazebo.global_costmap_clear_service
        )

        self.initial_pose_client = None
        self.nomotion_update_client = None
        if config.localization.use_amcl:
            from nav2_msgs.srv import SetInitialPose

            self.initial_pose_client = self._client(
                SetInitialPose, config.localization.set_initial_pose_service
            )
            self.nomotion_update_client = self._client(
                Empty, config.localization.nomotion_update_service
            )

        self.path_provider = Nav2ThroughPosesPathProvider(
            self,
            ActionClient(
                self,
                ComputePathThroughPoses,
                config.topics.compute_path_through_poses_action,
                callback_group=self._group,
            ),
            config.frames.map_frame,
            config.planner_id,
        )

    def _client(self, service_type, service_name: str):
        """Create a service client on the shared reentrant callback group."""
        return self.create_client(service_type, service_name, callback_group=self._group)

    def now_sim_sec(self) -> float:
        """Current simulation time in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_scan(self, msg: LaserScan) -> None:
        """Bin the scan and publish it to waiting control steps."""
        stamp = stamp_to_seconds(msg.header.stamp)
        if stamp < self._stale_threshold:
            return
        bins = self._binner.bin_metres(
            np.asarray(msg.ranges, dtype=np.float64), geometry_from_scan(msg)
        )
        with self._condition:
            self._scan_seq += 1
            self._scan = ScanSnapshot(self._scan_seq, stamp, bins)
            self._condition.notify_all()

    def _on_odom(self, msg: Odometry) -> None:
        """Store the twist only; the pose field is excluded by the frame contract."""
        stamp = stamp_to_seconds(msg.header.stamp)
        if stamp < self._stale_threshold:
            return
        linear = float(np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))
        with self._condition:
            self._odom_seq += 1
            self._odom = OdomSnapshot(
                self._odom_seq, stamp, linear, float(msg.twist.twist.angular.z)
            )
            self._condition.notify_all()

    def discard_before(self, stamp_sec: float) -> None:
        """Drop buffered sensor data stamped before ``stamp_sec`` (a reset boundary)."""
        with self._condition:
            self._stale_threshold = float(stamp_sec)
            self._scan = None
            self._odom = None

    def latest_scan(self) -> ScanSnapshot | None:
        """Most recent accepted ``/scan`` snapshot."""
        with self._condition:
            return self._scan

    def latest_odom(self) -> OdomSnapshot | None:
        """Most recent accepted ``/odom`` snapshot."""
        with self._condition:
            return self._odom

    def sensor_sequences(self) -> tuple[int, int]:
        """Current ``(scan_seq, odom_seq)`` counters."""
        with self._condition:
            return self._scan_seq, self._odom_seq

    def wait_for_sensors(
        self, after_scan_seq: int, after_odom_seq: int, timeout_sec: float
    ) -> tuple[ScanSnapshot | None, OdomSnapshot | None]:
        """Block until newer scan *and* odom snapshots arrive, or the timeout expires.

        Returns whatever is available at the deadline so the caller can decide
        between a sensor-timeout stop and a truncation.
        """
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                fresh_scan = self._scan is not None and self._scan.seq > after_scan_seq
                fresh_odom = self._odom is not None and self._odom.seq > after_odom_seq
                if fresh_scan and fresh_odom:
                    return self._scan, self._odom
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return self._scan, self._odom
                self._condition.wait(remaining)

    def publish_cmd_vel(self, linear: float, angular: float) -> None:
        """Publish one ``/cmd_vel`` command."""
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_vel_publisher.publish(msg)

    def call_service(self, client, request, timeout_sec: float, *, required: bool = True):
        """Call a service asynchronously and wait on a threading event.

        Returns ``None`` if the service never became available (when ``required``
        is false) or if the call did not complete in time.
        """
        if not client.wait_for_service(timeout_sec=timeout_sec if required else 0.2):
            if required:
                self.get_logger().warning(f"service unavailable: {client.srv_name}")
            return None
        return _await_future(client.call_async(request), timeout_sec)

    def validate_cmd_vel_ownership(self) -> None:
        """Fail fast if any Nav2 node is also publishing ``/cmd_vel``."""
        names = [
            info.node_name
            for info in self.get_publishers_info_by_topic(self._config.topics.cmd_vel)
        ]
        forbidden = find_forbidden_publishers(names)
        if forbidden:
            raise RuntimeError(
                "Forbidden /cmd_vel publishers active: " + ", ".join(forbidden)
            )

    def validate_node_graph(self, timeout_sec: float = 5.0) -> None:
        """Fail fast on a ROS graph that breaks the training contract.

        Waits for the planner to appear so the check does not run against a graph
        that has not been discovered yet.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and "planner_server" not in self.get_node_names():
            time.sleep(0.2)
        problems = find_graph_conflicts(self.get_node_names(), self._config)
        if problems:
            raise RuntimeError("ROS graph conflict: " + "; ".join(problems))

    def validate_map_to_odom(self, timeout_sec: float = 5.0) -> None:
        """Fail fast unless ``map->odom`` is being published."""
        deadline = time.monotonic() + timeout_sec
        last_error = "not published"
        while time.monotonic() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    self._config.frames.map_frame,
                    self._config.frames.odom_frame,
                    rclpy.time.Time(),
                )
                return
            except TransformException as error:
                last_error = str(error)
                time.sleep(0.1)
        raise RuntimeError(
            f"map->odom transform is unavailable ({last_error}); start the static "
            "transform publisher (or AMCL when use_amcl:=true) before training"
        )


#: Nodes that must be unique; a second copy means two launches are running.
SINGLETON_NODES: tuple[str, ...] = ("planner_server", "map_server")


def find_graph_conflicts(node_names: list[str], config: DeliveryEnvConfig) -> list[str]:
    """Describe every ROS graph condition that invalidates a training run.

    Per-node action and service introspection returns nothing under CycloneDDS,
    so node names are the only reliable signal here.
    """
    counts = Counter(node_names)
    problems = []
    for name in SINGLETON_NODES:
        if counts[name] > 1:
            problems.append(
                f"{counts[name]} '{name}' nodes are running; only one launch may be active"
            )
    if not config.localization.use_amcl and counts["amcl"]:
        problems.append(
            "'amcl' is running while use_amcl is False -- it publishes map->odom on /tf and "
            "fights the static transform, so map->base_link never settles"
        )
    return problems


class RosBridge:
    """Owns the bridge node and the background ``MultiThreadedExecutor`` thread."""

    def __init__(self, config: DeliveryEnvConfig) -> None:
        """Start rclpy if needed, create the node and begin spinning."""
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self.node = RosBridgeNode(config)
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(
            target=self._executor.spin, name="limo_rl_ros_executor", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the executor thread and tear the node down."""
        self._executor.shutdown(timeout_sec=2.0)
        self._thread.join(timeout=2.0)
        self.node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()
