"""ROS-side contracts: QoS, reset requests, obstacle SDF and Nav2 frame checking.

Skipped entirely when ROS 2 is not on the interpreter path.
"""

import math

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 is not sourced")

from limo_delivery_rl_v2.gazebo_reset import (  # noqa: E402
    obstacle_sdf,
    populate_set_entity_state_request,
)
from limo_delivery_rl_v2.ros_bridge import (  # noqa: E402
    FrameContractError,
    Nav2PlanningError,
    Nav2ThroughPosesPathProvider,
    cmd_vel_qos,
    find_graph_conflicts,
    odom_qos,
    scan_qos,
    stamp_to_seconds,
)
from limo_delivery_rl_v2.state import DeliveryEnvConfig, ObstacleSpec  # noqa: E402


# --------------------------------------------------------------------- QoS


def test_scan_qos_matches_the_specification():
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

    profile = scan_qos()

    assert profile.reliability == ReliabilityPolicy.BEST_EFFORT
    assert profile.durability == DurabilityPolicy.VOLATILE
    assert profile.history == HistoryPolicy.KEEP_LAST
    assert profile.depth == 5


def test_odom_qos_matches_the_specification():
    from rclpy.qos import ReliabilityPolicy

    profile = odom_qos()

    assert profile.reliability == ReliabilityPolicy.BEST_EFFORT
    assert profile.depth == 10


def test_cmd_vel_qos_is_reliable():
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

    profile = cmd_vel_qos()

    assert profile.reliability == ReliabilityPolicy.RELIABLE
    assert profile.durability == DurabilityPolicy.VOLATILE
    assert profile.depth == 10


def test_stamp_conversion_is_exact_to_the_nanosecond():
    from builtin_interfaces.msg import Time

    stamp = Time(sec=12, nanosec=500_000_000)

    assert stamp_to_seconds(stamp) == pytest.approx(12.5)


# ------------------------------------------------------------ Gazebo reset


def test_set_entity_state_request_carries_pose_and_a_zeroed_twist():
    from gazebo_msgs.srv import SetEntityState

    request = SetEntityState.Request()
    populate_set_entity_state_request(request, "limo_car", "world", (1.5, -0.25, math.pi / 2))

    assert request.state.name == "limo_car"
    assert request.state.reference_frame == "world"
    assert request.state.pose.position.x == pytest.approx(1.5)
    assert request.state.pose.position.y == pytest.approx(-0.25)
    assert request.state.pose.position.z == pytest.approx(0.0)
    assert request.state.pose.orientation.z == pytest.approx(math.sin(math.pi / 4))
    assert request.state.pose.orientation.w == pytest.approx(math.cos(math.pi / 4))
    for axis in ("x", "y", "z"):
        assert getattr(request.state.twist.linear, axis) == 0.0
        assert getattr(request.state.twist.angular, axis) == 0.0


def test_obstacle_sdf_describes_the_configured_box():
    spec = ObstacleSpec()
    sdf = obstacle_sdf(spec)

    assert spec.name == "rl_fixed_obstacle"
    assert (spec.x, spec.y) == (2.07, -0.18)
    assert f'<model name="{spec.name}">' in sdf
    assert "<size>0.25 0.25 1.0</size>" in sdf
    assert "<static>true</static>" in sdf


def test_the_static_map_is_only_ever_read_never_written():
    """Obstacles live in Gazebo only; nothing may write map.yaml or map.pgm.

    Docstrings are stripped first -- several modules legitimately *explain* that
    the map ranges were verified against map.pgm.
    """
    import ast
    import pathlib

    package = pathlib.Path(__file__).resolve().parents[1] / "limo_delivery_rl_v2"

    def executable_source(path: pathlib.Path) -> str:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
                if isinstance(getattr(body[0], "value", None), ast.Constant):
                    body.pop(0)
        return ast.unparse(tree)

    referencing = {
        source.name
        for source in package.glob("*.py")
        if "map.pgm" in executable_source(source) or "yaml_path" in executable_source(source)
    }

    # Only the config dataclass names the files and only map_utils touches them.
    assert referencing == {"state.py", "map_utils.py"}
    map_utils = (package / "map_utils.py").read_text()
    assert "is_file()" in map_utils
    for writing in ("write_text", "write_bytes", '"w"', "'w'", "shutil"):
        assert writing not in map_utils


# ------------------------------------------------------------------- Nav2


class _FakeHeader:
    def __init__(self, frame_id: str) -> None:
        self.frame_id = frame_id
        self.stamp = None


class _FakePosition:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class _FakePose:
    def __init__(self, x: float, y: float) -> None:
        self.pose = type("Inner", (), {"position": _FakePosition(x, y)})()


class _FakePath:
    def __init__(self, frame_id: str, points) -> None:
        self.header = _FakeHeader(frame_id)
        self.poses = [_FakePose(x, y) for x, y in points]


class _FakeResult:
    def __init__(self, path) -> None:
        self.result = type("Wrapper", (), {"path": path})()


class _FakeGoalHandle:
    def __init__(self, accepted: bool, path) -> None:
        self.accepted = accepted
        self._path = path

    def get_result_async(self):
        return _ImmediateFuture(_FakeResult(self._path))


class _ImmediateFuture:
    def __init__(self, value) -> None:
        self._value = value

    def add_done_callback(self, callback):
        callback(self)

    def cancel(self):
        pass

    def result(self):
        return self._value


class _FakeActionClient:
    def __init__(self, *, ready: bool = True, accepted: bool = True, path=None) -> None:
        self.ready = ready
        self.accepted = accepted
        self.path = path
        self.sent_goal = None

    def wait_for_server(self, timeout_sec):
        return self.ready

    def send_goal_async(self, goal):
        self.sent_goal = goal
        return _ImmediateFuture(_FakeGoalHandle(self.accepted, self.path))


class _FakeNode:
    def get_clock(self):
        from rclpy.clock import Clock

        return Clock()


def provider(client) -> Nav2ThroughPosesPathProvider:
    """A path provider wired to a fake action client."""
    return Nav2ThroughPosesPathProvider(_FakeNode(), client, "map", "GridBased")


def test_path_request_sends_every_waypoint_as_one_goal():
    config = DeliveryEnvConfig()
    client = _FakeActionClient(path=_FakePath("map", [(0.0, 0.0), (1.0, 0.0)]))

    provider(client).request_path(config.waypoints, (0.0, 0.0, 0.0), 1.0)

    goal = client.sent_goal
    assert len(goal.goals) == len(config.waypoints)
    assert goal.planner_id == "GridBased"
    assert goal.use_start is True
    assert all(pose.header.frame_id == "map" for pose in goal.goals)
    assert goal.goals[-1].pose.position.x == pytest.approx(9.5)


def test_path_points_are_returned_in_order():
    client = _FakeActionClient(path=_FakePath("map", [(0.0, 0.0), (0.5, 0.1), (1.0, 0.0)]))

    points = provider(client).request_path(((1.0, 0.0, 0.0),), None, 1.0)

    assert points == ((0.0, 0.0), (0.5, 0.1), (1.0, 0.0))


def test_a_path_in_the_wrong_frame_is_an_immediate_error():
    client = _FakeActionClient(path=_FakePath("odom", [(0.0, 0.0), (1.0, 0.0)]))

    with pytest.raises(FrameContractError, match="odom"):
        provider(client).request_path(((1.0, 0.0, 0.0),), None, 1.0)


def test_missing_server_rejected_goal_and_empty_path_all_raise():
    with pytest.raises(Nav2PlanningError, match="unavailable"):
        provider(_FakeActionClient(ready=False)).request_path(((1.0, 0.0, 0.0),), None, 0.01)

    rejected = _FakeActionClient(accepted=False, path=_FakePath("map", [(0.0, 0.0)]))
    with pytest.raises(Nav2PlanningError, match="rejected"):
        provider(rejected).request_path(((1.0, 0.0, 0.0),), None, 0.01)

    empty = _FakeActionClient(path=_FakePath("map", []))
    with pytest.raises(Nav2PlanningError, match="empty"):
        provider(empty).request_path(((1.0, 0.0, 0.0),), None, 0.01)


def test_training_configuration_excludes_amcl_and_uses_an_identity_map_to_odom():
    localization = DeliveryEnvConfig().localization

    assert localization.use_amcl is False
    assert (localization.map_to_odom_x, localization.map_to_odom_y) == (0.0, 0.0)
    assert localization.map_to_odom_yaw == 0.0


# ------------------------------------------------------------- graph checks

HEALTHY_GRAPH = [
    "gazebo",
    "gazebo_ros_state",
    "four_diff_controller",
    "robot_state_publisher",
    "map_server",
    "planner_server",
    "map_to_odom_static_broadcaster",
    "lifecycle_manager_planner_only",
]


def test_a_planner_only_graph_reports_no_conflict():
    assert find_graph_conflicts(HEALTHY_GRAPH, DeliveryEnvConfig()) == []


def test_amcl_running_during_training_is_a_conflict():
    problems = find_graph_conflicts(HEALTHY_GRAPH + ["amcl"], DeliveryEnvConfig())

    assert len(problems) == 1
    assert "amcl" in problems[0]
    assert "map->odom" in problems[0]


def test_amcl_is_allowed_when_use_amcl_is_true():
    from dataclasses import replace

    base = DeliveryEnvConfig()
    config = replace(base, localization=replace(base.localization, use_amcl=True))

    assert find_graph_conflicts(HEALTHY_GRAPH + ["amcl"], config) == []


def test_two_launches_are_detected_by_duplicate_singleton_nodes():
    doubled = HEALTHY_GRAPH + ["planner_server", "map_server"]

    problems = find_graph_conflicts(doubled, DeliveryEnvConfig())

    assert len(problems) == 2
    assert any("planner_server" in p and "2" in p for p in problems)
    assert any("map_server" in p and "2" in p for p in problems)


def test_the_real_failure_from_the_field_is_reported_in_full():
    """The graph that crashed a training run: old and new launch both active."""
    observed = [
        "amcl", "four_diff_controller", "gazebo", "gazebo_ros_laser_sensor",
        "gazebo_ros_state", "lifecycle_manager_localization",
        "lifecycle_manager_planner_only", "lifecycle_manager_planner_only",
        "map_server", "map_server", "map_to_odom_static_broadcaster",
        "planner_server", "planner_server", "robot_state_publisher", "rviz",
    ]

    problems = find_graph_conflicts(observed, DeliveryEnvConfig())

    assert len(problems) == 3
    assert any("amcl" in p for p in problems)
