"""Gazebo/Nav2 reset ordering, verified against a recording double."""

import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 is not sourced")

from limo_delivery_rl_v2.gazebo_reset import (  # noqa: E402
    GazeboResetError,
    GazeboResetManager,
    ObstacleManager,
)
from limo_delivery_rl_v2.geometry import Pose2D  # noqa: E402
from limo_delivery_rl_v2.ros_bridge import PoseSnapshot  # noqa: E402
from limo_delivery_rl_v2.state import DeliveryEnvConfig, LocalizationConfig  # noqa: E402


class _Response:
    def __init__(self, success: bool = True, status_message: str = "") -> None:
        self.success = success
        self.status_message = status_message


class _Client:
    def __init__(self, name: str) -> None:
        self.srv_name = name


class _PoseProvider:
    def __init__(self, pose: Pose2D | None) -> None:
        self._pose = pose

    def snapshot(self):
        return None if self._pose is None else PoseSnapshot(0.0, self._pose)


class RecordingNode:
    """Records the order of every bridge call the reset manager makes."""

    def __init__(self, *, pose: Pose2D | None = Pose2D(0.0, 0.0, 0.0), failures=()) -> None:
        self.calls: list[str] = []
        self.failures = set(failures)
        self.pose_provider = _PoseProvider(pose)
        self.set_entity_state_client = _Client("/set_entity_state")
        self.spawn_entity_client = _Client("/spawn_entity")
        self.delete_entity_client = _Client("/delete_entity")
        self.pause_client = _Client("/pause_physics")
        self.unpause_client = _Client("/unpause_physics")
        self.global_costmap_client = _Client("/global_costmap/clear_entirely_global_costmap")
        self.initial_pose_client = _Client("/set_initial_pose")
        self.nomotion_update_client = _Client("/request_nomotion_update")

    def publish_cmd_vel(self, linear, angular):
        assert (linear, angular) == (0.0, 0.0)
        if "zero_command" not in self.calls:
            self.calls.append("zero_command")
        elif self.calls[-1] != "zero_command":
            self.calls.append("zero_command")

    def call_service(self, client, request, timeout_sec, *, required=True):
        name = client.srv_name.strip("/").replace("/", "_")
        self.calls.append(name)
        if name in self.failures:
            return None
        return _Response()

    def validate_map_to_odom(self, timeout_sec):
        self.calls.append("validate_map_to_odom")

    def get_clock(self):
        from rclpy.clock import Clock

        return Clock()


def build(config: DeliveryEnvConfig | None = None, **node_kwargs):
    """Return ``(node, reset_manager)`` wired to a recording node."""
    config = config or DeliveryEnvConfig()
    node = RecordingNode(**node_kwargs)
    obstacles = ObstacleManager(node, config)
    return node, GazeboResetManager(node, config, obstacles)


def fast_config(**localization) -> DeliveryEnvConfig:
    """A config whose reset waits are short enough for a unit test."""
    from dataclasses import replace

    base = DeliveryEnvConfig()
    gazebo = replace(
        base.gazebo,
        zero_command_hold_sec=0.0,
        pose_settle_timeout_sec=0.2,
        service_timeout_sec=0.2,
    )
    return replace(
        base,
        gazebo=gazebo,
        localization=LocalizationConfig(**localization) if localization else base.localization,
    )


def test_reset_follows_the_specified_order():
    node, manager = build(fast_config())

    manager.reset_robot((0.1, 0.0, 0.0))

    assert node.calls == [
        "zero_command",            # 1. repeated stop commands
        "pause_physics",           # 2. pause
        "delete_entity",           # 3. remove the previous obstacle
        "set_entity_state",        # 4. pose + twist in one call
        "unpause_physics",         # 5. unpause
        "zero_command",            # 6. stop again
        "pause_physics",           # 7. stabilise the pose once more
        "set_entity_state",
        "unpause_physics",
        "validate_map_to_odom",    # 8. static transform must be live
        "global_costmap_clear_entirely_global_costmap",  # 9. only running costmaps
        # 10. tf pose match is verified by the pose provider, no service call
    ]


def test_obstacles_are_deleted_before_the_robot_is_teleported():
    node, manager = build(fast_config())

    manager.reset_robot((0.0, 0.0, 0.0))

    assert node.calls.index("delete_entity") < node.calls.index("set_entity_state")


def test_no_local_costmap_service_is_required():
    node, manager = build(fast_config())

    manager.reset_robot((0.0, 0.0, 0.0))

    assert not any("local_costmap" in call for call in node.calls)


def test_training_reset_never_touches_amcl():
    node, manager = build(fast_config())

    manager.reset_robot((0.0, 0.0, 0.0))

    assert "set_initial_pose" not in node.calls
    assert "request_nomotion_update" not in node.calls


def test_hardware_reset_seeds_amcl_instead_of_the_static_transform():
    config = fast_config(use_amcl=True)
    node, manager = build(config)

    manager.reset_robot((0.0, 0.0, 0.0))

    assert "set_initial_pose" in node.calls
    assert "request_nomotion_update" in node.calls
    assert "validate_map_to_odom" not in node.calls


def test_reset_fails_loudly_when_the_pose_never_matches():
    node, manager = build(fast_config(), pose=Pose2D(5.0, 5.0, 0.0))

    with pytest.raises(GazeboResetError, match="never matched"):
        manager.reset_robot((0.0, 0.0, 0.0))


def test_reset_fails_loudly_when_the_teleport_service_fails():
    node, manager = build(fast_config(), failures={"set_entity_state"})

    with pytest.raises(GazeboResetError, match="/set_entity_state"):
        manager.reset_robot((0.0, 0.0, 0.0))


def test_missing_obstacle_on_delete_is_tolerated():
    config = fast_config()
    node = RecordingNode()

    def call_service(client, request, timeout_sec, *, required=True):
        node.calls.append(client.srv_name.strip("/"))
        if client.srv_name == "/delete_entity":
            return _Response(False, "Entity [rl_fixed_obstacle] does not exist")
        return _Response()

    node.call_service = call_service
    ObstacleManager(node, config).remove_all()

    assert "delete_entity" in node.calls


def test_spawn_failure_is_reported():
    config = fast_config()
    node = RecordingNode(failures={"spawn_entity"})

    with pytest.raises(GazeboResetError, match="spawn obstacle"):
        ObstacleManager(node, config).spawn_all()


def test_disabled_obstacles_are_a_no_op():
    from dataclasses import replace

    base = fast_config()
    config = replace(base, obstacles=replace(base.obstacles, enabled=False))
    node = RecordingNode()

    ObstacleManager(node, config).spawn_all()
    ObstacleManager(node, config).remove_all()

    assert node.calls == []


def test_reset_world_is_not_used():
    """Rewinding the clock would invalidate every buffered tf2 transform.

    The module docstring explains the choice, so only executable code is checked.
    """
    import ast
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1] / "limo_delivery_rl_v2" / "gazebo_reset.py"
    )
    tree = ast.parse(path.read_text())
    code = ast.unparse(
        ast.Module(
            body=[node for node in tree.body if not isinstance(node, ast.Expr)], type_ignores=[]
        )
    )

    assert "reset_world" not in code
    assert "reset_simulation" not in code
    import dataclasses

    gazebo = DeliveryEnvConfig().gazebo
    for field in dataclasses.fields(gazebo):
        assert "reset_world" not in str(getattr(gazebo, field.name)), field.name
