"""Planner-only Nav2 stack for RL training.

Started nodes:

* ``map_server``  - serves the static map (no training obstacle is ever in it)
* ``planner_server`` - hosts ``GridBased`` / navfn and the global costmap
* ``lifecycle_manager`` - autostarts the two above
* ``static_transform_publisher`` (``map -> odom``) when ``use_amcl:=false``
* ``amcl`` instead of the static transform when ``use_amcl:=true``

Never started: ``controller_server``, ``velocity_smoother``, ``behavior_server``
and ``waypoint_follower``.  All four can publish ``/cmd_vel``, which must be
owned exclusively by the RL bridge node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the planner-only launch description."""
    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_amcl = LaunchConfiguration("use_amcl")
    autostart = LaunchConfiguration("autostart")
    map_to_odom_x = LaunchConfiguration("map_to_odom_x")
    map_to_odom_y = LaunchConfiguration("map_to_odom_y")
    map_to_odom_yaw = LaunchConfiguration("map_to_odom_yaw")

    default_params = PathJoinSubstitution(
        [FindPackageShare("limo_delivery_rl_v2"), "config", "nav2_planner_only_rl.yaml"]
    )
    common = [params_file, {"use_sim_time": use_sim_time}]

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=common + [{"yaml_filename": map_yaml}],
    )

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=common,
    )

    # Training: Gazebo odometry is ground truth and the world frame coincides
    # with the map frame, so map->odom is a fixed (identity by default) offset.
    map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_static_broadcaster",
        output="screen",
        condition=UnlessCondition(use_amcl),
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=[
            "--x", map_to_odom_x,
            "--y", map_to_odom_y,
            "--z", "0.0",
            "--yaw", map_to_odom_yaw,
            "--pitch", "0.0",
            "--roll", "0.0",
            "--frame-id", "map",
            "--child-frame-id", "odom",
        ],
    )

    # Hardware deployment only. AMCL is excluded from training because training
    # obstacles are absent from map.pgm, so scan matching would be corrupted and
    # map->odom would drift with obstacle placement.
    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        condition=IfCondition(use_amcl),
        parameters=common,
    )

    lifecycle_planner_only = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_planner_only",
        output="screen",
        condition=UnlessCondition(use_amcl),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": autostart},
            {"node_names": ["map_server", "planner_server"]},
        ],
    )

    lifecycle_with_amcl = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_planner_only",
        output="screen",
        condition=IfCondition(use_amcl),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": autostart},
            {"node_names": ["map_server", "amcl", "planner_server"]},
        ],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
            DeclareLaunchArgument("map", default_value="/home/kim/limo_ws/map.yaml"),
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "use_amcl",
                default_value="false",
                description="false: static map->odom for training. true: AMCL for hardware.",
            ),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("map_to_odom_x", default_value="0.0"),
            DeclareLaunchArgument("map_to_odom_y", default_value="0.0"),
            DeclareLaunchArgument("map_to_odom_yaw", default_value="0.0"),
            map_server,
            planner_server,
            map_to_odom,
            amcl,
            lifecycle_planner_only,
            lifecycle_with_amcl,
        ]
    )
