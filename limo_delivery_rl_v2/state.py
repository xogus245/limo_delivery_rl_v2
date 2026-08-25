"""Configuration dataclasses and shared enums for the LIMO waypoint RL environment.

Architecture A: Nav2 supplies only the global path (``ComputePathThroughPoses``)
while the RL policy emits ``(v, omega)`` directly on ``/cmd_vel``.  Every value
here is a *training* constant; nothing in this module touches ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StopReason(str, Enum):
    """Reason an episode ended, mirrored into ``info['reason']``."""

    NONE = "none"
    SUCCESS = "success"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    STUCK = "stuck"
    PATH_FAILED = "path_failed"
    PATH_DEVIATION = "path_deviation"
    SENSOR_TIMEOUT = "sensor_timeout"


class SafetyMode(str, Enum):
    """Outcome of the hard safety gate applied to every command."""

    OK = "ok"
    INVALID_ACTION = "invalid_action"
    SENSOR_TIMEOUT = "sensor_timeout"
    TF_TIMEOUT = "tf_timeout"
    IMMINENT_COLLISION = "imminent_collision"

    @property
    def is_hard_stop(self) -> bool:
        """Whether this mode forced ``(0, 0)`` regardless of the policy action."""
        return self is not SafetyMode.OK


@dataclass(frozen=True, slots=True)
class TopicConfig:
    """ROS topic and action names used by the bridge node."""

    cmd_vel: str = "/cmd_vel"
    scan: str = "/scan"
    odom: str = "/odom"
    compute_path_through_poses_action: str = "compute_path_through_poses"


@dataclass(frozen=True, slots=True)
class FrameConfig:
    """Frame ids enforced by the frame contract."""

    map_frame: str = "map"
    odom_frame: str = "odom"
    base_frame: str = "base_link"


@dataclass(frozen=True, slots=True)
class MapConfig:
    """Static occupancy map shared by Nav2 and the training scripts."""

    yaml_path: str = "/home/kim/limo_ws/map.yaml"
    image_path: str = "/home/kim/limo_ws/map.pgm"
    resolution: float = 0.05
    origin_x: float = -30.2
    origin_y: float = -4.72
    occupied_thresh: float = 0.65
    free_thresh: float = 0.25


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Observation shaping constants.

    ``lidar_max_range`` matches the simulated LiDAR (``8.0 m``); the clipping and
    normalisation reference must never drift away from the sensor definition.
    """

    lidar_bins: int = 24
    lidar_frame_stack: int = 4
    lidar_max_range: float = 8.0
    lidar_min_range: float = 0.20
    lookahead_points: int = 5
    lookahead_spacing: float = 0.40
    max_relative_position: float = 3.0
    max_waypoint_distance: float = 20.0
    nearest_search_window: float = 5.0


@dataclass(frozen=True, slots=True)
class ControlConfig:
    """Actuation limits.

    ``max_train_linear_speed`` bounds the policy action; ``max_linear_speed`` is
    the absolute hardware/simulator ceiling enforced by the safety gate.
    """

    max_train_linear_speed: float = 0.42
    max_linear_speed: float = 0.55
    max_angular_speed: float = 0.9
    max_linear_accel: float = 1.0
    max_linear_decel: float = 1.0
    max_angular_accel: float = 1.8
    control_dt: float = 0.05


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    """Hard safety thresholds; no distance-proportional slowdown is applied."""

    # Must stay above the simulated LiDAR floor (0.20000000298 m), otherwise a
    # contact-range return can miss the <= comparison and never terminate.
    collision_distance: float = 0.25
    danger_distance: float = 0.80
    sensor_timeout_sec: float = 0.50
    tf_timeout_sec: float = 0.50
    max_path_deviation: float = 2.50


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Episode bookkeeping thresholds (20 Hz control)."""

    waypoint_radius: float = 0.60
    waypoint_hold_steps: int = 5
    #: Half-width of the plane-crossing capture band, in metres. ``0.0`` disables
    #: it, leaving the strict radius-and-hold rule the specification defines.
    #: A positive value also accepts a waypoint the robot drove past off-centre,
    #: which the radius rule alone misses entirely.
    waypoint_capture_width: float = 0.0
    max_steps: int = 5000
    stuck_speed_threshold: float = 0.016
    stuck_step_limit: int = 300
    sensor_lost_step_limit: int = 40


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Per-step reward weights, all bounded so no term dominates a single step."""

    progress_gain: float = 10.0
    progress_delta_clip: float = 0.05
    waypoint_reward: float = 20.0
    success_reward: float = 100.0
    collision_penalty: float = -100.0
    stuck_penalty: float = -80.0
    danger_penalty_scale: float = 0.10
    deviation_penalty_scale: float = 0.02
    deviation_reference: float = 2.50
    time_penalty: float = -0.01
    smoothness_penalty_scale: float = 0.02


@dataclass(frozen=True, slots=True)
class StartPoseConfig:
    """Episode start pose and its randomisation envelope, in the ``map`` frame."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    x_jitter: float = 0.20
    y_jitter: float = 0.15
    yaw_jitter: float = 0.087


@dataclass(frozen=True, slots=True)
class GazeboConfig:
    """Gazebo entity and service names used during reset."""

    entity_name: str = "limo_car"
    reference_frame: str = "world"
    set_entity_state_service: str = "/set_entity_state"
    spawn_entity_service: str = "/spawn_entity"
    delete_entity_service: str = "/delete_entity"
    pause_physics_service: str = "/pause_physics"
    unpause_physics_service: str = "/unpause_physics"
    global_costmap_clear_service: str = "/global_costmap/clear_entirely_global_costmap"
    service_timeout_sec: float = 5.0
    zero_command_hold_sec: float = 0.5
    pose_settle_tolerance: float = 0.10
    pose_settle_timeout_sec: float = 5.0
    sensor_refresh_timeout_sec: float = 2.0
    path_request_timeout_sec: float = 10.0


@dataclass(frozen=True, slots=True)
class ObstacleSpec:
    """A single box obstacle spawned *after* the global path is planned."""

    name: str = "rl_fixed_obstacle"
    x: float = 2.07
    y: float = -0.18
    yaw: float = 0.0
    size_x: float = 0.25
    size_y: float = 0.25
    height: float = 1.0


@dataclass(frozen=True, slots=True)
class ObstacleConfig:
    """Training obstacles. They are never written into ``map.pgm``."""

    enabled: bool = True
    specs: tuple[ObstacleSpec, ...] = (ObstacleSpec(),)


@dataclass(frozen=True, slots=True)
class LocalizationConfig:
    """Localization source.

    Training uses a static ``map->odom`` transform because Gazebo odometry is
    ground truth; AMCL is reserved for hardware deployment (``use_amcl=True``).
    The Gazebo world frame coincides with the map frame for this workspace, so
    the default offset is identity.
    """

    use_amcl: bool = False
    map_to_odom_x: float = 0.0
    map_to_odom_y: float = 0.0
    map_to_odom_yaw: float = 0.0
    set_initial_pose_service: str = "/set_initial_pose"
    nomotion_update_service: str = "/request_nomotion_update"


#: Ordered fixed waypoints for the first experiment, in the ``map`` frame.
WAYPOINTS: tuple[tuple[float, float, float], ...] = (
    (3.0, 0.0, 0.0),
    (6.0, 0.0, 0.0),
    (9.5, 0.0, 0.0),
)


@dataclass(frozen=True, slots=True)
class DeliveryEnvConfig:
    """Aggregate configuration for :class:`~limo_delivery_rl_v2.delivery_env.LimoWaypointRLEnv`."""

    topics: TopicConfig = TopicConfig()
    frames: FrameConfig = FrameConfig()
    map: MapConfig = MapConfig()
    observation: ObservationConfig = ObservationConfig()
    control: ControlConfig = ControlConfig()
    safety: SafetyConfig = SafetyConfig()
    episode: EpisodeConfig = EpisodeConfig()
    reward: RewardConfig = RewardConfig()
    start_pose: StartPoseConfig = StartPoseConfig()
    gazebo: GazeboConfig = GazeboConfig()
    obstacles: ObstacleConfig = ObstacleConfig()
    localization: LocalizationConfig = LocalizationConfig()
    waypoints: tuple[tuple[float, float, float], ...] = WAYPOINTS
    planner_id: str = "GridBased"
    node_name: str = "limo_waypoint_rl_bridge"


def observation_dim(config: ObservationConfig) -> int:
    """Return the flattened observation size implied by ``config`` (112 by default)."""
    return (
        config.lidar_bins * config.lidar_frame_stack
        + config.lookahead_points * 2
        + 2  # waypoint distance + heading error
        + 2  # measured linear + angular velocity
        + 2  # previous normalized action
    )


def stage_config(
    base: DeliveryEnvConfig | None = None,
    *,
    waypoint_count: int | None = None,
    waypoint_radius: float | None = None,
    waypoint_hold_steps: int | None = None,
    waypoint_capture_width: float | None = None,
    obstacles_enabled: bool | None = None,
) -> DeliveryEnvConfig:
    """Build a curriculum-stage config, leaving unspecified fields at their default.

    Stages differ only in how many waypoints are active and how forgiving the
    arrival test is; the observation layout is identical across stages, so a
    policy trained on one stage loads directly into the next.
    """
    from dataclasses import replace

    config = base or DeliveryEnvConfig()
    if waypoint_count is not None:
        if not 1 <= waypoint_count <= len(config.waypoints):
            raise ValueError(
                f"waypoint_count must be between 1 and {len(config.waypoints)}"
            )
        config = replace(config, waypoints=config.waypoints[:waypoint_count])
    episode = config.episode
    if waypoint_radius is not None:
        episode = replace(episode, waypoint_radius=waypoint_radius)
    if waypoint_hold_steps is not None:
        episode = replace(episode, waypoint_hold_steps=waypoint_hold_steps)
    if waypoint_capture_width is not None:
        episode = replace(episode, waypoint_capture_width=waypoint_capture_width)
    config = replace(config, episode=episode)
    if obstacles_enabled is not None:
        config = replace(config, obstacles=replace(config.obstacles, enabled=obstacles_enabled))
    return config
