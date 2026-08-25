"""End-to-end environment contract, exercised against the offline backend."""

import numpy as np
import pytest

from limo_delivery_rl_v2.delivery_env import LimoWaypointRLEnv
from limo_delivery_rl_v2.env_backend import BackendFrame, OfflineBackend
from limo_delivery_rl_v2.geometry import Pose2D
from limo_delivery_rl_v2.state import DeliveryEnvConfig, SafetyMode, StopReason

FORWARD = np.array([1.0, 0.0], dtype=np.float32)
IDLE = np.array([-1.0, 0.0], dtype=np.float32)


def make_env(config: DeliveryEnvConfig | None = None) -> LimoWaypointRLEnv:
    """An environment on the deterministic offline backend."""
    return LimoWaypointRLEnv(config=config, enable_ros=False)


def run_episode(env: LimoWaypointRLEnv, action=FORWARD, limit: int = 6000):
    """Drive a fixed action until the episode ends; return the final info."""
    env.reset(seed=42)
    info: dict = {}
    for _ in range(limit):
        _obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info
    raise AssertionError("episode did not terminate")


def test_spaces_match_the_specification():
    env = make_env()

    assert env.observation_space.shape == (112,)
    assert env.action_space.shape == (2,)
    assert env.action_space.low.tolist() == [-1.0, -1.0]
    assert env.action_space.high.tolist() == [1.0, 1.0]
    assert env.action_space.dtype == np.float32
    env.close()


def test_reset_returns_an_in_bounds_observation():
    env = make_env()

    observation, info = env.reset(seed=1)

    assert env.observation_space.contains(observation)
    assert info["reason"] == StopReason.NONE.value
    env.close()


def test_gymnasium_environment_checker_accepts_the_environment():
    from gymnasium.utils.env_checker import check_env

    env = make_env()
    check_env(env, skip_render_check=True)
    env.close()


def test_reset_is_deterministic_for_a_fixed_seed():
    first = make_env().reset(seed=7)[0]
    second = make_env().reset(seed=7)[0]

    np.testing.assert_array_equal(first, second)


def test_start_pose_randomization_stays_inside_the_configured_envelope():
    config = DeliveryEnvConfig()
    env = make_env(config)
    backend = env._backend
    assert isinstance(backend, OfflineBackend)

    for seed in range(25):
        env.reset(seed=seed)
        pose = backend.pose
        assert abs(pose.x - config.start_pose.x) <= config.start_pose.x_jitter + 1e-9
        assert abs(pose.y - config.start_pose.y) <= config.start_pose.y_jitter + 1e-9
        assert abs(pose.yaw - config.start_pose.yaw) <= config.start_pose.yaw_jitter + 1e-9
    env.close()


def test_every_step_observation_stays_inside_the_declared_space():
    env = make_env()
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)

    for _ in range(200):
        observation, _r, terminated, truncated, _i = env.step(
            np.array([0.4, 0.3], dtype=np.float32)
        )
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break
    env.close()


def test_driving_forward_reaches_every_waypoint_and_succeeds():
    env = make_env()

    info = run_episode(env)
    summary = info["episode_summary"]

    assert info["reason"] == StopReason.SUCCESS.value
    assert summary["waypoints_reached"] == 3.0
    assert summary["reward_waypoint"] == pytest.approx(60.0)
    assert summary["reward_success"] == pytest.approx(100.0)
    assert summary["path_length_ratio"] > 0.9
    env.close()


def test_waypoint_switch_discontinuity_is_measured_not_hidden():
    env = make_env()

    summary = run_episode(env)["episode_summary"]

    # Two switches (the third waypoint completes instead of switching), each
    # saturating the clipped progress delta at -0.5.
    assert summary["waypoint_switch_progress_sum"] == pytest.approx(-1.0)
    env.close()


def test_idling_ends_the_episode_as_stuck_not_as_a_timeout():
    env = make_env()

    info = run_episode(env, action=IDLE, limit=2000)

    assert info["reason"] == StopReason.STUCK.value
    assert info["episode_summary"]["reward_stuck"] == pytest.approx(-80.0)
    assert float(info["episode_summary"]["steps"]) == pytest.approx(300.0, abs=2)
    env.close()


def test_invalid_actions_stop_the_robot_without_crashing():
    env = make_env()
    env.reset(seed=5)

    _obs, _reward, _term, _trunc, info = env.step(np.array([np.nan, 0.0], dtype=np.float32))

    assert info["safety_mode"] == SafetyMode.INVALID_ACTION.value
    env.close()


def test_forward_speed_is_not_reduced_near_an_obstacle():
    """A close obstacle must change the reward, never the commanded speed."""

    class NearObstacleBackend(OfflineBackend):
        """Offline backend reporting an obstacle 0.30 m ahead."""

        def _frame(self) -> BackendFrame:
            frame = super()._frame()
            bins = frame.lidar_metres.copy()
            bins[12] = 0.30
            return BackendFrame(
                pose=frame.pose,
                linear_velocity=frame.linear_velocity,
                angular_velocity=frame.angular_velocity,
                lidar_metres=bins,
                scan_age=0.0,
                odom_age=0.0,
                tf_age=0.0,
            )

    config = DeliveryEnvConfig()
    near = LimoWaypointRLEnv(config=config, enable_ros=False, backend=NearObstacleBackend(config))
    far = make_env(config)
    near.reset(seed=11)
    far.reset(seed=11)

    for _ in range(30):
        _o, near_reward, _t, _tr, near_info = near.step(FORWARD)
        _o, far_reward, _t, _tr, _far_info = far.step(FORWARD)

    assert near_info["safety_mode"] == SafetyMode.OK.value
    assert near._previous_linear == pytest.approx(far._previous_linear)
    assert near._previous_linear == pytest.approx(config.control.max_train_linear_speed)
    assert near_reward < far_reward, "only the reward reflects the danger"
    near.close()
    far.close()


def test_collision_terminates_immediately_with_the_full_penalty():
    class CollisionBackend(OfflineBackend):
        """Offline backend reporting contact range on every bin."""

        def _frame(self) -> BackendFrame:
            frame = super()._frame()
            return BackendFrame(
                pose=frame.pose,
                linear_velocity=frame.linear_velocity,
                angular_velocity=frame.angular_velocity,
                lidar_metres=np.full_like(frame.lidar_metres, 0.21),
                scan_age=0.0,
                odom_age=0.0,
                tf_age=0.0,
            )

    config = DeliveryEnvConfig()
    env = LimoWaypointRLEnv(config=config, enable_ros=False, backend=CollisionBackend(config))
    env.reset(seed=2)

    _obs, reward, terminated, truncated, info = env.step(FORWARD)

    assert terminated and not truncated
    assert info["reason"] == StopReason.COLLISION.value
    assert info["safety_mode"] == SafetyMode.IMMINENT_COLLISION.value
    assert reward < -99.0
    env.close()


def test_a_safety_stop_is_not_also_punished_as_being_stuck():
    class DeadSensorBackend(OfflineBackend):
        """Offline backend whose sensors never refresh."""

        def _frame(self) -> BackendFrame:
            frame = super()._frame()
            return BackendFrame(
                pose=frame.pose,
                linear_velocity=0.0,
                angular_velocity=0.0,
                lidar_metres=frame.lidar_metres,
                scan_age=5.0,
                odom_age=5.0,
                tf_age=0.0,
            )

    config = DeliveryEnvConfig()
    env = LimoWaypointRLEnv(config=config, enable_ros=False, backend=DeadSensorBackend(config))
    env.reset(seed=2)

    info: dict = {}
    for _ in range(200):
        _obs, _reward, terminated, truncated, info = env.step(FORWARD)
        if terminated or truncated:
            break

    assert info["reason"] == StopReason.SENSOR_TIMEOUT.value
    assert info["episode_summary"]["reward_stuck"] == pytest.approx(0.0)
    env.close()


def test_lidar_is_normalized_by_the_eight_metre_sensor_range():
    env = make_env()
    env.reset(seed=6)

    normalized = env._normalized_lidar(env._frame)

    # Offline backend reports free space everywhere: 8.0 m / 8.0 m == 1.0.
    assert np.all(normalized == pytest.approx(1.0))
    half = env._normalized_lidar(
        BackendFrame(
            pose=env._pose,
            linear_velocity=0.0,
            angular_velocity=0.0,
            lidar_metres=np.full(24, 4.0, dtype=np.float32),
            scan_age=0.0,
            odom_age=0.0,
            tf_age=0.0,
        )
    )
    assert np.all(half == pytest.approx(0.5))
    env.close()


def test_step_info_exposes_every_reward_term():
    from limo_delivery_rl_v2.reward import REWARD_TERMS

    env = make_env()
    env.reset(seed=9)

    _obs, _reward, _t, _tr, info = env.step(FORWARD)

    for term in REWARD_TERMS:
        assert f"reward/{term}" in info
    env.close()


def test_episode_summary_reports_the_evaluation_metrics():
    env = make_env()

    summary = run_episode(env)["episode_summary"]

    for key in (
        "reason",
        "completion_time",
        "waypoints_reached",
        "path_length",
        "nav2_path_length",
        "path_length_ratio",
        "mean_cross_track_error",
        "max_cross_track_error",
        "min_obstacle_distance",
        "mean_speed",
        "mean_action_delta",
    ):
        assert key in summary
    env.close()


def test_odom_pose_is_never_consulted():
    """The env only ever reads a pose from ``BackendFrame.pose`` (the tf lookup)."""
    source = (__import__("pathlib").Path(__file__).resolve().parents[1]
              / "limo_delivery_rl_v2" / "delivery_env.py").read_text()

    assert "frame.pose" in source
    assert ".pose.pose" not in source


def test_pose_falls_back_only_for_observation_building():
    env = make_env()
    env.reset(seed=4)
    known = env._pose

    env._frame = BackendFrame(
        pose=None,
        linear_velocity=0.0,
        angular_velocity=0.0,
        lidar_metres=np.full(24, 8.0, dtype=np.float32),
        scan_age=0.0,
        odom_age=0.0,
        tf_age=float("inf"),
    )
    observation = env._get_obs()

    assert env._pose == Pose2D(known.x, known.y, known.yaw)
    assert env.observation_space.contains(observation)
    env.close()
