"""Gymnasium environment: Nav2 plans the global path, PPO drives the robot.

Architecture A.  Nav2 contributes a single ``ComputePathThroughPoses`` path per
episode and nothing else; the policy outputs ``(v, omega)`` which, after the hard
safety gate, is published directly on ``/cmd_vel``.  There is no residual term
added to a Nav2 steering command and no distance-proportional slowdown.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from limo_delivery_rl_v2.action_scaler import ACTION_DIM, ActionScaler, RateLimiter, sanitize_action
from limo_delivery_rl_v2.env_backend import BackendFrame, EnvBackend, OfflineBackend
from limo_delivery_rl_v2.geometry import Pose2D, euclidean
from limo_delivery_rl_v2.map_utils import validate_existing_map
from limo_delivery_rl_v2.metrics import EpisodeMetrics
from limo_delivery_rl_v2.observation import (
    LidarFrameStack,
    build_observation,
    observation_space_bounds,
)
from limo_delivery_rl_v2.path_tracker import PathTracker, path_length
from limo_delivery_rl_v2.reward import RewardContext, compute_reward
from limo_delivery_rl_v2.safety_controller import apply_safety_limits
from limo_delivery_rl_v2.state import (
    DeliveryEnvConfig,
    ObstacleSpec,
    SafetyMode,
    StopReason,
    observation_dim,
)
from limo_delivery_rl_v2.termination import TerminationContext, termination_status
from limo_delivery_rl_v2.waypoint_manager import WaypointManager


class LimoWaypointRLEnv(gym.Env):
    """Waypoint-following environment with LiDAR-only obstacle avoidance."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: DeliveryEnvConfig | None = None,
        *,
        enable_ros: bool = True,
        backend: EnvBackend | None = None,
    ) -> None:
        """Build the spaces and either attach to ROS or fall back to the offline model."""
        super().__init__()
        self.config = config or DeliveryEnvConfig()
        validate_existing_map(self.config.map)

        low, high = observation_space_bounds(self.config.observation)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
            high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.path_tracker = PathTracker(self.config.observation)
        self.waypoint_manager = WaypointManager(self.config.waypoints, self.config.episode)
        self.lidar_stack = LidarFrameStack(self.config.observation)
        self.action_scaler = ActionScaler(self.config.control)
        self.rate_limiter = RateLimiter(self.config.control)
        self.metrics = EpisodeMetrics(control_dt=self.config.control.control_dt)

        self.enable_ros = enable_ros
        self._backend = backend if backend is not None else self._make_backend(enable_ros)

        self._pose = Pose2D(0.0, 0.0, 0.0)
        self._frame = self._idle_frame()
        self._previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._previous_linear = 0.0
        self._previous_angular = 0.0
        self._previous_waypoint_distance = 0.0
        self._steps = 0
        self._stuck_steps = 0
        self._sensor_lost_steps = 0
        self._path_available = False
        self._stop_reason = StopReason.NONE
        self.episode_obstacles: tuple[ObstacleSpec, ...] = ()

    # ------------------------------------------------------------------ setup

    def _make_backend(self, enable_ros: bool) -> EnvBackend:
        """Instantiate the ROS backend, importing ``rclpy`` only when needed."""
        if not enable_ros:
            return OfflineBackend(self.config)
        from limo_delivery_rl_v2.ros_backend import RosBackend

        return RosBackend(self.config)

    def _idle_frame(self) -> BackendFrame:
        """A neutral frame used before the first backend response arrives."""
        return BackendFrame(
            pose=None,
            linear_velocity=0.0,
            angular_velocity=0.0,
            lidar_metres=np.full(
                self.config.observation.lidar_bins,
                self.config.observation.lidar_max_range,
                dtype=np.float32,
            ),
            scan_age=float("inf"),
            odom_age=float("inf"),
            tf_age=float("inf"),
        )

    def _sample_obstacles(self) -> tuple[ObstacleSpec, ...]:
        """Draw this episode's obstacle poses.

        Randomised poses stay inside ranges verified free in ``map.pgm``, and
        every one of them blocks the path centreline, so the policy has to pick
        a side from ``/scan`` instead of memorising one turn direction.
        """
        obstacles = self.config.obstacles
        if not obstacles.enabled:
            return ()
        if not obstacles.randomize:
            return obstacles.specs
        from dataclasses import replace

        return tuple(
            replace(
                spec,
                x=float(self.np_random.uniform(*obstacles.x_range)),
                y=float(self.np_random.uniform(*obstacles.y_range)),
            )
            for spec in obstacles.specs
        )

    def _sample_start_pose(self) -> tuple[float, float, float]:
        """Draw a jittered start pose around the configured nominal pose."""
        start = self.config.start_pose
        return (
            start.x + float(self.np_random.uniform(-start.x_jitter, start.x_jitter)),
            start.y + float(self.np_random.uniform(-start.y_jitter, start.y_jitter)),
            start.yaw + float(self.np_random.uniform(-start.yaw_jitter, start.yaw_jitter)),
        )

    # ------------------------------------------------------------------- gym

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        """Reset the simulator, plan the path, spawn obstacles and observe."""
        super().reset(seed=seed)
        options = options or {}
        start_pose = options.get("start_pose") or self._sample_start_pose()

        self._steps = 0
        self._stuck_steps = 0
        self._sensor_lost_steps = 0
        self._previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._previous_linear = 0.0
        self._previous_angular = 0.0
        self._stop_reason = StopReason.NONE
        self.waypoint_manager.reset(start_xy=(start_pose[0], start_pose[1]))

        self.episode_obstacles = self._sample_obstacles()
        path, frame = self._backend.reset_episode(
            start_pose, self.config.waypoints, self.episode_obstacles
        )
        self._frame = frame
        if frame.pose is None:
            raise RuntimeError(
                "map->base_link is unavailable after reset; refusing to start an episode "
                "on an unknown pose"
            )
        self._pose = frame.pose
        self.path_tracker.set_path(path)
        self._path_available = self.path_tracker.is_available
        if not self._path_available:
            raise RuntimeError("Nav2 returned a path with fewer than two poses")
        self.path_tracker.update(self._pose)

        self.lidar_stack.reset(self._normalized_lidar(frame))
        self._previous_waypoint_distance = self.waypoint_manager.measure(self._pose)[0]
        self.metrics = EpisodeMetrics(
            control_dt=self.config.control.control_dt,
            nav2_path_length=path_length(self.path_tracker.points),
        )
        return self._get_obs(), {"reason": StopReason.NONE.value}

    def step(self, action):
        """Apply one policy action and advance the simulation by one control period."""
        normalized, action_valid = sanitize_action(action)
        linear_target, angular_target = self.action_scaler.scale(normalized)
        linear, angular, safety_mode = apply_safety_limits(
            linear_target=linear_target,
            angular_target=angular_target,
            previous_linear=self._previous_linear,
            previous_angular=self._previous_angular,
            min_obstacle_distance=float(self._frame.lidar_metres.min()),
            scan_age=self._frame.scan_age,
            odom_age=self._frame.odom_age,
            tf_age=self._frame.tf_age,
            action_valid=action_valid,
            control=self.config.control,
            safety=self.config.safety,
            rate_limiter=self.rate_limiter,
        )

        frame = self._backend.apply_command(linear, angular)
        self._frame = frame
        previous_pose = self._pose
        # A missing tf snapshot keeps the last known pose only so an observation
        # can still be built; the staleness itself is reported through tf_age and
        # drives both the safety gate and the sensor-loss termination.
        self._pose = frame.pose if frame.pose is not None else previous_pose

        self.path_tracker.update(self._pose)
        cross_track_error = self.path_tracker.cross_track_error(self._pose)
        waypoint = self.waypoint_manager.update(self._pose)
        min_obstacle_distance = float(frame.lidar_metres.min())
        self.lidar_stack.push(self._normalized_lidar(frame))

        self._steps += 1
        self._sensor_lost_steps = (
            self._sensor_lost_steps + 1 if self._sensors_lost(frame) else 0
        )
        # A hard safety stop must not also be punished as being stuck: the robot
        # was not allowed to move in the first place.
        moving = abs(frame.linear_velocity) >= self.config.episode.stuck_speed_threshold
        if safety_mode is SafetyMode.OK and not moving and not self.waypoint_manager.completed:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        terminated, truncated, reason = self._check_done(
            TerminationContext(
                min_obstacle_distance=min_obstacle_distance,
                waypoints_completed=self.waypoint_manager.completed,
                path_available=self._path_available,
                sensor_lost=self._sensor_lost_steps >= self.config.episode.sensor_lost_step_limit,
                cross_track_error=cross_track_error,
                steps=self._steps,
                stuck_steps=self._stuck_steps,
            )
        )
        self._stop_reason = StopReason(reason)

        reward, terms = self._compute_reward(
            RewardContext(
                previous_waypoint_distance=self._previous_waypoint_distance,
                current_waypoint_distance=waypoint.distance,
                waypoint_switched=waypoint.switched,
                waypoint_bonus_granted=waypoint.bonus_granted,
                min_obstacle_distance=min_obstacle_distance,
                cross_track_error=cross_track_error,
                action=normalized,
                previous_action=self._previous_action,
                stop_reason=self._stop_reason,
            )
        )

        self.metrics.update(
            reward=reward,
            reward_terms=terms,
            travelled_distance=euclidean(
                (previous_pose.x, previous_pose.y), (self._pose.x, self._pose.y)
            ),
            waypoints_reached=self.waypoint_manager.reached_count,
            min_obstacle_distance=min_obstacle_distance,
            cross_track_error=cross_track_error,
            linear_speed=abs(frame.linear_velocity),
            action_delta=float(np.mean(np.abs(normalized - self._previous_action))),
        )

        if terminated or truncated:
            self._publish_stop()

        self._previous_waypoint_distance = waypoint.distance
        self._previous_action = normalized
        self._previous_linear = 0.0 if (terminated or truncated) else linear
        self._previous_angular = 0.0 if (terminated or truncated) else angular

        info: dict[str, object] = {
            "reason": reason,
            "safety_mode": safety_mode.value,
            "waypoint_index": self.waypoint_manager.index,
            "waypoints_reached": self.waypoint_manager.reached_count,
            "min_obstacle_distance": min_obstacle_distance,
            "cross_track_error": cross_track_error,
        }
        info.update({f"reward/{name}": value for name, value in terms.items()})
        if terminated or truncated:
            info["episode_summary"] = self.metrics.summary(self._stop_reason)
        return self._get_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        """Release the backend."""
        self._backend.close()

    # --------------------------------------------------------------- helpers

    def _normalized_lidar(self, frame: BackendFrame) -> NDArray[np.float32]:
        """Scale a frame's binned ranges into ``[0, 1]``."""
        max_range = self.config.observation.lidar_max_range
        return np.clip(
            np.asarray(frame.lidar_metres, dtype=np.float32) / np.float32(max_range), 0.0, 1.0
        ).astype(np.float32)

    def _sensors_lost(self, frame: BackendFrame) -> bool:
        """Whether any sensing input exceeded its timeout on this step."""
        safety = self.config.safety
        return (
            frame.scan_age > safety.sensor_timeout_sec
            or frame.odom_age > safety.sensor_timeout_sec
            or frame.tf_age > safety.tf_timeout_sec
        )

    def _publish_stop(self) -> None:
        """Command zero velocity at the end of an episode."""
        self._backend.apply_command(0.0, 0.0)

    def _get_obs(self) -> NDArray[np.float32]:
        """Assemble the 112-dimensional normalized observation."""
        distance, heading_error = self.waypoint_manager.measure(self._pose)
        return build_observation(
            lidar_stack=self.lidar_stack.flat(),
            relative_path=self.path_tracker.relative_lookahead(self._pose),
            waypoint_distance=distance,
            waypoint_heading_error=heading_error,
            linear_velocity=self._frame.linear_velocity,
            angular_velocity=self._frame.angular_velocity,
            previous_action=self._previous_action,
            observation=self.config.observation,
            control=self.config.control,
        )

    def _compute_reward(self, context: RewardContext) -> tuple[float, dict[str, float]]:
        """Evaluate the bounded reward terms for one step."""
        return compute_reward(context, self.config.reward, self.config.safety)

    def _check_done(self, context: TerminationContext) -> tuple[bool, bool, str]:
        """Evaluate the termination priority order for one step."""
        terminated, truncated, reason = termination_status(
            context, self.config.episode, self.config.safety
        )
        return terminated, truncated, reason.value


def main() -> None:
    """Smoke-check the environment without starting a training run."""
    parser = argparse.ArgumentParser(
        description="Smoke-check the LIMO waypoint RL environment."
    )
    parser.add_argument("--no-ros", action="store_true", help="Use the offline unicycle backend.")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to run.")
    args = parser.parse_args()

    env = LimoWaypointRLEnv(enable_ros=not args.no_ros)
    observation, _ = env.reset(seed=42)
    expected = observation_dim(env.config.observation)
    print(
        f"observation_shape={observation.shape} (expected {expected}) "
        f"action_shape={env.action_space.shape}"
    )
    total = 0.0
    for _ in range(max(args.steps, 1)):
        observation, reward, terminated, truncated, info = env.step(
            np.array([0.5, 0.0], dtype=np.float32)
        )
        total += reward
        if terminated or truncated:
            break
    print(f"total_reward={total:.3f} reason={info['reason']} safety={info['safety_mode']}")
    env.close()


if __name__ == "__main__":
    main()
