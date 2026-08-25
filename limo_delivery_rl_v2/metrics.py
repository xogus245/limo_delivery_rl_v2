"""Per-episode metric accumulation shared by training callbacks and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from limo_delivery_rl_v2.state import StopReason


@dataclass
class EpisodeMetrics:
    """Accumulates one episode's statistics and emits them as a flat dict.

    Construct a fresh instance per episode rather than clearing one in place; a
    reset method has to re-declare every field and silently skips any field added
    later.
    """

    control_dt: float = 0.05
    steps: int = 0
    nav2_path_length: float = 0.0
    travelled_length: float = 0.0
    waypoints_reached: int = 0
    min_obstacle_distance: float = float("inf")
    cross_track_error_sum: float = 0.0
    max_cross_track_error: float = 0.0
    speed_sum: float = 0.0
    action_delta_sum: float = 0.0
    waypoint_switch_progress_sum: float = 0.0
    reward_sum: float = 0.0
    reward_terms: dict[str, float] = field(default_factory=dict)

    def update(
        self,
        *,
        reward: float,
        reward_terms: dict[str, float],
        travelled_distance: float,
        waypoints_reached: int,
        min_obstacle_distance: float,
        cross_track_error: float,
        linear_speed: float,
        action_delta: float,
    ) -> None:
        """Fold one control step into the accumulators."""
        self.steps += 1
        self.reward_sum += reward
        for name, value in reward_terms.items():
            self.reward_terms[name] = self.reward_terms.get(name, 0.0) + value
        self.waypoint_switch_progress_sum += reward_terms.get(
            "progress_at_waypoint_switch", 0.0
        )
        self.travelled_length += travelled_distance
        self.waypoints_reached = waypoints_reached
        self.min_obstacle_distance = min(self.min_obstacle_distance, min_obstacle_distance)
        self.cross_track_error_sum += abs(cross_track_error)
        self.max_cross_track_error = max(self.max_cross_track_error, abs(cross_track_error))
        self.speed_sum += linear_speed
        self.action_delta_sum += action_delta

    def summary(self, reason: StopReason) -> dict[str, float | str]:
        """Return the episode summary placed into the terminal ``info`` dict."""
        steps = max(self.steps, 1)
        ratio = (
            self.travelled_length / self.nav2_path_length
            if self.nav2_path_length > 1e-6
            else 0.0
        )
        summary: dict[str, float | str] = {
            "reason": reason.value,
            "steps": float(self.steps),
            "completion_time": self.steps * self.control_dt,
            "waypoints_reached": float(self.waypoints_reached),
            "path_length": self.travelled_length,
            "nav2_path_length": self.nav2_path_length,
            "path_length_ratio": ratio,
            "min_obstacle_distance": (
                0.0 if self.min_obstacle_distance == float("inf") else self.min_obstacle_distance
            ),
            "mean_cross_track_error": self.cross_track_error_sum / steps,
            "max_cross_track_error": self.max_cross_track_error,
            "mean_speed": self.speed_sum / steps,
            "mean_action_delta": self.action_delta_sum / steps,
            "waypoint_switch_progress_sum": self.waypoint_switch_progress_sum,
            "total_reward": self.reward_sum,
        }
        for name, value in self.reward_terms.items():
            summary[f"reward_{name}"] = value
        return summary
