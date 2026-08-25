"""TensorBoard logging of per-episode outcomes and reward-term breakdowns."""

from __future__ import annotations

from collections.abc import Mapping

from stable_baselines3.common.callbacks import BaseCallback

from limo_delivery_rl_v2.reward import REWARD_TERMS
from limo_delivery_rl_v2.state import StopReason

#: Episode outcomes that get their own counter and rate.
COUNTED_REASONS: tuple[StopReason, ...] = (
    StopReason.SUCCESS,
    StopReason.COLLISION,
    StopReason.STUCK,
    StopReason.TIMEOUT,
    StopReason.PATH_DEVIATION,
    StopReason.PATH_FAILED,
    StopReason.SENSOR_TIMEOUT,
)


class EpisodeMetricCallback(BaseCallback):
    """Records episode outcomes, path quality and reward terms to TensorBoard.

    Per-episode quantities use ``record_mean`` so each TensorBoard point is the
    average over the rollout rather than whichever episode happened to finish
    last.  Counters are cumulative.
    """

    def __init__(self, *, print_every_episodes: int = 0, verbose: int = 0) -> None:
        """Optionally also print a progress line every ``print_every_episodes``."""
        super().__init__(verbose=verbose)
        self.print_every_episodes = max(int(print_every_episodes), 0)
        self.episode_count = 0
        self.reason_counts: dict[str, int] = {reason.value: 0 for reason in COUNTED_REASONS}

    def _on_step(self) -> bool:
        """Fold every finished episode in this step into the logger."""
        for done, info in zip(self.locals.get("dones", ()), self.locals.get("infos", ())):
            if not bool(done):
                continue
            summary = info.get("episode_summary")
            if not isinstance(summary, Mapping):
                continue
            self._record_episode(summary)
        return True

    def _record_episode(self, summary: Mapping[str, float | str]) -> None:
        """Write one episode's metrics."""
        self.episode_count += 1
        reason = str(summary.get("reason", StopReason.NONE.value))
        if reason in self.reason_counts:
            self.reason_counts[reason] += 1

        self.logger.record("episode/count", self.episode_count)
        for name, count in self.reason_counts.items():
            self.logger.record(f"episode/{name}_count", count)
        self.logger.record(
            "episode/success_rate",
            self.reason_counts[StopReason.SUCCESS.value] / max(self.episode_count, 1),
        )
        for key in (
            "waypoints_reached",
            "completion_time",
            "path_length_ratio",
            "min_obstacle_distance",
            "waypoint_switch_progress_sum",
            "mean_cross_track_error",
            "max_cross_track_error",
            "mean_speed",
            "mean_action_delta",
        ):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                self.logger.record_mean(f"episode/{key}", float(value))
        for term in REWARD_TERMS:
            value = summary.get(f"reward_{term}")
            if isinstance(value, (int, float)):
                self.logger.record_mean(f"reward/{term}", float(value))

        if self.print_every_episodes and self.episode_count % self.print_every_episodes == 0:
            self._print_progress(summary, reason)

    def _print_progress(self, summary: Mapping[str, float | str], reason: str) -> None:
        """Print a one-line human-readable progress update."""
        success_rate = 100.0 * self.reason_counts[StopReason.SUCCESS.value] / max(
            self.episode_count, 1
        )
        print(
            f"[진행] episode={self.episode_count}"
            f" | reason={reason}"
            f" | waypoints={summary.get('waypoints_reached', 0)}"
            f" | reward={float(summary.get('total_reward', 0.0)):.2f}"
            f" | success_rate={success_rate:.1f}%"
            f" | timestep={self.num_timesteps:,}",
            flush=True,
        )
