"""Episode termination decisions with an explicit, tested priority order."""

from __future__ import annotations

from dataclasses import dataclass

from limo_delivery_rl_v2.state import EpisodeConfig, SafetyConfig, StopReason


@dataclass(frozen=True, slots=True)
class TerminationContext:
    """Signals consulted when deciding whether an episode ends."""

    min_obstacle_distance: float
    waypoints_completed: bool
    path_available: bool
    sensor_lost: bool
    cross_track_error: float
    steps: int
    stuck_steps: int


def termination_status(
    context: TerminationContext,
    episode: EpisodeConfig,
    safety: SafetyConfig,
) -> tuple[bool, bool, StopReason]:
    """Return ``(terminated, truncated, reason)``.

    Priority, highest first:

    1. ``COLLISION``   -- safety-critical, and the reward must reflect the crash.
    2. ``SUCCESS``     -- a completed run is never downgraded to a truncation.
    3. ``PATH_FAILED`` -- without a path the remaining checks are meaningless.
    4. ``SENSOR_TIMEOUT`` -- stale sensors make the remaining checks unreliable.
    5. ``PATH_DEVIATION``
    6. ``STUCK``
    7. ``TIMEOUT``

    Only the first two are MDP terminations; the rest are truncations.
    """
    if context.min_obstacle_distance <= safety.collision_distance:
        return True, False, StopReason.COLLISION
    if context.waypoints_completed:
        return True, False, StopReason.SUCCESS
    if not context.path_available:
        return False, True, StopReason.PATH_FAILED
    if context.sensor_lost:
        return False, True, StopReason.SENSOR_TIMEOUT
    if abs(context.cross_track_error) > safety.max_path_deviation:
        return False, True, StopReason.PATH_DEVIATION
    if context.stuck_steps >= episode.stuck_step_limit:
        return False, True, StopReason.STUCK
    if context.steps >= episode.max_steps:
        return False, True, StopReason.TIMEOUT
    return False, False, StopReason.NONE
