"""Bounded per-step reward terms.

Every shaping term is clipped so that no single step can swamp the episode-level
outcomes (collision ``-100``, stuck ``-80``, completion ``+100``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from limo_delivery_rl_v2.state import RewardConfig, SafetyConfig, StopReason


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Everything the reward function needs from one control step."""

    previous_waypoint_distance: float
    current_waypoint_distance: float
    waypoint_switched: bool
    waypoint_bonus_granted: bool
    min_obstacle_distance: float
    cross_track_error: float
    action: NDArray[np.float32]
    previous_action: NDArray[np.float32]
    stop_reason: StopReason


#: Names of every reward term, in TensorBoard logging order.
REWARD_TERMS: tuple[str, ...] = (
    "progress",
    "progress_at_waypoint_switch",
    "waypoint",
    "success",
    "collision",
    "danger",
    "deviation",
    "time",
    "smoothness",
    "stuck",
)


def compute_reward(
    context: RewardContext,
    reward: RewardConfig,
    safety: SafetyConfig,
) -> tuple[float, dict[str, float]]:
    """Return ``(total_reward, per_term_breakdown)``.

    The ``progress`` term is intentionally discontinuous at a waypoint switch:
    ``previous_waypoint_distance`` refers to the old waypoint while
    ``current_waypoint_distance`` refers to the new one, so the clipped delta
    saturates at ``-0.5``.  This is measured rather than corrected in the first
    training round -- see ``progress_at_waypoint_switch``.
    """
    delta = float(
        np.clip(
            context.previous_waypoint_distance - context.current_waypoint_distance,
            -reward.progress_delta_clip,
            reward.progress_delta_clip,
        )
    )
    progress = reward.progress_gain * delta

    danger_span = max(safety.danger_distance - safety.collision_distance, 1e-6)
    danger_ratio = float(
        np.clip((safety.danger_distance - context.min_obstacle_distance) / danger_span, 0.0, 1.0)
    )
    danger = -reward.danger_penalty_scale * danger_ratio

    deviation = -reward.deviation_penalty_scale * float(
        np.clip(abs(context.cross_track_error) / reward.deviation_reference, 0.0, 1.0)
    )

    smoothness = -reward.smoothness_penalty_scale * float(
        np.mean(np.abs(np.asarray(context.action, dtype=np.float64)
                       - np.asarray(context.previous_action, dtype=np.float64)))
    )

    terms = {
        "progress": progress,
        "progress_at_waypoint_switch": progress if context.waypoint_switched else 0.0,
        "waypoint": reward.waypoint_reward if context.waypoint_bonus_granted else 0.0,
        "success": reward.success_reward if context.stop_reason is StopReason.SUCCESS else 0.0,
        "collision": reward.collision_penalty if context.stop_reason is StopReason.COLLISION else 0.0,
        "danger": danger,
        "deviation": deviation,
        "time": reward.time_penalty,
        "smoothness": smoothness,
        "stuck": reward.stuck_penalty if context.stop_reason is StopReason.STUCK else 0.0,
    }
    # progress_at_waypoint_switch is a measurement of `progress`, not an extra payout.
    total = sum(value for name, value in terms.items() if name != "progress_at_waypoint_switch")
    return float(total), terms
