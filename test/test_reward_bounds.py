"""Per-step bounds on every reward term, and the waypoint-switch measurement."""

import numpy as np
import pytest

from limo_delivery_rl_v2.reward import REWARD_TERMS, RewardContext, compute_reward
from limo_delivery_rl_v2.state import RewardConfig, SafetyConfig, StopReason

REWARD = RewardConfig()
SAFETY = SafetyConfig()
ZERO = np.zeros(2, dtype=np.float32)


def context(**changes) -> RewardContext:
    """A neutral context: no progress, no danger, on the path, still running."""
    values = {
        "previous_waypoint_distance": 5.0,
        "current_waypoint_distance": 5.0,
        "waypoint_switched": False,
        "waypoint_bonus_granted": False,
        "min_obstacle_distance": 8.0,
        "cross_track_error": 0.0,
        "action": ZERO,
        "previous_action": ZERO,
        "stop_reason": StopReason.NONE,
    }
    values.update(changes)
    return RewardContext(**values)


def terms_of(**changes) -> dict[str, float]:
    """Return only the term breakdown for a context."""
    return compute_reward(context(**changes), REWARD, SAFETY)[1]


def test_all_documented_terms_are_reported():
    assert set(terms_of().keys()) == set(REWARD_TERMS)


def test_idle_step_only_pays_the_time_penalty():
    total, terms = compute_reward(context(), REWARD, SAFETY)

    assert terms["time"] == pytest.approx(-0.01)
    assert total == pytest.approx(-0.01)


@pytest.mark.parametrize("delta", [-10.0, -0.05, -0.01, 0.0, 0.01, 0.05, 10.0])
def test_progress_reward_is_clipped_to_half_a_point_per_step(delta):
    terms = terms_of(previous_waypoint_distance=5.0, current_waypoint_distance=5.0 - delta)

    assert -0.5 - 1e-9 <= terms["progress"] <= 0.5 + 1e-9


def test_nominal_forward_progress_is_about_two_tenths_per_step():
    # 0.42 m/s at 20 Hz advances ~0.021 m in one control step.
    terms = terms_of(previous_waypoint_distance=5.0, current_waypoint_distance=5.0 - 0.021)

    assert terms["progress"] == pytest.approx(0.21)


def test_waypoint_switch_saturates_progress_at_the_lower_bound():
    # Previous distance is to the old waypoint (inside 0.6 m), current is to the
    # new one 3 m away. This discontinuity is measured, not corrected.
    terms = terms_of(
        previous_waypoint_distance=0.55,
        current_waypoint_distance=3.0,
        waypoint_switched=True,
    )

    assert terms["progress"] == pytest.approx(-0.5)
    assert terms["progress_at_waypoint_switch"] == pytest.approx(-0.5)


def test_switch_measurement_is_zero_on_ordinary_steps():
    assert terms_of()["progress_at_waypoint_switch"] == 0.0


def test_switch_measurement_does_not_double_count_into_the_total():
    total, terms = compute_reward(
        context(
            previous_waypoint_distance=0.55,
            current_waypoint_distance=3.0,
            waypoint_switched=True,
        ),
        REWARD,
        SAFETY,
    )

    assert total == pytest.approx(terms["progress"] + terms["time"] + terms["danger"])


def test_waypoint_and_success_bonuses_match_the_specification():
    assert terms_of(waypoint_bonus_granted=True)["waypoint"] == pytest.approx(20.0)
    assert terms_of(stop_reason=StopReason.SUCCESS)["success"] == pytest.approx(100.0)


def test_collision_and_stuck_penalties_match_the_specification():
    assert terms_of(stop_reason=StopReason.COLLISION)["collision"] == pytest.approx(-100.0)
    assert terms_of(stop_reason=StopReason.STUCK)["stuck"] == pytest.approx(-80.0)


@pytest.mark.parametrize("distance", [0.0, 0.1, 0.25, 0.4, 0.8, 1.0, 8.0])
def test_danger_penalty_stays_within_one_tenth_per_step(distance):
    danger = terms_of(min_obstacle_distance=distance)["danger"]

    assert -0.10 - 1e-9 <= danger <= 0.0


def test_danger_penalty_activates_at_eighty_centimetres_and_saturates_at_collision():
    assert terms_of(min_obstacle_distance=0.80)["danger"] == pytest.approx(0.0)
    assert terms_of(min_obstacle_distance=0.525)["danger"] == pytest.approx(-0.05)
    assert terms_of(min_obstacle_distance=0.25)["danger"] == pytest.approx(-0.10)
    assert terms_of(min_obstacle_distance=0.10)["danger"] == pytest.approx(-0.10)


@pytest.mark.parametrize("error", [-10.0, -2.5, -1.0, 0.0, 1.25, 2.5, 10.0])
def test_deviation_penalty_stays_within_two_hundredths_per_step(error):
    deviation = terms_of(cross_track_error=error)["deviation"]

    assert -0.02 - 1e-9 <= deviation <= 0.0


def test_deviation_penalty_is_symmetric_and_saturates_at_the_deviation_limit():
    assert terms_of(cross_track_error=1.25)["deviation"] == pytest.approx(-0.01)
    assert terms_of(cross_track_error=-1.25)["deviation"] == pytest.approx(-0.01)
    assert terms_of(cross_track_error=2.5)["deviation"] == pytest.approx(-0.02)


def test_smoothness_penalty_stays_within_two_hundredths_per_step():
    worst = terms_of(
        action=np.array([1.0, 1.0], dtype=np.float32),
        previous_action=np.array([-1.0, -1.0], dtype=np.float32),
    )["smoothness"]

    assert worst == pytest.approx(-0.04)
    assert terms_of()["smoothness"] == pytest.approx(0.0)
    # A realistic one-step action change stays an order of magnitude smaller.
    modest = terms_of(
        action=np.array([0.1, 0.1], dtype=np.float32), previous_action=ZERO
    )["smoothness"]
    assert -0.02 <= modest <= 0.0


def test_shaping_terms_cannot_outweigh_forward_progress_on_a_normal_step():
    shaping = terms_of(min_obstacle_distance=0.25, cross_track_error=10.0)
    worst_case = shaping["danger"] + shaping["deviation"] + shaping["time"] - 0.02
    progress = terms_of(
        previous_waypoint_distance=5.0, current_waypoint_distance=5.0 - 0.021
    )["progress"]

    assert progress + worst_case > 0.0


def test_episode_outcomes_dominate_a_single_step_of_shaping():
    step_floor = -0.5 - 0.10 - 0.02 - 0.01 - 0.04

    assert abs(REWARD.collision_penalty) > 100 * abs(step_floor) / 100
    assert REWARD.collision_penalty < REWARD.stuck_penalty < 0.0 < REWARD.success_reward
    assert REWARD.waypoint_reward < REWARD.success_reward
