"""Training/evaluation entry points and the TensorBoard metric callback."""

import numpy as np
import pytest

from limo_delivery_rl_v2.evaluate_ppo import EPISODE_FIELDS, aggregate, write_reports
from limo_delivery_rl_v2.metrics import EpisodeMetrics
from limo_delivery_rl_v2.reward import REWARD_TERMS
from limo_delivery_rl_v2.state import StopReason
from limo_delivery_rl_v2.tb_callback import EpisodeMetricCallback
from limo_delivery_rl_v2.train_ppo import DEFAULT_HYPERPARAMETERS, build_parser

#: Every series the specification requires in TensorBoard.
REQUIRED_SCALARS = (
    "episode/count",
    "episode/success_count",
    "episode/collision_count",
    "episode/stuck_count",
    "episode/timeout_count",
    "episode/success_rate",
    "episode/waypoints_reached",
    "episode/completion_time",
    "episode/path_length_ratio",
    "episode/min_obstacle_distance",
    "episode/waypoint_switch_progress_sum",
) + tuple(f"reward/{term}" for term in REWARD_TERMS)


class _RecordingLogger:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def record(self, key, value, exclude=None):
        self.values[key] = value

    def record_mean(self, key, value, exclude=None):
        self.values[key] = value


class _FakeModel:
    """Stands in for the SB3 algorithm; ``BaseCallback.logger`` reads model.logger."""

    def __init__(self) -> None:
        self.logger = _RecordingLogger()
        self.num_timesteps = 0


def attached_callback() -> EpisodeMetricCallback:
    """A callback wired to a recording logger."""
    callback = EpisodeMetricCallback()
    callback.model = _FakeModel()
    return callback


def summary(reason: StopReason = StopReason.SUCCESS) -> dict[str, float | str]:
    """A realistic episode summary produced by EpisodeMetrics."""
    metrics = EpisodeMetrics(control_dt=0.05, nav2_path_length=9.5)
    terms = {term: 0.0 for term in REWARD_TERMS}
    terms["progress"] = 0.21
    terms["progress_at_waypoint_switch"] = -0.5
    metrics.update(
        reward=0.2,
        reward_terms=terms,
        travelled_distance=0.021,
        waypoints_reached=3,
        min_obstacle_distance=0.9,
        cross_track_error=0.05,
        linear_speed=0.42,
        action_delta=0.01,
    )
    return metrics.summary(reason)


# --------------------------------------------------------------- PPO config


def test_default_hyperparameters_match_the_specification():
    assert DEFAULT_HYPERPARAMETERS == {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.15,
        "ent_coef": 0.003,
    }


def test_training_defaults_match_the_specification():
    args = build_parser().parse_args([])

    # total_timesteps is an operator dial, not a contract; only its type matters.
    assert isinstance(args.total_timesteps, int) and args.total_timesteps > 0
    assert args.checkpoint_freq == 50_000
    assert args.seed == 42
    assert args.device == "cpu"
    assert args.log_std_init == pytest.approx(-2.0)


def test_training_does_not_start_a_run_at_import_time():
    import limo_delivery_rl_v2.train_ppo as module

    assert callable(module.main)


# ------------------------------------------------------------- TB callback


def test_callback_records_every_required_scalar():
    callback = attached_callback()
    callback.locals = {"dones": [True], "infos": [{"episode_summary": summary()}]}

    callback._on_step()

    missing = [key for key in REQUIRED_SCALARS if key not in callback.logger.values]
    assert missing == []


def test_callback_counts_outcomes_and_the_success_rate():
    callback = attached_callback()
    for reason in (StopReason.SUCCESS, StopReason.COLLISION, StopReason.SUCCESS, StopReason.STUCK):
        callback.locals = {"dones": [True], "infos": [{"episode_summary": summary(reason)}]}
        callback._on_step()

    values = callback.logger.values
    assert values["episode/count"] == 4
    assert values["episode/success_count"] == 2
    assert values["episode/collision_count"] == 1
    assert values["episode/stuck_count"] == 1
    assert values["episode/success_rate"] == pytest.approx(0.5)


def test_callback_ignores_steps_without_a_finished_episode():
    callback = attached_callback()
    callback.locals = {"dones": [False], "infos": [{"episode_summary": summary()}]}

    callback._on_step()

    assert callback.episode_count == 0
    assert callback.logger.values == {}


def test_callback_records_the_waypoint_switch_measurement():
    callback = attached_callback()
    callback.locals = {"dones": [True], "infos": [{"episode_summary": summary()}]}

    callback._on_step()

    assert callback.logger.values["episode/waypoint_switch_progress_sum"] == pytest.approx(-0.5)
    assert callback.logger.values["reward/progress_at_waypoint_switch"] == pytest.approx(-0.5)


# ---------------------------------------------------------------- evaluation


def episode(reason: StopReason, **changes) -> dict[str, float | str]:
    """A minimal per-episode evaluation record."""
    record: dict[str, float | str] = {
        "reason": reason.value,
        "steps": 400.0,
        "completion_time": 20.0,
        "waypoints_reached": 3.0,
        "path_length": 9.6,
        "nav2_path_length": 9.5,
        "path_length_ratio": 1.01,
        "mean_cross_track_error": 0.08,
        "max_cross_track_error": 0.4,
        "min_obstacle_distance": 0.9,
        "mean_speed": 0.4,
        "mean_action_delta": 0.02,
        "total_reward": 180.0,
    }
    record.update(changes)
    return record


def test_aggregate_reports_every_specified_evaluation_metric():
    result = aggregate(
        [
            episode(StopReason.SUCCESS),
            episode(StopReason.COLLISION, waypoints_reached=1.0),
            episode(StopReason.STUCK, waypoints_reached=2.0),
            episode(StopReason.TIMEOUT, waypoints_reached=2.0),
        ]
    )

    assert result["success_rate"] == pytest.approx(0.25)
    assert result["collision_rate"] == pytest.approx(0.25)
    assert result["stuck_rate"] == pytest.approx(0.25)
    assert result["timeout_rate"] == pytest.approx(0.25)
    assert result["mean_waypoints_reached"] == pytest.approx(2.0)
    assert result["mean_path_length_ratio"] == pytest.approx(1.01)
    assert result["max_cross_track_error"] == pytest.approx(0.4)
    for key in ("mean_completion_time", "mean_cross_track_error", "mean_speed", "mean_action_delta"):
        assert key in result


def test_completion_time_averages_successful_episodes_only():
    result = aggregate(
        [
            episode(StopReason.SUCCESS, completion_time=20.0),
            episode(StopReason.COLLISION, completion_time=5.0),
        ]
    )

    assert result["mean_completion_time"] == pytest.approx(20.0)


def test_aggregate_handles_an_empty_run():
    result = aggregate([])

    assert result["episodes"] == 0.0
    assert result["success_rate"] == 0.0


def test_reports_are_written_as_json_and_csv(tmp_path):
    import csv
    import json

    episodes = [episode(StopReason.SUCCESS), episode(StopReason.COLLISION)]
    json_path = tmp_path / "eval.json"
    csv_path = tmp_path / "eval.csv"

    write_reports(episodes, aggregate(episodes), json_path, csv_path)

    payload = json.loads(json_path.read_text())
    assert payload["summary"]["success_rate"] == pytest.approx(0.5)
    assert len(payload["episodes"]) == 2

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert list(rows[0].keys()) == list(EPISODE_FIELDS)


# ------------------------------------------------------------------ metrics


def test_metrics_track_path_ratio_and_extremes():
    metrics = EpisodeMetrics(control_dt=0.05, nav2_path_length=10.0)
    terms = {term: 0.0 for term in REWARD_TERMS}

    for error, distance in ((0.1, 2.0), (-0.9, 0.5), (0.2, 3.0)):
        metrics.update(
            reward=1.0,
            reward_terms=terms,
            travelled_distance=4.0,
            waypoints_reached=1,
            min_obstacle_distance=distance,
            cross_track_error=error,
            linear_speed=0.3,
            action_delta=0.05,
        )

    result = metrics.summary(StopReason.TIMEOUT)
    assert result["path_length"] == pytest.approx(12.0)
    assert result["path_length_ratio"] == pytest.approx(1.2)
    assert result["max_cross_track_error"] == pytest.approx(0.9)
    assert result["min_obstacle_distance"] == pytest.approx(0.5)
    assert result["completion_time"] == pytest.approx(0.15)
    assert result["mean_speed"] == pytest.approx(0.3)


def test_a_fresh_metrics_instance_carries_nothing_over():
    """Every field resets because a new episode builds a new instance."""
    import dataclasses

    used = EpisodeMetrics(control_dt=0.05, nav2_path_length=10.0)
    used.update(
        reward=5.0,
        reward_terms={term: 1.0 for term in REWARD_TERMS},
        travelled_distance=1.0,
        waypoints_reached=2,
        min_obstacle_distance=0.4,
        cross_track_error=1.0,
        linear_speed=0.2,
        action_delta=0.1,
    )
    fresh = EpisodeMetrics(control_dt=0.05, nav2_path_length=8.0)

    carried = [
        field.name
        for field in dataclasses.fields(EpisodeMetrics)
        if field.name not in ("control_dt", "nav2_path_length")
        and getattr(used, field.name) != getattr(fresh, field.name)
    ]
    assert carried, "the update should have moved something"
    result = fresh.summary(StopReason.NONE)
    assert result["path_length"] == 0.0
    assert result["waypoints_reached"] == 0.0
    assert result["total_reward"] == 0.0
    assert np.isfinite(result["min_obstacle_distance"])
