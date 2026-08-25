"""Deterministic evaluation of a trained waypoint policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")

from limo_delivery_rl_v2.delivery_env import LimoWaypointRLEnv  # noqa: E402
from limo_delivery_rl_v2.state import DeliveryEnvConfig, StopReason, stage_config  # noqa: E402

#: Per-episode fields written to the CSV/JSON report.
EPISODE_FIELDS: tuple[str, ...] = (
    "reason",
    "steps",
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
    "total_reward",
)


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean, or ``0.0`` for an empty sequence."""
    return float(statistics.fmean(values)) if values else 0.0


def aggregate(episodes: Sequence[dict[str, float | str]]) -> dict[str, float]:
    """Summarise per-episode records into the reported evaluation metrics."""
    count = max(len(episodes), 1)

    def rate(reason: StopReason) -> float:
        return sum(1 for e in episodes if e.get("reason") == reason.value) / count

    successes = [e for e in episodes if e.get("reason") == StopReason.SUCCESS.value]
    return {
        "episodes": float(len(episodes)),
        "success_rate": rate(StopReason.SUCCESS),
        "collision_rate": rate(StopReason.COLLISION),
        "stuck_rate": rate(StopReason.STUCK),
        "timeout_rate": rate(StopReason.TIMEOUT),
        "path_deviation_rate": rate(StopReason.PATH_DEVIATION),
        "mean_waypoints_reached": _mean([float(e["waypoints_reached"]) for e in episodes]),
        "mean_completion_time": _mean([float(e["completion_time"]) for e in successes]),
        "mean_path_length": _mean([float(e["path_length"]) for e in episodes]),
        "mean_nav2_path_length": _mean([float(e["nav2_path_length"]) for e in episodes]),
        "mean_path_length_ratio": _mean([float(e["path_length_ratio"]) for e in episodes]),
        "mean_cross_track_error": _mean([float(e["mean_cross_track_error"]) for e in episodes]),
        "max_cross_track_error": max(
            [float(e["max_cross_track_error"]) for e in episodes], default=0.0
        ),
        "mean_min_obstacle_distance": _mean(
            [float(e["min_obstacle_distance"]) for e in episodes]
        ),
        "mean_speed": _mean([float(e["mean_speed"]) for e in episodes]),
        "mean_action_delta": _mean([float(e["mean_action_delta"]) for e in episodes]),
        "mean_total_reward": _mean([float(e["total_reward"]) for e in episodes]),
    }


def write_reports(
    episodes: Sequence[dict[str, float | str]],
    summary: dict[str, float],
    json_path: Path | None,
    csv_path: Path | None,
) -> None:
    """Persist the evaluation report as JSON and/or CSV."""
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"summary": summary, "episodes": list(episodes)}, indent=2),
            encoding="utf-8",
        )
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EPISODE_FIELDS))
            writer.writeheader()
            for episode in episodes:
                writer.writerow({field: episode.get(field, "") for field in EPISODE_FIELDS})


def main() -> None:
    """Run deterministic evaluation episodes and report the aggregate metrics."""
    parser = argparse.ArgumentParser(description="Evaluate a trained waypoint policy.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true", help="Sample instead of using the mean action.")
    parser.add_argument("--no-ros", action="store_true")
    defaults = DeliveryEnvConfig()
    parser.add_argument(
        "--waypoints",
        type=int,
        default=len(defaults.waypoints),
        help="Evaluate the stage that uses only the first N waypoints.",
    )
    parser.add_argument("--waypoint-radius", type=float, default=defaults.episode.waypoint_radius)
    parser.add_argument(
        "--waypoint-hold-steps", type=int, default=defaults.episode.waypoint_hold_steps
    )
    parser.add_argument(
        "--waypoint-capture-width",
        type=float,
        default=defaults.episode.waypoint_capture_width,
    )
    parser.add_argument("--randomize-obstacles", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("runs/limo_delivery_rl_v2/eval.json"))
    parser.add_argument("--csv-out", type=Path, default=Path("runs/limo_delivery_rl_v2/eval.csv"))
    args = parser.parse_args()

    from stable_baselines3 import PPO

    config = stage_config(
        waypoint_count=args.waypoints,
        waypoint_radius=args.waypoint_radius,
        waypoint_hold_steps=args.waypoint_hold_steps,
        waypoint_capture_width=args.waypoint_capture_width,
        obstacles_randomized=args.randomize_obstacles,
    )
    env = LimoWaypointRLEnv(config=config, enable_ros=not args.no_ros)
    model = PPO.load(args.model_path, env=env, device="cpu")
    episodes: list[dict[str, float | str]] = []
    try:
        for index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + index)
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=not args.stochastic)
                observation, _reward, terminated, truncated, info = env.step(action)
            summary = dict(info.get("episode_summary", {"reason": StopReason.NONE.value}))
            episodes.append(summary)
            print(
                f"episode={index} reason={summary.get('reason')} "
                f"waypoints={summary.get('waypoints_reached')} "
                f"reward={float(summary.get('total_reward', 0.0)):.2f}"
            )
    finally:
        env.close()

    summary = aggregate(episodes)
    write_reports(episodes, summary, args.json_out, args.csv_out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
