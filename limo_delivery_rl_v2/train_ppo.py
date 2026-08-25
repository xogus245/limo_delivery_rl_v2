"""PPO training entry point for the LIMO waypoint RL environment.

Previous checkpoints are not loadable: the action space changed from a single
steering value to ``(v, omega)``, so training starts from scratch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Fast DDS intermittently drops Gazebo Empty-service replies in this setup.
# Select the installed Cyclone DDS backend before any ROS module is imported.
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")

from limo_delivery_rl_v2.delivery_env import LimoWaypointRLEnv  # noqa: E402

DEFAULT_HYPERPARAMETERS: dict[str, object] = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "clip_range": 0.15,
    "ent_coef": 0.003,
}


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the training script."""
    parser = argparse.ArgumentParser(description="Train the PPO waypoint policy.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("runs/limo_delivery_rl_v2/checkpoints")
    )
    parser.add_argument("--save-path", type=Path, default=Path("runs/limo_delivery_rl_v2/final_model"))
    parser.add_argument(
        "--tensorboard-log", type=Path, default=Path("runs/limo_delivery_rl_v2/tensorboard")
    )
    parser.add_argument("--tb-log-name", default="ppo_waypoint")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_HYPERPARAMETERS["n_steps"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HYPERPARAMETERS["batch_size"])
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--no-ros", action="store_true", help="Train against the offline backend.")
    parser.add_argument(
        "--check-env", action="store_true", help="Run the SB3 env checker before training."
    )
    parser.add_argument("--progress-every-episodes", type=int, default=1)
    return parser


def main() -> None:
    """Train a PPO policy and write checkpoints plus TensorBoard logs."""
    args = build_parser().parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    from limo_delivery_rl_v2.tb_callback import EpisodeMetricCallback

    if args.check_env:
        from stable_baselines3.common.env_checker import check_env

        checker_env = LimoWaypointRLEnv(enable_ros=not args.no_ros)
        try:
            check_env(checker_env, warn=True)
        finally:
            checker_env.close()

    env = LimoWaypointRLEnv(enable_ros=not args.no_ros)
    callback = CallbackList(
        [
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(args.checkpoint_dir),
                name_prefix="ppo_waypoint",
            ),
            EpisodeMetricCallback(print_every_episodes=args.progress_every_episodes),
        ]
    )
    hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
    hyperparameters["n_steps"] = args.n_steps
    hyperparameters["batch_size"] = args.batch_size
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(args.tensorboard_log),
        policy_kwargs={"log_std_init": args.log_std_init},
        verbose=1,
        **hyperparameters,
    )
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name=args.tb_log_name)
        model.save(args.save_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
