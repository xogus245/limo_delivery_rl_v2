"""Static map sanity checks used before an environment is allowed to start."""

from __future__ import annotations

from pathlib import Path

from limo_delivery_rl_v2.state import MapConfig


def validate_existing_map(config: MapConfig) -> None:
    """Raise ``FileNotFoundError`` unless both map files referenced by ``config`` exist.

    Training obstacles are never baked into these files; they exist only so the
    global planner and the environment agree on the same static world.
    """
    if not Path(config.yaml_path).is_file():
        raise FileNotFoundError(config.yaml_path)
    if not Path(config.image_path).is_file():
        raise FileNotFoundError(config.image_path)
