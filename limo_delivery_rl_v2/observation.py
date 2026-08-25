"""112-dimensional observation assembly.

Layout (all values normalized into ``[-1, 1]``)::

    [  0: 96)  LiDAR frame stack   4 frames x 24 bins, oldest first
    [ 96:106)  path lookahead      5 points x (x, y) in base_link
    [106:108)  waypoint distance, waypoint heading error
    [108:110)  measured linear, angular velocity
    [110:112)  previous normalized action

No absolute ``map`` coordinate is ever fed to the network.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from limo_delivery_rl_v2.state import ControlConfig, ObservationConfig, observation_dim


class LidarFrameStack:
    """Rolling history of normalized LiDAR bins, oldest frame first.

    Four frames at 20 Hz span 0.20 s, which is what lets the policy infer the
    motion of dynamic obstacles from an otherwise memoryless observation.
    """

    def __init__(self, config: ObservationConfig) -> None:
        """Allocate an empty stack sized from ``config``."""
        self._bins = config.lidar_bins
        self._frames = config.lidar_frame_stack
        self._buffer: deque[NDArray[np.float32]] = deque(maxlen=self._frames)
        self.reset(np.ones(self._bins, dtype=np.float32))

    def reset(self, normalized: NDArray[np.float32]) -> None:
        """Fill every frame with ``normalized`` so the stack starts unbiased."""
        frame = self._validate(normalized)
        self._buffer.clear()
        for _ in range(self._frames):
            self._buffer.append(frame.copy())

    def push(self, normalized: NDArray[np.float32]) -> None:
        """Append the newest frame, dropping the oldest."""
        self._buffer.append(self._validate(normalized))

    def as_array(self) -> NDArray[np.float32]:
        """Return the stack as ``(frames, bins)``, oldest row first."""
        return np.asarray(self._buffer, dtype=np.float32)

    def flat(self) -> NDArray[np.float32]:
        """Return the stack flattened to ``(frames * bins,)``."""
        return self.as_array().reshape(-1)

    def _validate(self, normalized: NDArray[np.float32]) -> NDArray[np.float32]:
        """Coerce one frame to a clipped ``(bins,)`` float32 vector."""
        frame = np.asarray(normalized, dtype=np.float32).reshape(-1)
        if frame.size != self._bins:
            raise ValueError(f"expected {self._bins} LiDAR bins, got {frame.size}")
        return np.clip(frame, 0.0, 1.0).astype(np.float32)


def observation_space_bounds(
    config: ObservationConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the ``(low, high)`` arrays for the Gymnasium observation space."""
    lidar = config.lidar_bins * config.lidar_frame_stack
    path = config.lookahead_points * 2
    low = np.concatenate(
        (
            np.zeros(lidar, dtype=np.float32),
            np.full(path, -1.0, dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.full(4, -1.0, dtype=np.float32),
        )
    )
    high = np.ones(low.size, dtype=np.float32)
    return low.astype(np.float32), high


def build_observation(
    *,
    lidar_stack: NDArray[np.float32],
    relative_path: NDArray[np.float32],
    waypoint_distance: float,
    waypoint_heading_error: float,
    linear_velocity: float,
    angular_velocity: float,
    previous_action: NDArray[np.float32],
    observation: ObservationConfig,
    control: ControlConfig,
) -> NDArray[np.float32]:
    """Assemble the normalized observation vector.

    ``lidar_stack`` is expected pre-normalized to ``[0, 1]`` and
    ``relative_path`` to ``[-1, 1]``; both are re-clipped anyway so that no
    upstream mistake can emit an observation outside the declared Box.
    """
    distance = float(
        np.clip(waypoint_distance, 0.0, observation.max_waypoint_distance)
        / observation.max_waypoint_distance
    )
    heading = float(np.clip(waypoint_heading_error / np.pi, -1.0, 1.0))
    linear = float(np.clip(linear_velocity / control.max_train_linear_speed, -1.0, 1.0))
    angular = float(np.clip(angular_velocity / control.max_angular_speed, -1.0, 1.0))
    vector = np.concatenate(
        (
            np.clip(np.asarray(lidar_stack, dtype=np.float32).reshape(-1), 0.0, 1.0),
            np.clip(np.asarray(relative_path, dtype=np.float32).reshape(-1), -1.0, 1.0),
            np.array([distance, heading, linear, angular], dtype=np.float32),
            np.clip(np.asarray(previous_action, dtype=np.float32).reshape(-1), -1.0, 1.0),
        )
    ).astype(np.float32)
    expected = observation_dim(observation)
    if vector.size != expected:
        raise ValueError(f"observation must be {expected}-dimensional, got {vector.size}")
    return vector
