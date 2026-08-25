"""Normalized action -> physical command scaling and acceleration limiting.

The policy owns both speed and steering; there is no residual term added to a
Nav2 controller output and no distance-based automatic slowdown.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from limo_delivery_rl_v2.state import ControlConfig

ACTION_DIM = 2


def sanitize_action(action) -> tuple[NDArray[np.float32], bool]:
    """Return a finite ``(2,)`` action clipped to ``[-1, 1]`` and a validity flag.

    ``NaN``/``Inf``/wrong-sized actions are reported invalid so the safety gate
    can force a full stop instead of guessing a command.
    """
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    if values.size != ACTION_DIM or not np.all(np.isfinite(values)):
        return zero, False
    return np.clip(values, -1.0, 1.0).astype(np.float32), True


class ActionScaler:
    """Maps ``[-1, 1]^2`` onto ``v in [0, 0.42]`` and ``omega in [-0.9, 0.9]``."""

    def __init__(self, config: ControlConfig) -> None:
        """Bind the scaler to the control limits."""
        self._config = config

    def scale(self, action: NDArray[np.float32]) -> tuple[float, float]:
        """Return the ``(v_target, omega_target)`` implied by a normalized action."""
        linear = (float(action[0]) + 1.0) * 0.5 * self._config.max_train_linear_speed
        angular = float(action[1]) * self._config.max_angular_speed
        return linear, angular


class RateLimiter:
    """Applies per-step acceleration bounds to linear and angular commands."""

    def __init__(self, config: ControlConfig) -> None:
        """Bind the limiter to the acceleration limits and control period."""
        self._config = config

    @property
    def max_linear_step(self) -> float:
        """Largest allowed ``|delta v|`` in one control step (accelerating)."""
        return self._config.max_linear_accel * self._config.control_dt

    @property
    def max_angular_step(self) -> float:
        """Largest allowed ``|delta omega|`` in one control step."""
        return self._config.max_angular_accel * self._config.control_dt

    def limit(
        self,
        linear_target: float,
        angular_target: float,
        previous_linear: float,
        previous_angular: float,
    ) -> tuple[float, float]:
        """Clamp the command change since the previous step."""
        decel_step = self._config.max_linear_decel * self._config.control_dt
        linear_step = decel_step if linear_target < previous_linear else self.max_linear_step
        linear = previous_linear + float(
            np.clip(linear_target - previous_linear, -linear_step, linear_step)
        )
        angular = previous_angular + float(
            np.clip(
                angular_target - previous_angular,
                -self.max_angular_step,
                self.max_angular_step,
            )
        )
        return linear, angular
