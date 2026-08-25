"""Global-path progress tracking and ``base_link`` lookahead sampling.

The progress index is monotonic by construction: the nearest-point search only
ever scans forward from the last accepted index, so a policy that loops back on
itself cannot farm progress reward twice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from limo_delivery_rl_v2.geometry import Pose2D, map_to_base_link
from limo_delivery_rl_v2.state import ObservationConfig


def path_length(points: Sequence[tuple[float, float]]) -> float:
    """Total arc length of a polyline."""
    return float(
        sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    )


class PathTracker:
    """Tracks where the robot is along the Nav2 global path (``map`` frame)."""

    def __init__(self, config: ObservationConfig) -> None:
        """Configure lookahead sampling from the observation config."""
        self._config = config
        self._points: tuple[tuple[float, float], ...] = ()
        self._cumulative: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self._progress_index = 0

    def set_path(self, points: Sequence[tuple[float, float]]) -> None:
        """Replace the tracked path and reset the progress index."""
        self._points = tuple((float(x), float(y)) for x, y in points)
        self._progress_index = 0
        if len(self._points) < 2:
            self._cumulative = np.zeros(len(self._points), dtype=np.float64)
            return
        deltas = np.hypot(
            np.diff([p[0] for p in self._points]),
            np.diff([p[1] for p in self._points]),
        )
        self._cumulative = np.concatenate(([0.0], np.cumsum(deltas)))

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        """The tracked path in the ``map`` frame."""
        return self._points

    @property
    def is_available(self) -> bool:
        """Whether a usable path (>= 2 points) is loaded."""
        return len(self._points) >= 2

    @property
    def progress_index(self) -> int:
        """Monotonically non-decreasing index of the closest path point."""
        return self._progress_index

    def update(self, pose: Pose2D) -> int:
        """Advance the progress index to the nearest forward path point.

        Only the window of points within ``nearest_search_window`` metres ahead of
        the current index is searched, which keeps the cost constant on long paths
        and makes backwards jumps impossible.
        """
        if not self.is_available:
            return self._progress_index
        start = self._progress_index
        limit = self._cumulative[start] + self._config.nearest_search_window
        end = int(np.searchsorted(self._cumulative, limit, side="right"))
        end = max(end, start + 1)
        window = np.asarray(self._points[start:end], dtype=np.float64)
        distances = np.hypot(window[:, 0] - pose.x, window[:, 1] - pose.y)
        self._progress_index = start + int(np.argmin(distances))
        return self._progress_index

    def cross_track_error(self, pose: Pose2D) -> float:
        """Signed lateral offset from the path segment at the progress index.

        Positive means the robot is to the left of the direction of travel.
        """
        if not self.is_available:
            return 0.0
        end = min(max(self._progress_index, 1), len(self._points) - 1)
        ax, ay = self._points[end - 1]
        bx, by = self._points[end]
        vx, vy = bx - ax, by - ay
        length = math.hypot(vx, vy)
        if length <= 1e-9:
            return 0.0
        return float((vx * (pose.y - ay) - vy * (pose.x - ax)) / length)

    def lookahead_points(self, pose: Pose2D) -> tuple[tuple[float, float], ...]:
        """Sample the path ahead of the robot at fixed arc-length intervals.

        Samples beyond the end of the path clamp to the final point so the
        observation stays a fixed size as the goal is approached.
        """
        count = self._config.lookahead_points
        if not self.is_available:
            return tuple((pose.x, pose.y) for _ in range(count))
        base = self._cumulative[min(self._progress_index, len(self._cumulative) - 1)]
        return tuple(
            self._point_at_arc_length(base + self._config.lookahead_spacing * (k + 1))
            for k in range(count)
        )

    def relative_lookahead(self, pose: Pose2D) -> NDArray[np.float32]:
        """Lookahead points as normalized ``base_link`` coordinates, shape ``(n, 2)``."""
        limit = self._config.max_relative_position
        local = [map_to_base_link(point, pose) for point in self.lookahead_points(pose)]
        return (
            np.clip(np.asarray(local, dtype=np.float32), -limit, limit) / np.float32(limit)
        ).astype(np.float32)

    def _point_at_arc_length(self, target: float) -> tuple[float, float]:
        """Linearly interpolate the path point at arc length ``target``."""
        if target >= self._cumulative[-1]:
            return self._points[-1]
        index = int(np.searchsorted(self._cumulative, target, side="right"))
        index = min(max(index, 1), len(self._points) - 1)
        span = self._cumulative[index] - self._cumulative[index - 1]
        if span <= 1e-9:
            return self._points[index]
        ratio = (target - self._cumulative[index - 1]) / span
        ax, ay = self._points[index - 1]
        bx, by = self._points[index]
        return (ax + (bx - ax) * ratio, ay + (by - ay) * ratio)
