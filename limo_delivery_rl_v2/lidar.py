"""Angle-aware down-sampling of the 720-beam LaserScan into 24 policy bins.

The array index of the forward direction is never assumed.  ``angle_min`` and
``angle_increment`` from the ``LaserScan`` message are preserved and every bin's
angular span is derived from them, so a change in sensor mounting or field of
view cannot silently rotate the observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Fraction of a sector absorbed when an angle sits on a bin edge.
EDGE_TOLERANCE: float = 1e-9


@dataclass(frozen=True, slots=True)
class LidarGeometry:
    """Angular layout of a ``sensor_msgs/LaserScan``."""

    angle_min: float
    angle_increment: float
    beam_count: int

    @property
    def angle_max(self) -> float:
        """Angle of the last beam."""
        return self.angle_min + self.angle_increment * (self.beam_count - 1)

    @property
    def span(self) -> float:
        """Total angular field of view covered by the beams."""
        return self.angle_increment * (self.beam_count - 1)


def geometry_from_scan(msg) -> LidarGeometry:
    """Build a :class:`LidarGeometry` from a ``sensor_msgs/LaserScan`` message."""
    return LidarGeometry(
        angle_min=float(msg.angle_min),
        angle_increment=float(msg.angle_increment),
        beam_count=len(msg.ranges),
    )


def bin_edges(geometry: LidarGeometry, bins: int) -> NDArray[np.float64]:
    """Return the ``bins + 1`` angular edges that split the field of view evenly."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    return np.linspace(geometry.angle_min, geometry.angle_max, bins + 1, dtype=np.float64)


def bin_index_for_angle(angle: float, geometry: LidarGeometry, bins: int) -> int:
    """Return the bin owning ``angle``.

    Edges are half-open (``[left, right)``) except for the final bin, which
    includes its right edge so that the last beam is never dropped.  A relative
    tolerance absorbs the rounding of a reconstructed edge angle, which would
    otherwise fall one bin short (``2.999999999999999`` floors to ``2``).
    """
    sector = geometry.span / bins
    if sector <= 0.0:
        raise ValueError("LiDAR geometry has a non-positive angular span")
    raw = int(np.floor((angle - geometry.angle_min) / sector + EDGE_TOLERANCE))
    return int(min(max(raw, 0), bins - 1))


def beam_bin_indices(geometry: LidarGeometry, bins: int) -> NDArray[np.int64]:
    """Return the bin index of every beam, shape ``(beam_count,)``."""
    sector = geometry.span / bins
    if sector <= 0.0:
        raise ValueError("LiDAR geometry has a non-positive angular span")
    angles = geometry.angle_min + geometry.angle_increment * np.arange(
        geometry.beam_count, dtype=np.float64
    )
    indices = np.floor(
        (angles - geometry.angle_min) / sector + EDGE_TOLERANCE
    ).astype(np.int64)
    return np.clip(indices, 0, bins - 1)


def sanitize_ranges(
    ranges: NDArray[np.float64], *, max_range: float
) -> NDArray[np.float64]:
    """Replace ``NaN``/``Inf``/negative returns with ``max_range`` and clip to ``[0, max_range]``.

    A no-return beam means "nothing within range", which is the same information
    as a maximum-range hit, so it must not be read as an obstacle at 0 m.
    """
    sanitized = np.asarray(ranges, dtype=np.float64).copy()
    invalid = ~np.isfinite(sanitized) | (sanitized < 0.0)
    sanitized[invalid] = max_range
    return np.clip(sanitized, 0.0, max_range)


class LidarBinner:
    """Down-samples raw beams to per-bin minima, caching the beam->bin mapping."""

    def __init__(self, bins: int, max_range: float) -> None:
        """Store the bin count and the clipping/normalisation reference range."""
        if bins <= 0:
            raise ValueError("bins must be positive")
        self._bins = int(bins)
        self._max_range = float(max_range)
        self._geometry: LidarGeometry | None = None
        self._assignment: NDArray[np.int64] | None = None

    def bin_metres(
        self, ranges: NDArray[np.float64], geometry: LidarGeometry
    ) -> NDArray[np.float32]:
        """Return the minimum sanitized range of each angular bin, in metres.

        Bins covered by no beam fall back to ``max_range``.
        """
        if geometry.beam_count != len(ranges):
            geometry = LidarGeometry(
                geometry.angle_min, geometry.angle_increment, len(ranges)
            )
        if self._geometry != geometry or self._assignment is None:
            self._assignment = beam_bin_indices(geometry, self._bins)
            self._geometry = geometry
        sanitized = sanitize_ranges(ranges, max_range=self._max_range)
        binned = np.full(self._bins, self._max_range, dtype=np.float64)
        np.minimum.at(binned, self._assignment, sanitized)
        return binned.astype(np.float32)
