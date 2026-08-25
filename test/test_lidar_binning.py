"""LiDAR down-sampling: angular bin boundaries and invalid-value handling."""

import math

import numpy as np
import pytest

from limo_delivery_rl_v2.lidar import (
    LidarBinner,
    LidarGeometry,
    beam_bin_indices,
    bin_edges,
    bin_index_for_angle,
    sanitize_ranges,
)
from limo_delivery_rl_v2.state import ObservationConfig

FOV = math.radians(240.0)
BEAMS = 720
BINS = 24


def simulated_geometry() -> LidarGeometry:
    """The exact geometry published by the Gazebo ray sensor in this workspace."""
    return LidarGeometry(
        angle_min=-FOV / 2.0, angle_increment=FOV / (BEAMS - 1), beam_count=BEAMS
    )


def test_simulated_scan_splits_into_equal_angular_bins():
    geometry = simulated_geometry()

    counts = np.bincount(beam_bin_indices(geometry, BINS), minlength=BINS)

    assert counts.tolist() == [30] * BINS


def test_bin_edges_span_the_full_field_of_view():
    geometry = simulated_geometry()

    edges = bin_edges(geometry, BINS)

    assert edges.size == BINS + 1
    assert edges[0] == pytest.approx(geometry.angle_min)
    assert edges[-1] == pytest.approx(geometry.angle_max)
    assert np.allclose(np.diff(edges), FOV / BINS)


def test_bin_boundaries_are_half_open_from_the_left():
    geometry = simulated_geometry()
    sector = geometry.span / BINS
    # Well below one beam spacing (0.0058 rad) but far above float noise.
    nudge = sector * 1e-6

    for index in range(BINS):
        left = geometry.angle_min + sector * index
        assert bin_index_for_angle(left, geometry, BINS) == index
        assert bin_index_for_angle(left + nudge, geometry, BINS) == index
        assert bin_index_for_angle(left + sector * 0.5, geometry, BINS) == index
        assert bin_index_for_angle(left - nudge, geometry, BINS) == max(index - 1, 0)


def test_last_beam_stays_inside_the_final_bin():
    geometry = simulated_geometry()

    assert bin_index_for_angle(geometry.angle_max, geometry, BINS) == BINS - 1
    assert beam_bin_indices(geometry, BINS)[-1] == BINS - 1


def test_forward_direction_is_derived_from_angles_not_array_index():
    geometry = simulated_geometry()
    indices = beam_bin_indices(geometry, BINS)

    # 0 rad sits exactly on the boundary between bin 11 and bin 12 for a
    # symmetric 240-degree, 720-beam scan.
    assert bin_index_for_angle(0.0, geometry, BINS) == 12
    assert indices[BEAMS // 2 - 1] == 11
    assert indices[BEAMS // 2] == 12


def test_forward_obstacle_lands_in_a_forward_bin():
    geometry = simulated_geometry()
    binner = LidarBinner(BINS, 8.0)
    ranges = np.full(BEAMS, 8.0)
    ranges[BEAMS // 2] = 1.25

    binned = binner.bin_metres(ranges, geometry)

    assert binned[12] == pytest.approx(1.25)
    assert float(binned.min()) == pytest.approx(1.25)


def test_sanitize_maps_nan_inf_and_negative_to_max_range():
    raw = np.array([np.nan, np.inf, -np.inf, -1.0, 0.0, 3.0, 99.0])

    sanitized = sanitize_ranges(raw, max_range=8.0)

    assert sanitized.tolist() == [8.0, 8.0, 8.0, 8.0, 0.0, 3.0, 8.0]


def test_binning_never_reports_a_false_zero_for_invalid_beams():
    geometry = simulated_geometry()
    binner = LidarBinner(BINS, 8.0)
    ranges = np.full(BEAMS, np.nan)

    binned = binner.bin_metres(ranges, geometry)

    assert np.all(binned == np.float32(8.0))


def test_binning_clips_to_the_eight_metre_sensor_range():
    config = ObservationConfig()
    binner = LidarBinner(config.lidar_bins, config.lidar_max_range)
    ranges = np.concatenate((np.full(BEAMS - 1, 20.0), [4.0]))

    binned = binner.bin_metres(ranges, simulated_geometry())

    # 8.0 m is the simulated sensor maximum; the observation reference range must
    # never drift away from it.
    assert config.lidar_max_range == pytest.approx(8.0)
    assert binned.max() == pytest.approx(8.0)
    assert binned[-1] == pytest.approx(4.0)


def test_binner_recomputes_assignment_when_geometry_changes():
    binner = LidarBinner(BINS, 8.0)
    wide = simulated_geometry()
    narrow = LidarGeometry(angle_min=-math.pi / 2, angle_increment=math.pi / 359, beam_count=360)

    binner.bin_metres(np.full(BEAMS, 8.0), wide)
    ranges = np.full(360, 8.0)
    ranges[0] = 0.5

    binned = binner.bin_metres(ranges, narrow)

    assert binned.shape == (BINS,)
    assert binned[0] == pytest.approx(0.5)
