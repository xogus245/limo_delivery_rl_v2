"""Observation dimensionality, layout and normalization ranges."""

import numpy as np
import pytest

from limo_delivery_rl_v2.observation import (
    LidarFrameStack,
    build_observation,
    observation_space_bounds,
)
from limo_delivery_rl_v2.state import ControlConfig, ObservationConfig, observation_dim

OBS = ObservationConfig()
CONTROL = ControlConfig()

LIDAR_SLICE = slice(0, 96)
PATH_SLICE = slice(96, 106)
WAYPOINT_SLICE = slice(106, 108)
VELOCITY_SLICE = slice(108, 110)
ACTION_SLICE = slice(110, 112)


def observation(**changes) -> np.ndarray:
    """Build an observation from neutral inputs plus overrides."""
    values = {
        "lidar_stack": np.ones(OBS.lidar_bins * OBS.lidar_frame_stack, dtype=np.float32),
        "relative_path": np.zeros((OBS.lookahead_points, 2), dtype=np.float32),
        "waypoint_distance": 0.0,
        "waypoint_heading_error": 0.0,
        "linear_velocity": 0.0,
        "angular_velocity": 0.0,
        "previous_action": np.zeros(2, dtype=np.float32),
        "observation": OBS,
        "control": CONTROL,
    }
    values.update(changes)
    return build_observation(**values)


def test_observation_is_exactly_one_hundred_and_twelve_dimensional():
    assert observation_dim(OBS) == 112
    assert observation().shape == (112,)
    assert observation().dtype == np.float32


def test_dimension_breakdown_matches_the_specification():
    assert OBS.lidar_bins * OBS.lidar_frame_stack == 96
    assert OBS.lookahead_points * 2 == 10
    assert 96 + 10 + 2 + 2 + 2 == 112


def test_space_bounds_cover_the_vector_and_have_the_right_shape():
    low, high = observation_space_bounds(OBS)

    assert low.shape == high.shape == (112,)
    assert low.dtype == high.dtype == np.float32
    assert np.all(low[LIDAR_SLICE] == 0.0)
    assert np.all(low[PATH_SLICE] == -1.0)
    assert low[WAYPOINT_SLICE].tolist() == [0.0, -1.0]
    assert np.all(high == 1.0)


def test_every_field_lands_in_its_documented_slice():
    vector = observation(
        lidar_stack=np.full(96, 0.25, dtype=np.float32),
        relative_path=np.full((5, 2), 0.5, dtype=np.float32),
        waypoint_distance=10.0,
        waypoint_heading_error=np.pi / 2.0,
        linear_velocity=0.21,
        angular_velocity=-0.45,
        previous_action=np.array([0.3, -0.7], dtype=np.float32),
    )

    assert np.all(vector[LIDAR_SLICE] == pytest.approx(0.25))
    assert np.all(vector[PATH_SLICE] == pytest.approx(0.5))
    assert vector[WAYPOINT_SLICE] == pytest.approx([0.5, 0.5])
    assert vector[VELOCITY_SLICE] == pytest.approx([0.5, -0.5])
    assert vector[ACTION_SLICE] == pytest.approx([0.3, -0.7], abs=1e-6)


def test_normalization_references_match_the_specification():
    assert OBS.lidar_max_range == pytest.approx(8.0)
    assert OBS.max_waypoint_distance == pytest.approx(20.0)
    assert OBS.max_relative_position == pytest.approx(3.0)
    assert CONTROL.max_train_linear_speed == pytest.approx(0.42)
    assert CONTROL.max_angular_speed == pytest.approx(0.9)


@pytest.mark.parametrize(
    "extreme",
    [
        {"waypoint_distance": 1e6},
        {"waypoint_distance": -5.0},
        {"waypoint_heading_error": 100.0},
        {"waypoint_heading_error": -100.0},
        {"linear_velocity": 50.0},
        {"angular_velocity": -50.0},
        {"previous_action": np.array([9.0, -9.0], dtype=np.float32)},
        {"lidar_stack": np.full(96, 7.0, dtype=np.float32)},
    ],
)
def test_out_of_range_inputs_are_clipped_into_the_space(extreme):
    low, high = observation_space_bounds(OBS)

    vector = observation(**extreme)

    assert np.all(vector >= low - 1e-6)
    assert np.all(vector <= high + 1e-6)


def test_a_wrong_sized_vector_is_rejected_rather_than_silently_reshaped():
    with pytest.raises(ValueError, match="112-dimensional"):
        observation(relative_path=np.zeros((4, 2), dtype=np.float32))


# ------------------------------------------------------------- frame stacking


def test_frame_stack_starts_uniformly_filled():
    stack = LidarFrameStack(OBS)
    stack.reset(np.full(24, 0.5, dtype=np.float32))

    assert stack.as_array().shape == (4, 24)
    assert np.all(stack.flat() == pytest.approx(0.5))


def test_frame_stack_keeps_four_frames_oldest_first():
    stack = LidarFrameStack(OBS)
    stack.reset(np.zeros(24, dtype=np.float32))

    for value in (0.1, 0.2, 0.3, 0.4, 0.5):
        stack.push(np.full(24, value, dtype=np.float32))

    rows = stack.as_array()
    assert rows.shape == (4, 24)
    assert [float(row[0]) for row in rows] == pytest.approx([0.2, 0.3, 0.4, 0.5])


def test_frame_stack_spans_two_tenths_of_a_second_at_twenty_hertz():
    assert OBS.lidar_frame_stack * CONTROL.control_dt == pytest.approx(0.20)


def test_frame_stack_clips_and_rejects_wrong_bin_counts():
    stack = LidarFrameStack(OBS)
    stack.push(np.full(24, 5.0, dtype=np.float32))

    assert np.all(stack.as_array()[-1] == 1.0)
    with pytest.raises(ValueError, match="24 LiDAR bins"):
        stack.push(np.zeros(12, dtype=np.float32))
