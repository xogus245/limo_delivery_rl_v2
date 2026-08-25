"""Global-path progress, lookahead sampling and cross-track error."""

import math

import numpy as np
import pytest

from limo_delivery_rl_v2.geometry import Pose2D
from limo_delivery_rl_v2.path_tracker import PathTracker, path_length
from limo_delivery_rl_v2.state import ObservationConfig


def straight_path(length_m: float = 10.0, spacing: float = 0.05):
    """A dense straight path along +x, as navfn would produce."""
    count = int(length_m / spacing) + 1
    return [(index * spacing, 0.0) for index in range(count)]


def tracker(config: ObservationConfig | None = None) -> PathTracker:
    """A tracker preloaded with a straight path."""
    instance = PathTracker(config or ObservationConfig())
    instance.set_path(straight_path())
    return instance


def test_progress_index_follows_the_robot_forward():
    instance = tracker()

    assert instance.update(Pose2D(0.0, 0.0, 0.0)) == 0
    assert instance.update(Pose2D(1.0, 0.0, 0.0)) == 20
    assert instance.update(Pose2D(4.0, 0.0, 0.0)) == 80


def test_nearest_search_window_bounds_a_single_update():
    config = ObservationConfig()
    instance = tracker(config)

    # One update can advance at most `nearest_search_window` metres of arc
    # length, which is ~240 control steps of travel at the training speed.
    jumped = instance.update(Pose2D(9.9, 0.0, 0.0))

    assert jumped == pytest.approx(config.nearest_search_window / 0.05, abs=1)


def test_progress_index_never_decreases_when_the_robot_moves_backwards():
    instance = tracker()
    instance.update(Pose2D(5.0, 0.0, 0.0))

    assert instance.update(Pose2D(1.0, 0.0, 0.0)) == 100
    assert instance.update(Pose2D(0.0, 0.0, 0.0)) == 100
    assert instance.progress_index == 100


def test_progress_index_is_monotonic_over_a_noisy_trajectory():
    instance = tracker()
    rng = np.random.default_rng(7)
    previous = 0

    for step in range(200):
        x = step * 0.04
        pose = Pose2D(x + float(rng.normal(0.0, 0.05)), float(rng.normal(0.0, 0.1)), 0.0)
        current = instance.update(pose)
        assert current >= previous
        previous = current


def test_cross_track_error_sign_distinguishes_left_and_right():
    instance = tracker()

    instance.update(Pose2D(2.0, 0.5, 0.0))
    assert instance.cross_track_error(Pose2D(2.0, 0.5, 0.0)) == pytest.approx(0.5, abs=1e-9)
    assert instance.cross_track_error(Pose2D(2.0, -0.5, 0.0)) == pytest.approx(-0.5, abs=1e-9)


def test_lookahead_samples_five_points_at_forty_centimetre_spacing():
    config = ObservationConfig()
    instance = tracker(config)
    pose = Pose2D(1.0, 0.0, 0.0)
    instance.update(pose)

    points = instance.lookahead_points(pose)

    assert len(points) == config.lookahead_points
    for index, point in enumerate(points):
        assert point[0] == pytest.approx(1.0 + config.lookahead_spacing * (index + 1), abs=1e-6)
        assert point[1] == pytest.approx(0.0, abs=1e-9)


def test_lookahead_clamps_to_the_final_point_near_the_goal():
    instance = tracker()
    # Walk forward the way the control loop does; the nearest-point search only
    # scans a bounded window ahead of the current index.
    for step in range(200):
        instance.update(Pose2D(step * 0.05, 0.0, 0.0))
    pose = Pose2D(9.9, 0.0, 0.0)
    instance.update(pose)

    points = instance.lookahead_points(pose)

    assert points[-1] == pytest.approx((10.0, 0.0), abs=1e-6)
    assert all(point[0] <= 10.0 + 1e-9 for point in points)


def test_relative_lookahead_is_normalized_into_the_unit_range():
    config = ObservationConfig()
    instance = tracker(config)
    pose = Pose2D(1.0, 0.3, math.pi / 2.0)
    instance.update(pose)

    relative = instance.relative_lookahead(pose)

    assert relative.shape == (config.lookahead_points, 2)
    assert np.all(np.abs(relative) <= 1.0)
    # Facing +y while the path runs along +x: the path is to the robot's right.
    assert relative[0][1] < 0.0


def test_relative_lookahead_clips_far_points_to_three_metres():
    config = ObservationConfig()
    instance = PathTracker(config)
    instance.set_path([(0.0, 0.0), (50.0, 0.0)])
    pose = Pose2D(0.0, 0.0, 0.0)
    instance.update(pose)

    relative = instance.relative_lookahead(pose)

    assert np.all(relative <= 1.0)
    assert np.max(relative) <= 1.0


def test_empty_path_is_reported_unavailable():
    instance = PathTracker(ObservationConfig())
    instance.set_path([(1.0, 1.0)])

    assert not instance.is_available
    assert instance.cross_track_error(Pose2D(0.0, 0.0, 0.0)) == 0.0


def test_path_length_sums_segment_distances():
    assert path_length([(0.0, 0.0), (3.0, 4.0), (3.0, 8.0)]) == pytest.approx(9.0)
