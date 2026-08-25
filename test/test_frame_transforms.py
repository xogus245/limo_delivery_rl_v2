"""Numeric verification of the ``map`` -> ``base_link`` transform contract."""

import math

import pytest

from limo_delivery_rl_v2.geometry import (
    Pose2D,
    base_link_to_map,
    heading_error_to,
    map_to_base_link,
    wrap_angle,
    yaw_from_quaternion,
    yaw_to_quaternion_z_w,
)


def test_identity_pose_leaves_map_coordinates_unchanged():
    pose = Pose2D(0.0, 0.0, 0.0)

    assert map_to_base_link((2.0, -1.0), pose) == pytest.approx((2.0, -1.0))


def test_pure_translation_subtracts_the_robot_position():
    pose = Pose2D(3.0, 1.0, 0.0)

    assert map_to_base_link((5.0, 4.0), pose) == pytest.approx((2.0, 3.0))


def test_ninety_degree_heading_rotates_into_the_robot_frame():
    # Robot at (1, 1) facing +y: a point 2 m further along +y is 2 m ahead,
    # and a point 2 m along +x is 2 m to the robot's right.
    pose = Pose2D(1.0, 1.0, math.pi / 2.0)

    assert map_to_base_link((1.0, 3.0), pose) == pytest.approx((2.0, 0.0), abs=1e-9)
    assert map_to_base_link((3.0, 1.0), pose) == pytest.approx((0.0, -2.0), abs=1e-9)


def test_known_pose_matches_hand_computed_values():
    pose = Pose2D(2.0, -1.0, math.radians(30.0))
    point = (4.0, 1.0)

    x, y = map_to_base_link(point, pose)

    dx, dy = 2.0, 2.0
    assert x == pytest.approx(dx * math.cos(math.radians(30.0)) + dy * math.sin(math.radians(30.0)))
    assert y == pytest.approx(-dx * math.sin(math.radians(30.0)) + dy * math.cos(math.radians(30.0)))
    assert (x, y) == pytest.approx((2.732050808, 0.732050808), abs=1e-6)


@pytest.mark.parametrize("yaw", [-3.0, -1.2, 0.0, 0.4, 2.9])
def test_transform_round_trips_through_the_inverse(yaw):
    pose = Pose2D(-4.5, 7.25, yaw)
    point = (1.5, -2.5)

    assert base_link_to_map(map_to_base_link(point, pose), pose) == pytest.approx(point)


def test_heading_error_is_signed_and_wrapped():
    pose = Pose2D(0.0, 0.0, math.pi - 0.1)

    assert heading_error_to((1.0, 0.0), pose) == pytest.approx(-(math.pi - 0.1))
    assert heading_error_to((-1.0, 0.05), pose) < 0.2
    assert abs(heading_error_to((-1.0, 0.05), pose)) <= math.pi


@pytest.mark.parametrize("yaw", [-3.0, -0.7, 0.0, 1.1, 3.0])
def test_yaw_quaternion_round_trip(yaw):
    z, w = yaw_to_quaternion_z_w(yaw)

    assert yaw_from_quaternion(0.0, 0.0, z, w) == pytest.approx(yaw)


def test_wrap_angle_maps_into_a_single_revolution():
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-3.0 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
