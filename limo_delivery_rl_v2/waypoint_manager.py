"""Ordered waypoint progression implemented inside the RL environment.

``FollowWaypoints`` is deliberately not used: that Nav2 action drives the
controller server, which would publish a competing ``/cmd_vel``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from limo_delivery_rl_v2.geometry import Pose2D, euclidean, heading_error_to
from limo_delivery_rl_v2.state import EpisodeConfig


@dataclass(frozen=True, slots=True)
class WaypointUpdate:
    """Result of advancing the waypoint state machine by one control step."""

    distance: float
    heading_error: float
    reached: bool
    switched: bool
    completed: bool
    bonus_granted: bool


class WaypointManager:
    """Tracks progress through the ordered waypoint list in the ``map`` frame."""

    def __init__(
        self,
        waypoints: Sequence[tuple[float, float, float]],
        config: EpisodeConfig,
    ) -> None:
        """Store the waypoint list and the arrival radius / hold requirement."""
        if not waypoints:
            raise ValueError("at least one waypoint is required")
        self._waypoints = tuple((float(x), float(y), float(yaw)) for x, y, yaw in waypoints)
        self._config = config
        self._index = 0
        self._hold_steps = 0
        self._reached_count = 0
        self._rewarded: set[int] = set()
        self._completed = False
        self._origin = (self._waypoints[0][0], self._waypoints[0][1])

    def reset(self, start_xy: tuple[float, float] | None = None) -> None:
        """Return the manager to the first waypoint.

        ``start_xy`` anchors the approach direction of the first waypoint, which
        the plane-crossing capture test needs.
        """
        self._index = 0
        self._hold_steps = 0
        self._reached_count = 0
        self._rewarded.clear()
        self._completed = False
        if start_xy is not None:
            self._origin = (float(start_xy[0]), float(start_xy[1]))

    @property
    def waypoints(self) -> tuple[tuple[float, float, float], ...]:
        """The ordered waypoint list."""
        return self._waypoints

    @property
    def index(self) -> int:
        """Index of the waypoint currently being pursued."""
        return self._index

    @property
    def current(self) -> tuple[float, float, float]:
        """The waypoint currently being pursued."""
        return self._waypoints[self._index]

    @property
    def reached_count(self) -> int:
        """How many waypoints have been confirmed reached this episode."""
        return self._reached_count

    @property
    def completed(self) -> bool:
        """Whether every waypoint has been reached in order."""
        return self._completed

    @property
    def hold_steps(self) -> int:
        """Consecutive steps spent inside the current waypoint radius."""
        return self._hold_steps

    def measure(self, pose: Pose2D) -> tuple[float, float]:
        """Return ``(distance, heading_error)`` from ``pose`` to the current waypoint."""
        target = (self.current[0], self.current[1])
        return euclidean((pose.x, pose.y), target), heading_error_to(target, pose)

    def approach_origin(self) -> tuple[float, float]:
        """Point the current waypoint is approached from: the previous one, or the start."""
        if self._index == 0:
            return self._origin
        previous = self._waypoints[self._index - 1]
        return (previous[0], previous[1])

    def has_driven_past(self, pose: Pose2D) -> bool:
        """Whether the robot crossed the current waypoint's plane inside the capture band.

        The radius test alone silently misses a waypoint the robot drove past
        off-centre -- it simply never enters the circle, so the episode runs on.
        This catches that case without widening the radius for a centred pass.
        """
        width = self._config.waypoint_capture_width
        if width <= 0.0:
            return False
        origin_x, origin_y = self.approach_origin()
        target = self.current
        dx, dy = target[0] - origin_x, target[1] - origin_y
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return False
        unit_x, unit_y = dx / length, dy / length
        offset_x, offset_y = pose.x - target[0], pose.y - target[1]
        along = offset_x * unit_x + offset_y * unit_y
        lateral = abs(-offset_x * unit_y + offset_y * unit_x)
        return along > 0.0 and lateral <= width

    def update(self, pose: Pose2D) -> WaypointUpdate:
        """Advance the state machine.

        A waypoint counts as reached once the robot has stayed inside the radius
        for ``waypoint_hold_steps`` consecutive steps, or -- when a capture band
        is configured -- once it has driven past the waypoint's plane. Its bonus
        is granted exactly once either way.
        """
        distance, heading_error = self.measure(pose)
        if self._completed:
            return WaypointUpdate(distance, heading_error, False, False, True, False)

        if distance <= self._config.waypoint_radius:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        held = self._hold_steps >= self._config.waypoint_hold_steps
        if not held and not self.has_driven_past(pose):
            return WaypointUpdate(distance, heading_error, False, False, False, False)

        bonus_granted = self._index not in self._rewarded
        if bonus_granted:
            self._rewarded.add(self._index)
            self._reached_count += 1

        is_last = self._index >= len(self._waypoints) - 1
        switched = False
        if is_last:
            self._completed = True
        else:
            self._index += 1
            self._hold_steps = 0
            switched = True
        distance, heading_error = self.measure(pose)
        return WaypointUpdate(
            distance, heading_error, True, switched, self._completed, bonus_granted
        )
