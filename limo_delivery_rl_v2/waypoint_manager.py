"""Ordered waypoint progression implemented inside the RL environment.

``FollowWaypoints`` is deliberately not used: that Nav2 action drives the
controller server, which would publish a competing ``/cmd_vel``.
"""

from __future__ import annotations

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

    def reset(self) -> None:
        """Return the manager to the first waypoint."""
        self._index = 0
        self._hold_steps = 0
        self._reached_count = 0
        self._rewarded.clear()
        self._completed = False

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

    def update(self, pose: Pose2D) -> WaypointUpdate:
        """Advance the state machine.

        A waypoint counts as reached only after the robot has stayed inside the
        radius for ``waypoint_hold_steps`` consecutive steps, and its bonus is
        granted exactly once.
        """
        distance, heading_error = self.measure(pose)
        if self._completed:
            return WaypointUpdate(distance, heading_error, False, False, True, False)

        if distance <= self._config.waypoint_radius:
            self._hold_steps += 1
        else:
            self._hold_steps = 0

        if self._hold_steps < self._config.waypoint_hold_steps:
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
