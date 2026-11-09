"""
navigation.py
Simple waypoint-based navigation. Given a route name (a list of
waypoint keys from environment.WAYPOINTS), drives the robot base
smoothly from waypoint to waypoint: rotate to face the next point,
then move forward until close enough, then advance to the next one.

Movement is kinematic (like the manual-control demo this project is
based on) rather than wheel-torque physics, which keeps navigation
reliable for a live demo.
"""

import math
import pybullet as p

from house_data import WAYPOINTS, build_route_to, SCALE

MOVE_SPEED = 2.2 * SCALE       # m/s
TURN_SPEED = 2.5               # rad/s (rotation speed doesn't need to scale with size)
ARRIVE_RADIUS = 0.35 * SCALE   # meters
ANGLE_TOLERANCE = 0.08         # radians

_BOUND_X = 7.5 * SCALE
_BOUND_Y = 8.4 * SCALE


class Navigator:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.route = []
        self.route_index = 0
        self.active = False

    def start_route_to(self, destination_key):
        """Routes to ANY destination key from house_data.WAYPOINTS
        (e.g. 'bed', 'hall_sofa', 'living_room'), through the correct
        door automatically."""
        route_names = build_route_to(destination_key)
        self.route = [WAYPOINTS[name] for name in route_names]
        self.route_index = 0
        self.active = True

    def start_route_points(self, points):
        """points: list of (x, y) tuples"""
        self.route = list(points)
        self.route_index = 0
        self.active = True

    def is_done(self):
        return not self.active

    def step(self, dt=1.0 / 60.0):
        """Advances the robot one simulation step toward the current
        waypoint. Returns True while still navigating, False once the
        whole route is complete."""
        if not self.active:
            return False

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y, z = pos
        _, _, yaw = p.getEulerFromQuaternion(orn)

        target_x, target_y = self.route[self.route_index]
        dx, dy = target_x - x, target_y - y
        dist = math.hypot(dx, dy)

        if dist < ARRIVE_RADIUS:
            self.route_index += 1
            if self.route_index >= len(self.route):
                self.active = False
                return False
            return True

        desired_yaw = math.atan2(dy, dx)
        angle_diff = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > ANGLE_TOLERANCE:
            turn = TURN_SPEED if angle_diff > 0 else -TURN_SPEED
            forward = 0.0
        else:
            turn = angle_diff * 2.0  # gentle correction while moving
            forward = MOVE_SPEED

        new_yaw = yaw + turn * dt
        step_dist = forward * dt
        new_x = x + math.cos(new_yaw) * step_dist
        new_y = y + math.sin(new_yaw) * step_dist

        new_x = max(-_BOUND_X, min(_BOUND_X, new_x))
        new_y = max(-_BOUND_Y, min(_BOUND_Y, new_y))

        p.resetBasePositionAndOrientation(
            self.robot_id, [new_x, new_y, z], p.getQuaternionFromEuler([0, 0, new_yaw])
        )
        return True
