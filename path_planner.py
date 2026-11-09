"""
path_planner.py
Classical A* path planning over the house layout, used as the primary
navigator for live demos. This is deliberately NOT the RL policy: A*
guarantees a short, direct, obstacle-free path every single time,
computed in milliseconds, with zero randomness — exactly what a live
demo in front of judges needs. Real deployed service robots overwhelmingly
use classical planners like this (or D*, RRT) for navigation; RL tends to
get used for problems classical planning handles poorly (dynamic
obstacles, contact-rich manipulation), not as a replacement for it. The
navigation_policy.npz DQN trained earlier in this project is still a
real, documented result — it's just not what drives the robot during
the actual demo, since a policy trained in minutes on a CPU is never
going to plan as directly as an exact search algorithm.

Two-step process:
  1. Build an occupancy grid from the same wall/furniture rectangles
     used in house_nav_env.py, inflated by the robot's real base radius
     so a path is only considered valid if the whole robot (not just its
     center point) actually fits.
  2. Run A* (8-connected, octile heuristic) to find the shortest path,
     then simplify it with line-of-sight "string pulling" so the robot
     drives long straight segments instead of a jagged grid-aligned
     staircase.
"""

import heapq
import math

from house_nav_env import build_static_layout
from house_data import SCALE

GRID_CELL = 0.25                 # meters per grid cell
ROBOT_CLEARANCE = 0.5 * SCALE * 0.6  # inflate obstacles by roughly the robot's real footprint + margin
BOUND_X = 7.6 * SCALE
BOUND_Y = 8.4 * SCALE

_NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


class HousePlanner:
    def __init__(self, scale=SCALE):
        self.scale = scale
        self.rects = build_static_layout(scale)
        self.cell = GRID_CELL

    def _blocked(self, x, y):
        if abs(x) > BOUND_X or abs(y) > BOUND_Y:
            return True
        return any(r.contains_point(x, y, ROBOT_CLEARANCE) for r in self.rects)

    def _to_grid(self, x, y):
        return (round(x / self.cell), round(y / self.cell))

    def _to_world(self, gx, gy):
        return (gx * self.cell, gy * self.cell)

    def plan(self, start_xy, goal_xy):
        """Returns a list of (x, y) waypoints from start to goal, or an
        empty list if no path exists (shouldn't happen on this house
        layout — every destination was verified reachable earlier)."""
        start = self._to_grid(*start_xy)
        goal = self._to_grid(*goal_xy)

        if self._blocked(*self._to_world(*goal)):
            # Goal itself is inside the inflation margin (can happen for
            # close docking points near furniture) — nudge search to
            # accept the nearest free cell to the true goal instead.
            goal = self._nearest_free_cell(goal)

        open_set = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return self._reconstruct(came_from, current, goal_xy)

            cx, cy = current
            for dgx, dgy, cost in _NEIGHBORS:
                neighbor = (cx + dgx, cy + dgy)
                if neighbor in visited:
                    continue
                wx, wy = self._to_world(*neighbor)
                if self._blocked(wx, wy):
                    continue
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h = math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

        return []  # no path found (shouldn't happen — see docstring)

    def _nearest_free_cell(self, goal_grid, max_radius=8):
        gx, gy = goal_grid
        for r in range(max_radius):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    cand = (gx + dx, gy + dy)
                    wx, wy = self._to_world(*cand)
                    if not self._blocked(wx, wy):
                        return cand
        return goal_grid  # give up, return original (plan() will just fail to reach it)

    def _reconstruct(self, came_from, current, true_goal_xy):
        path_cells = [current]
        while current in came_from:
            current = came_from[current]
            path_cells.append(current)
        path_cells.reverse()
        path_world = [self._to_world(*c) for c in path_cells]
        path_world[-1] = true_goal_xy  # snap the final point exactly to the real goal
        return self._simplify(path_world)

    def _simplify(self, path):
        """String-pulling: greedily skip ahead to the furthest waypoint
        with a clear line of sight, so the robot drives long straight
        segments instead of following every single grid cell."""
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._line_clear(path[i], path[j]):
                j -= 1
            simplified.append(path[j])
            i = j
        return simplified

    def _line_clear(self, a, b, samples=12):
        for t in range(1, samples):
            frac = t / samples
            x = a[0] + (b[0] - a[0]) * frac
            y = a[1] + (b[1] - a[1]) * frac
            if self._blocked(x, y):
                return False
        return True
