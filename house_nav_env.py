"""
house_nav_env.py
A lightweight, dependency-free (just numpy) 2D simulation of the house
floor plan, built for fast RL training. It mirrors the wall/door
layout in environment.py using simple axis-aligned rectangles instead
of full PyBullet physics, so thousands of episodes can be simulated
per second on CPU with no GPU/PyTorch required.

Key design choices, explained because they matter for hardware later:

1. GOAL-CONDITIONED, not one-policy-per-place. The goal position is
   part of the observation (as a relative vector), so a single trained
   policy generalizes to ANY point in the house — including ones it
   never specifically trained on — rather than needing a separate
   model per destination.

2. LIDAR-STYLE OBSERVATION, not raw (x, y) state. The agent sees N
   distance readings to the nearest obstacle along evenly spaced
   angles, exactly like a real ultrasonic/LIDAR ring on a physical
   robot would report. This is what makes obstacle avoidance transfer
   to hardware: swap this function for real sensor readings later and
   the trained policy's input format stays identical.

3. DOMAIN RANDOMIZATION. Every training episode adds a few random
   extra obstacles in free space (not just the fixed house walls/
   furniture). This is standard practice for sim-to-real transfer: a
   policy that has only ever seen the exact fixed layout tends to fail
   the moment something in the real world is slightly different (a
   chair moved, a bag left on the floor). Training against randomized
   obstacles teaches genuine avoidance behavior, not map memorization.

4. The action space is 8 discrete compass directions (a small step in
   that world-frame direction) rather than raw wheel torques. This
   keeps training fast and stable, and is meant to be used as a
   HIGH-LEVEL planner: the trained policy picks *which way to go next*,
   while the existing, already-tested kinematic controller in
   navigation.py executes the actual safe motion. See
   navigation_rl.py for that integration. This hierarchical split
   (learned high-level direction + a bounded, safety-checked low-level
   controller) is the responsible way to put RL near real hardware —
   letting a freshly-trained policy directly command motors with no
   safety layer is not recommended for a hardware demo.
"""

import math
import random

import numpy as np

N_LIDAR_BEAMS = 12
LIDAR_MAX_RANGE = 4.0
STEP_SIZE = 0.35
MAX_EPISODE_STEPS = 220
GOAL_RADIUS = 0.4
COLLISION_MARGIN = 0.18  # robot "radius" for collision checking

ACTIONS = [
    (math.cos(a), math.sin(a))
    for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
]  # 8 compass directions as unit vectors


class Rect:
    """Axis-aligned rectangle obstacle: center (x, y) + half extents."""
    __slots__ = ("cx", "cy", "hx", "hy")

    def __init__(self, cx, cy, hx, hy):
        self.cx, self.cy, self.hx, self.hy = cx, cy, hx, hy

    def contains_point(self, x, y, margin=0.0):
        return (self.cx - self.hx - margin <= x <= self.cx + self.hx + margin and
                self.cy - self.hy - margin <= y <= self.cy + self.hy + margin)

    def ray_hit_distance(self, ox, oy, dx, dy, max_range):
        """Returns distance from (ox,oy) along direction (dx,dy) to this
        rectangle's boundary, or None if no hit within max_range."""
        t_min, t_max = 0.0, max_range
        for axis_o, axis_d, lo, hi in (
            (ox, dx, self.cx - self.hx, self.cx + self.hx),
            (oy, dy, self.cy - self.hy, self.cy + self.hy),
        ):
            if abs(axis_d) < 1e-9:
                if axis_o < lo or axis_o > hi:
                    return None
            else:
                t1 = (lo - axis_o) / axis_d
                t2 = (hi - axis_o) / axis_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None
        if t_min < 0:
            return None
        return t_min


def build_static_layout(scale=1.5):
    """Wall + furniture rectangles matching environment.py's fixed
    layout, so the RL-trained policy sees the same house it'll be
    tested/deployed in. Doors are treated as always-open gaps (the
    real sim opens them automatically as the robot approaches, so the
    planner can assume they're passable)."""
    s = scale
    rects = []

    def wall(cx, cy, hx, hy):
        rects.append(Rect(cx, cy, hx, hy))

    # Outer walls
    wall(0, 8 * s, 8 * s, 0.1 * s)
    wall(0, -8 * s, 8 * s, 0.1 * s)
    wall(-8 * s, 0, 0.1 * s, 8 * s)
    wall(8 * s, 0, 0.1 * s, 8 * s)

    # Divider walls — mirrors environment.py exactly. The gaps at
    # x≈-1.0*s and x≈1.0*s (between these walls and the mid-upper
    # divider) are the real door openings; doors are treated as always
    # passable here since the real sim auto-opens them as the robot
    # approaches.
    wall(-4.8 * s, 0, 3.2 * s, 0.1 * s)
    wall(4.8 * s, 0, 3.2 * s, 0.1 * s)
    wall(0, 4 * s, 0.1 * s, 4 * s)

    # Furniture footprints (approximate, matches environment.py positions)
    wall(-5 * s, 5 * s, 1.0 * s, 0.5 * s)     # living sofa
    wall(-5 * s, 3.2 * s, 0.6 * s, 0.4 * s)   # living table
    wall(5.5 * s, 5 * s, 1.2 * s, 0.9 * s)    # bed
    wall(5.5 * s, 2 * s, 0.8 * s, 0.5 * s)    # bedroom table
    wall(0, -6 * s, 1.0 * s, 0.5 * s)         # hall sofa

    return rects


class HouseNavEnv:
    def __init__(self, waypoints, scale=1.5, n_random_obstacles=3, seed=None):
        self.waypoints = waypoints  # dict name -> (x, y)
        self.scale = scale
        self.n_random_obstacles = n_random_obstacles
        self.static_rects = build_static_layout(scale)
        self.bound_x = 7.6 * scale
        self.bound_y = 8.4 * scale
        self.rng = random.Random(seed)
        self.rects = list(self.static_rects)
        self.pos = np.array([0.0, 0.0])
        self.goal = np.array([0.0, 0.0])
        self.steps = 0

    def _random_free_point(self):
        for _ in range(200):
            x = self.rng.uniform(-self.bound_x + 0.5, self.bound_x - 0.5)
            y = self.rng.uniform(-self.bound_y + 0.5, self.bound_y - 0.5)
            if not any(r.contains_point(x, y, COLLISION_MARGIN) for r in self.static_rects):
                return x, y
        return 0.0, -6.5 * self.scale  # fallback: entrance

    def _spawn_random_obstacles(self):
        extra = []
        for _ in range(self.n_random_obstacles):
            for _ in range(50):
                x, y = self._random_free_point()
                hx = self.rng.uniform(0.15, 0.4) * self.scale
                hy = self.rng.uniform(0.15, 0.4) * self.scale
                cand = Rect(x, y, hx, hy)
                # Don't block the goal or the start
                if cand.contains_point(*self.pos, COLLISION_MARGIN) or cand.contains_point(*self.goal, COLLISION_MARGIN):
                    continue
                extra.append(cand)
                break
        return extra

    def reset(self, goal_key=None, start_xy=None):
        self.pos = np.array(self._random_free_point() if start_xy is None else start_xy, dtype=float)

        if goal_key is None:
            goal_key = self.rng.choice(list(self.waypoints.keys()))
        self.goal = np.array(self.waypoints[goal_key], dtype=float)

        self.rects = list(self.static_rects) + self._spawn_random_obstacles()
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        return self._observe()

    def _lidar(self):
        readings = np.full(N_LIDAR_BEAMS, LIDAR_MAX_RANGE, dtype=float)
        for i in range(N_LIDAR_BEAMS):
            angle = 2 * math.pi * i / N_LIDAR_BEAMS
            dx, dy = math.cos(angle), math.sin(angle)
            best = LIDAR_MAX_RANGE
            for r in self.rects:
                d = r.ray_hit_distance(self.pos[0], self.pos[1], dx, dy, LIDAR_MAX_RANGE)
                if d is not None and d < best:
                    best = d
            readings[i] = best
        return readings / LIDAR_MAX_RANGE  # normalized 0..1

    def _observe(self):
        rel_goal = self.goal - self.pos
        dist = np.linalg.norm(rel_goal)
        goal_dir = rel_goal / dist if dist > 1e-6 else np.array([0.0, 0.0])
        lidar = self._lidar()
        # observation: [goal_dx_norm, goal_dy_norm, dist_norm, lidar x N]
        dist_norm = min(dist / (16.0 * self.scale), 1.0)
        return np.concatenate([goal_dir, [dist_norm], lidar]).astype(np.float32)

    def _collides(self, x, y):
        if abs(x) > self.bound_x or abs(y) > self.bound_y:
            return True
        return any(r.contains_point(x, y, COLLISION_MARGIN) for r in self.rects)

    def step(self, action_idx):
        self.steps += 1
        dx, dy = ACTIONS[action_idx]
        new_x = self.pos[0] + dx * STEP_SIZE
        new_y = self.pos[1] + dy * STEP_SIZE

        collided = self._collides(new_x, new_y)
        if not collided:
            self.pos = np.array([new_x, new_y])

        dist = np.linalg.norm(self.goal - self.pos)
        reached = dist < GOAL_RADIUS

        # Reward shaping: progress toward goal, small time penalty,
        # collision penalty (episode continues but the agent learns
        # bumping into things is bad), big bonus for reaching the goal.
        reward = (self.prev_dist - dist) * 4.0
        reward -= 0.02  # time penalty, encourages efficient paths
        if collided:
            reward -= 1.0
        if reached:
            reward += 20.0
        self.prev_dist = dist

        done = reached or self.steps >= MAX_EPISODE_STEPS
        info = {"reached": reached, "collided": collided}
        return self._observe(), reward, done, info

    @property
    def obs_dim(self):
        return 3 + N_LIDAR_BEAMS

    @property
    def n_actions(self):
        return len(ACTIONS)
