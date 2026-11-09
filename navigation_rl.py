"""
navigation_rl.py
Connects the trained navigation policy (navigation_policy.npz, from
train_navigation.py) to the real PyBullet simulation.

IMPORTANT DESIGN NOTE FOR HARDWARE:
This does NOT let the neural network directly command raw motor
outputs. Instead:
  1. The trained policy looks at real lidar-style raycasts + the
     relative goal and picks ONE of 8 compass directions (exactly the
     format it was trained on).
  2. A separate, simple, bounded safety layer then executes that
     direction using the same kind of small, checked motion steps as
     the original scripted navigator — including an immediate raycast
     safety check right before each move, so if something is close
     enough to be dangerous, the robot stops and re-queries the
     policy instead of blindly finishing the step.

This hierarchical split (learned high-level decision + a small,
inspectable, bounded low-level executor) is the responsible way to
bring an RL policy near real hardware. Letting a freshly trained
network directly drive motor PWM/torque with no safety layer is not
recommended, especially not before extensive additional validation on
the real robot.

If navigation_policy.npz doesn't exist yet (you haven't trained it),
RLNavigator raises a clear error — main.py catches this and falls back
to the original scripted Navigator automatically, so the demo never
breaks even if training hasn't been run.
"""

import math
import os

import numpy as np
import pybullet as p

from house_data import SCALE, WAYPOINTS
from house_nav_env import N_LIDAR_BEAMS, LIDAR_MAX_RANGE, ACTIONS, STEP_SIZE, GOAL_RADIUS
from tiny_nn import QNetwork

MODEL_PATH = os.path.join(os.path.dirname(__file__), "navigation_policy.npz")

MOVE_SPEED = 2.2 * SCALE
TURN_SPEED = 2.8                       # rad/s — bounded turning, matches the scripted navigator's feel
ANGLE_TOLERANCE = 0.12                 # radians — how "aligned" is close enough to start moving forward
SAFETY_STOP_DISTANCE = 0.35 * SCALE   # if a raycast is closer than this, don't move that way
# Re-plan often enough that distance traveled per decision roughly matches
# what the policy was trained on (STEP_SIZE meters per action), but not so
# often that near-tied Q-values cause rapid flip-flopping between similar
# directions — combined with the "stickiness" margin below, this is what
# fixes the earlier "spins/moves in circles" behavior.
REPLAN_EVERY_SECONDS = STEP_SIZE / MOVE_SPEED
STICKY_MARGIN = 1.5   # only switch away from the current heading if a clear alternative is meaningfully better
ARRIVE_RADIUS = GOAL_RADIUS            # match training's goal radius exactly, not a rescaled value
MAX_EPISODE_SECONDS = 45.0            # safety timeout so the robot can't wander forever


def _load_policy():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Run `python train_navigation.py` first "
            f"to train the navigation policy."
        )
    obs_dim = 3 + N_LIDAR_BEAMS
    net = QNetwork(obs_dim, 64, len(ACTIONS))
    net.load(MODEL_PATH)
    return net


def _real_lidar(robot_pos_xy, height=0.3):
    """Casts N_LIDAR_BEAMS rays outward from the robot's real position in
    the actual PyBullet scene (walls, furniture, doors, other objects —
    whatever's really there), matching the training env's beam layout."""
    x, y = robot_pos_xy
    from_points, to_points = [], []
    for i in range(N_LIDAR_BEAMS):
        angle = 2 * math.pi * i / N_LIDAR_BEAMS
        dx, dy = math.cos(angle), math.sin(angle)
        from_points.append([x, y, height])
        to_points.append([x + dx * LIDAR_MAX_RANGE, y + dy * LIDAR_MAX_RANGE, height])

    results = p.rayTestBatch(from_points, to_points)
    readings = np.full(N_LIDAR_BEAMS, LIDAR_MAX_RANGE, dtype=float)
    for i, res in enumerate(results):
        hit_fraction = res[2]  # 0..1 along the ray
        if hit_fraction < 1.0:
            readings[i] = hit_fraction * LIDAR_MAX_RANGE
    return readings / LIDAR_MAX_RANGE


def _observe(robot_pos_xy, goal_xy):
    rel = np.array(goal_xy) - np.array(robot_pos_xy)
    dist = np.linalg.norm(rel)
    goal_dir = rel / dist if dist > 1e-6 else np.array([0.0, 0.0])
    dist_norm = min(dist / (16.0 * SCALE), 1.0)
    lidar = _real_lidar(robot_pos_xy)
    return np.concatenate([goal_dir, [dist_norm], lidar]).astype(np.float32), dist


class RLNavigator:
    """Drop-in replacement for navigation.Navigator, but driven by the
    trained RL policy instead of a scripted waypoint route. Same
    start_route_to(...)/step(dt) interface so main.py barely has to
    change to use it."""

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.policy = _load_policy()
        self.goal_xy = None
        self.active = False
        self._time_since_replan = 0.0
        self._current_action_idx = None
        self._elapsed = 0.0

    def start_route_to(self, destination_key):
        self.goal_xy = WAYPOINTS[destination_key]
        self.active = True
        self._time_since_replan = 999.0  # force an immediate decision
        self._current_action_idx = None
        self._elapsed = 0.0

    def is_done(self):
        return not self.active

    def step(self, dt=1.0 / 60.0):
        if not self.active:
            return False

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y, z = pos
        _, _, current_yaw = p.getEulerFromQuaternion(orn)
        obs, dist = _observe((x, y), self.goal_xy)

        if dist < ARRIVE_RADIUS:
            self.active = False
            return False

        self._elapsed += dt
        if self._elapsed > MAX_EPISODE_SECONDS:
            print("[nav_rl] Timed out reaching the goal — stopping for safety.")
            self.active = False
            return False

        self._time_since_replan += dt
        if self._time_since_replan >= REPLAN_EVERY_SECONDS:
            self._current_action_idx = self._pick_direction(x, y, obs)
            self._time_since_replan = 0.0

        target_dx, target_dy = ACTIONS[self._current_action_idx]
        desired_yaw = math.atan2(target_dy, target_dx)
        angle_diff = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi

        # Smooth, bounded turn-then-move — same style as the scripted
        # navigator, and the fix for the "spins/circles" look: the robot's
        # heading now changes gradually and physically, instead of
        # snapping instantly to face a new direction every ~0.1s.
        if abs(angle_diff) > ANGLE_TOLERANCE:
            turn = TURN_SPEED if angle_diff > 0 else -TURN_SPEED
            forward = 0.0
        else:
            turn = angle_diff * 2.0  # gentle correction while moving
            forward = MOVE_SPEED

        new_yaw = current_yaw + turn * dt
        step_dist = forward * dt
        new_x = x + math.cos(new_yaw) * step_dist
        new_y = y + math.sin(new_yaw) * step_dist

        p.resetBasePositionAndOrientation(
            self.robot_id, [new_x, new_y, z], p.getQuaternionFromEuler([0, 0, new_yaw])
        )
        return True

    def _pick_direction(self, x, y, obs):
        """Ranks all 8 directions by the policy's learned Q-value, picks
        the best one that's actually clear right now — but sticks with
        the CURRENT direction unless a clear alternative is meaningfully
        better (STICKY_MARGIN) or the current one just became blocked.
        Without this stickiness, near-tied Q-values between adjacent
        directions cause rapid switching every replan tick, which is what
        was making the robot visibly spin/circle instead of committing to
        a heading. Returns an action index (not a raw vector), since we
        need it for the stickiness comparison next tick too."""
        q = self.policy.q_values(obs)

        def is_clear(action_idx):
            dx, dy = ACTIONS[action_idx]
            ahead = p.rayTest([x, y, 0.3], [x + dx * SAFETY_STOP_DISTANCE, y + dy * SAFETY_STOP_DISTANCE, 0.3])
            return ahead[0][2] >= 1.0

        ranked = np.argsort(-q)
        best_clear = next((a for a in ranked if is_clear(a)), None)

        if best_clear is None:
            # Every direction blocked (rare, tight corner) — hold the
            # current heading rather than force through something; the
            # next tick will re-check.
            return self._current_action_idx if self._current_action_idx is not None else int(ranked[0])

        current = self._current_action_idx
        if current is not None and is_clear(current) and q[best_clear] < q[current] + STICKY_MARGIN:
            return current  # not worth switching for a marginal gain

        return int(best_clear)
