"""
train_navigation.py
Trains a goal-conditioned navigation policy with DQN (Deep Q-Learning)
on house_nav_env.HouseNavEnv, using only numpy (see tiny_nn.py).

One policy learns to reach EVERY destination in the house (room 1,
room 2, bed, sofa, hall sofa, etc.) because the goal is part of the
observation, not baked into a separate model per place. It also
learns obstacle avoidance from lidar-style readings, trained against
randomized obstacle layouts each episode so it generalizes beyond the
exact fixed furniture positions.

Run:
    python train_navigation.py

Produces: navigation_policy.npz (loaded by navigation_rl.py)

Training here uses modest defaults so it finishes in a few minutes on
a normal laptop CPU. Increase N_EPISODES for a stronger policy before
a real competition/hardware run — see the printed success-rate curve
to judge whether it's converged enough yet.
"""

import time
import random
from collections import deque

import numpy as np

from house_nav_env import HouseNavEnv
from house_data import WAYPOINTS, SCALE
from tiny_nn import QNetwork

N_EPISODES = 4000
HIDDEN_DIM = 64
GAMMA = 0.95
BATCH_SIZE = 64
LR = 1e-3
REPLAY_CAPACITY = 50_000
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998          # multiplied per episode
TARGET_SYNC_EVERY_STEPS = 250
MIN_REPLAY_BEFORE_TRAIN = 500
EVAL_EVERY_EPISODES = 200


def evaluate(env, q_net, n_trials_per_goal=5, seed=999):
    """Greedy (no exploration) rollout on every named destination,
    reporting per-goal and overall success rate. This is the honest
    number to look at before trusting the policy for a demo."""
    rng = random.Random(seed)
    results = {}
    for goal_key in env.waypoints:
        successes = 0
        for _ in range(n_trials_per_goal):
            obs = env.reset(goal_key=goal_key)
            done = False
            while not done:
                q = q_net.q_values(obs)
                action = int(np.argmax(q))
                obs, _, done, info = env.step(action)
            if info["reached"]:
                successes += 1
        results[goal_key] = successes / n_trials_per_goal
    overall = float(np.mean(list(results.values())))
    return overall, results


def train():
    env = HouseNavEnv(WAYPOINTS, scale=SCALE, n_random_obstacles=3, seed=0)
    q_net = QNetwork(env.obs_dim, HIDDEN_DIM, env.n_actions, seed=1)
    target_net = QNetwork(env.obs_dim, HIDDEN_DIM, env.n_actions, seed=1)
    target_net.copy_from(q_net)

    replay = deque(maxlen=REPLAY_CAPACITY)
    epsilon = EPSILON_START
    total_steps = 0
    recent_success = deque(maxlen=100)
    recent_return = deque(maxlen=100)

    start_time = time.time()
    print(f"Training on {env.obs_dim}-dim observations, {env.n_actions} actions, "
          f"{len(env.waypoints)} destinations: {list(env.waypoints.keys())}\n")

    for episode in range(1, N_EPISODES + 1):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            if random.random() < epsilon:
                action = random.randrange(env.n_actions)
            else:
                action = int(np.argmax(q_net.q_values(obs)))

            next_obs, reward, done, info = env.step(action)
            replay.append((obs, action, reward, next_obs, done))
            obs = next_obs
            ep_return += reward
            total_steps += 1

            if len(replay) >= max(BATCH_SIZE, MIN_REPLAY_BEFORE_TRAIN):
                batch = random.sample(replay, BATCH_SIZE)
                obs_b = np.array([b[0] for b in batch])
                act_b = np.array([b[1] for b in batch])
                rew_b = np.array([b[2] for b in batch])
                next_b = np.array([b[3] for b in batch])
                done_b = np.array([b[4] for b in batch], dtype=float)

                q_next, _ = target_net.forward(next_b)
                target = rew_b + GAMMA * q_next.max(axis=1) * (1 - done_b)

                q_pred, cache = q_net.forward(obs_b)
                td_error = q_pred[np.arange(BATCH_SIZE), act_b] - target
                dQ = np.zeros_like(q_pred)
                dQ[np.arange(BATCH_SIZE), act_b] = 2 * td_error / BATCH_SIZE
                grads = q_net.backward(cache, dQ)
                q_net.adam_step(grads, lr=LR)

            if total_steps % TARGET_SYNC_EVERY_STEPS == 0:
                target_net.copy_from(q_net)

        recent_success.append(1.0 if info["reached"] else 0.0)
        recent_return.append(ep_return)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            elapsed = time.time() - start_time
            print(f"ep {episode:5d}/{N_EPISODES}  "
                  f"return={np.mean(recent_return):6.2f}  "
                  f"success(last100)={np.mean(recent_success)*100:5.1f}%  "
                  f"eps={epsilon:.3f}  "
                  f"elapsed={elapsed:.0f}s")

        if episode % EVAL_EVERY_EPISODES == 0:
            overall, per_goal = evaluate(env, q_net)
            print(f"  -- greedy eval @ ep {episode}: overall success = {overall*100:.1f}%")
            for k, v in per_goal.items():
                print(f"       {k:15s} {v*100:5.1f}%")

    q_net.save("navigation_policy.npz")
    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.0f}s. Saved -> navigation_policy.npz")

    print("\nFinal evaluation (25 greedy trials per destination):")
    overall, per_goal = evaluate(env, q_net, n_trials_per_goal=25)
    print(f"Overall success rate: {overall*100:.1f}%")
    for k, v in per_goal.items():
        print(f"  {k:15s} {v*100:5.1f}%")

    return q_net


if __name__ == "__main__":
    train()
