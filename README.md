# Voice-Controlled House Robot (PyBullet)

A simulated house — living room, bedroom, hallway, entrance — with a
mobile service robot that has a functional 3-DOF right arm, gripper,
onboard camera vision, and **trained voice command understanding**
(not keyword matching). Say "go to the bed" or "pick up the cup and
bring it to me" and the robot figures out what you mean and does it.

## What it understands

**Places** ("go to ___"): room 1 / room 2 / living room / bedroom /
bed / sofa / couch / living room sofa / entrance / hall — plus casual
phrasings like "take me to the bedroom" or "go to room2", since the
voice model is trained on hundreds of phrasing variations, not a fixed
keyword list.

**Objects** ("pick up ___ and bring it to me" / "get ___" / "fetch
___"): cup (red, living room table), glass (living room table), bottle
(blue, bedroom table), book (on the bed), plate (on the hall sofa).
Sizes are roughly 2x realistic scale so they're easy to see.

## 1. Install

```bash
pip install -r requirements.txt
```

**PyAudio note (needs the PortAudio system library first):**
- **Windows:** usually installs directly. If it fails: `pip install pipwin && pipwin install pyaudio`
- **macOS:** `brew install portaudio` then `pip install pyaudio`
- **Linux (Debian/Ubuntu):** `sudo apt-get install portaudio19-dev` then `pip install pyaudio`

No mic working yet? The app automatically falls back to a typed
`[type a command] >` prompt in the terminal — everything else runs
identically.

## 2. Train the voice model (do this once)

```bash
python train_voice_model.py
```

This generates ~365 synthetic labeled command sentences covering every
place and object, trains two scikit-learn classifiers (intent:
GOTO/FETCH/STOP, and target: which place/object), and saves them to
`voice_model.joblib`. Takes a couple of seconds. Re-run this any time
you add new places or objects in `house_data.py`.

## 3. Run

```bash
python view_house.py   # just look at the house + drive with WASD, no voice/mic needed
python train_navigation.py  # trains navigation_policy.npz (RL, ~3 min) — optional but recommended, see section 9
python main.py          # full voice-controlled demo
```

Try saying:
```
go to the bed
go to room 2
pick up the cup and bring it to me
get the glass
bring the book here
```

## 4. How it works

| File | Responsibility |
|---|---|
| `house_data.py` | Pure data: waypoints, place name aliases, object metadata. No pybullet dependency, so the voice model can train independently of the simulator. |
| `environment.py` | Builds the house geometry, furniture, sliding doors, and the 5 pickable objects, using `house_data.py`. |
| `robot.py` | The robot's URDF (mobile base, head+camera, decorative left arm, functional right arm + 2-finger gripper) and spawn/joint helpers. |
| `navigation.py` | Waypoint-follower that drives the robot to ANY named destination, routing through the correct door automatically. |
| `vision.py` | Captures a frame from the robot's head camera and runs HSV color-thresholding (OpenCV) to confirm the target object is visible before grasping. |
| `generate_training_data.py` | Builds the synthetic labeled sentence dataset used to train the voice model. |
| `train_voice_model.py` | Trains + saves the two scikit-learn classifiers to `voice_model.joblib`. |
| `voice_control.py` | Background microphone thread (`SpeechRecognition` + Google Web Speech API) + the trained `CommandParser`. Falls back to typed input if no mic. |
| `house_nav_env.py` | Fast, dependency-free 2D house simulation (lidar sensing, randomized obstacles) used to train the navigation RL policy. |
| `tiny_nn.py` | Minimal NumPy-only neural network (no PyTorch) used as the Q-network. |
| `train_navigation.py` | DQN training loop — produces `navigation_policy.npz`. |
| `navigation_rl.py` | Connects the trained policy to the real PyBullet scene, with a hierarchical safety layer. See section 9. |
| `main.py` | The state machine tying it all together: `IDLE → NAV_ONLY` (for GOTO) or `IDLE → NAV_TO_OBJECT → LOOK_AND_PICK → NAV_TO_DELIVERY → PLACE` (for FETCH). Uses the RL navigator automatically if trained, otherwise falls back to scripted navigation. |

**Pick mechanic:** the arm uses inverse kinematics
(`p.calculateInverseKinematics`) to reach toward the object, closes the
gripper fingers, then attaches the object with a fixed constraint
(`p.createConstraint`) — the standard, reliable way to simulate a firm
grasp in PyBullet without fighting contact-physics instability during
a live demo.

**Voice understanding:** two TF-IDF (character n-gram) + Logistic
Regression models — genuinely trained, not hardcoded. Held-out
accuracy on synthetic test data: 98.6% intent, 100% target. Verified
separately to generalize to phrasings never seen in training (e.g.
"take me to the bedroom").

**Delivery point:** fixed at the entrance (`house_data.DELIVERY_POINT`),
near the hall sofa — representing where you'd be sitting during the
demo.

## 5. Presenting this at the competition

- **Perception:** live camera feed + color segmentation (`vision.py`).
- **Planning:** waypoint routing through the correct door for any destination (`navigation.py`, `build_route_to`).
- **Manipulation:** IK-driven arm + gripper + constraint grasp (`robot.py`, `main.py`).
- **NLU:** trained intent + slot-filling classifiers, not keyword matching — show the held-out accuracy numbers and the generalization test as evidence it's real ML.

## 6. Adding more places or objects

Everything lives in `house_data.py` — add a new `WAYPOINTS` entry, a
`DESTINATIONS` alias, or an `OBJECTS_META` entry, add the matching
geometry in `environment.py`, then re-run `train_voice_model.py` so the
voice model learns the new vocabulary.

## 7. Known limitation — read before your demo

Navigation and arm-reaching currently use **scripted waypoint-following
and inverse kinematics**, not reinforcement learning. Only the voice
understanding is ML-trained so far. If you want RL-trained navigation
and arm-reach policies too (discussed separately, given your 3-day
timeline), that's a follow-up — ask and I'll build the training scripts.

## 8. House and object size

The whole house, furniture, and robot are scaled up from the original
layout (`house_data.SCALE = 1.5` for the house/furniture, robot grows
proportionally too). Pickable objects (cup, glass, bottle, book, plate)
are scaled up further on top of that for visibility. To resize
everything again, change `SCALE` in `house_data.py` — walls, doors,
furniture, waypoints, navigation speed/bounds, and camera distance all
follow it automatically, so nothing misaligns.

## 9. Trained RL navigation (obstacle avoidance)

`train_navigation.py` trains a real reinforcement-learning policy
(DQN — Deep Q-Learning) that learns to reach ANY named destination in
the house (all 9 waypoints: entrance, hallway, both doors, both rooms,
bed, both sofas) from any starting position, while avoiding obstacles
via lidar-style sensing.

**Why DQN instead of PPO + PyTorch/stable-baselines3:** installing
PyTorch pulled in ~2GB of unnecessary GPU/CUDA packages that don't even
apply here (no GPU in this environment) and filled the available disk.
More importantly, PyTorch is often too heavy to run on the kind of
embedded board (Raspberry Pi, Jetson Nano, etc.) real robot hardware
uses. So the whole training pipeline (`tiny_nn.py`, `house_nav_env.py`,
`train_navigation.py`) is built on plain NumPy instead — same core RL
ideas (experience replay, target network, epsilon-greedy exploration,
TD-learning), zero heavy dependencies, and the resulting policy is
just a handful of small weight matrices that run anywhere Python does.

**How it works:**
- `house_nav_env.py` — a fast, dependency-free 2D simulation of the
  house (walls/furniture as rectangles), with lidar-style raycasting
  (12 beams) for obstacle sensing, and a few randomly placed extra
  obstacles added every training episode (domain randomization) so the
  policy learns genuine avoidance behavior, not just memorizing the
  fixed floor plan.
- One policy, not one-per-place: the goal is given to the network as
  part of its input (a relative direction + distance vector), so a
  single trained model generalizes to every destination.
- `train_navigation.py` — the DQN training loop. Run it standalone:
  ```bash
  python train_navigation.py
  ```
  Produces `navigation_policy.npz`.
- `navigation_rl.py` — connects the trained policy to the real
  PyBullet simulation using actual raycasts against the real geometry.

**Honest, measured results from the training run included with this
project** (4000 episodes, ~3 minutes on a single CPU core, 25 greedy
evaluation trials per destination with random start positions and
random obstacles — not cherry-picked):

| Destination | Success rate |
|---|---|
| hallway_mid | 100% |
| hall_sofa | 96% |
| door_right | 92% |
| door_left | 88% |
| living_room | 80% |
| bedroom | 80% |
| entrance | 76% |
| bed | 64% |
| living_sofa | 60% |
| **Overall** | **81.8%** |

This is a solid foundation, not a finished, competition-ready policy.
`bed` and `living_sofa` are the weakest (likely because they're the
furthest destinations reachable only via specific door routes). If you
want higher success rates before your demo:
- Increase `N_EPISODES` in `train_navigation.py` (try 10,000–20,000)
  and re-run — training is fast (~2.5s per 100 episodes on one core),
  so this is very achievable in the time you have.
- Increase `HIDDEN_DIM` for more model capacity.
- Re-run `train_navigation.py` a few times with different `seed`
  values and keep the best result.

**IMPORTANT SAFETY NOTE FOR HARDWARE DEPLOYMENT:** `main.py` uses the
trained RL policy automatically if `navigation_policy.npz` exists, but
falls back to the original scripted waypoint navigator automatically
if it doesn't (or if you delete it) — so the demo never breaks even if
training hasn't been run yet or didn't converge well.

For real hardware, `navigation_rl.py` is deliberately built as a
**hierarchical** system, not raw end-to-end RL control of motors: the
trained policy only picks a high-level direction (one of 8 compass
directions), and a separate, simple, bounded safety layer actually
executes movement — including a real-time raycast safety check
immediately before every move that halts and re-plans if something is
too close. This is standard, responsible robotics practice. A
freshly-trained policy commanding raw motors with no safety layer is
not something we'd recommend before extensive additional validation,
a hardware e-stop, and testing well beyond what's possible in a
software simulation. Treat this project as the simulated foundation
for that work, not a finished hardware controller.

## 10. Grasping — still IK-based, not RL-trained, on purpose

Manipulation (the arm reaching for and grasping cup/glass/bottle/
book/plate) still uses inverse kinematics + a scripted grasp, not RL.
This was a deliberate scope decision discussed earlier: training a
reliable contact-rich grasping policy from scratch typically takes
professional robotics teams days, even with GPU-parallel simulators —
not achievable credibly in this project's timeframe. The IK approach
is fast (grasp completes in under 1 second now) and reliable, which
matters more for a live demo than an under-trained grasp policy would.

## 11. Fixing "grabbing from outside the room" and pick/listen speed

Three issues you reported, and what was actually wrong:

**Robot grasping from far outside the room.** Room-level navigation
(RL or scripted) was only getting the robot into the general area —
sometimes over a meter from the object, well beyond the arm's real
~0.76m reach (computed from the actual link lengths in `robot.py`).
The IK solver would still return *some* answer and the code would
still attach the object regardless of actual distance, which looked
like the robot grabbing things through walls. Fixed with a new
`STATE_FINAL_APPROACH` step in `main.py`: after room-level navigation
finishes, a short, simple, always-reliable scripted hop
(`house_data.DOCKING_POINTS`) drives the robot to a precise point
verified to be within real arm's reach of each specific object — cup,
glass, bottle, book, and plate all individually checked (math shown in
the code comments). This final hop deliberately does NOT use the RL
policy — it's a short, predictable move within a room already known to
be reached, which is exactly the kind of thing a simple, inspectable
controller should handle rather than a learned one, again in line with
the hierarchical "RL for big decisions, simple bounded controller for
precision" design.

While computing this, two objects turned out to be placed unreachably
far into their furniture (book in the middle of a large bed, bottle at
a table's center) — both repositioned to their furniture's front edge,
which is also just more realistic (a person leaves things at the edge
of a table/bed, not the middle of the mattress).

**Slow to start listening.** The microphone was doing a full 1-second
ambient-noise calibration before the app was ready. Reduced to 0.3s in
`voice_control.py`. Note: there's still an unavoidable ~0.5-2s network
round-trip per command to Google's speech API once you actually speak
— that's inherent to using their free API and can't be eliminated
without switching to a paid/offline engine.

**Slow to pick up objects.** The pick sequence's scripted wait timers
in `main.py` were cut roughly in half again — grasp now completes in
well under a second (0.65s) after arriving at the docking point, and
delivery/release in 0.35s.

