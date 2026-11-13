# Voice-Controlled House Robot

A simulated home-service robot built with **Python and PyBullet**. The project combines natural-language voice commands, indoor navigation, camera-based object confirmation, and simulated pick-and-deliver actions in a virtual house.

The robot can understand commands such as “go to the bedroom” and “pick up the cup and bring it to me.” It supports navigation to named locations and fetching predefined objects in the simulation.

## Features

- **Voice command understanding:** trained intent and target classifiers interpret commands for navigation, fetching, and stopping.
- **Speech input:** microphone input uses SpeechRecognition and falls back to typed commands when a microphone is unavailable.
- **House simulation:** a multi-room environment with furniture, doors, waypoints, and predefined objects.
- **Navigation:** waypoint/path planning with an optional trained DQN navigation policy and obstacle sensing.
- **Object perception:** the robot camera uses HSV-based color segmentation to check for a target object.
- **Simulated manipulation:** inverse kinematics and a gripper perform pick-and-place actions in PyBullet.
- **Delivery workflow:** fetched objects are carried to a fixed delivery point in the simulated house.

## Technology Stack

- Python
- PyBullet
- NumPy
- OpenCV
- SpeechRecognition
- PyAudio
- scikit-learn
- Joblib

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ayesha-Javed7155/Voice-Controlled-House-Robot.git
cd Voice-Controlled-House-Robot
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

**PyAudio note:** PyAudio may require the PortAudio system library. On Ubuntu/Debian, install it with:

```bash
sudo apt-get install portaudio19-dev
```

On macOS:

```bash
brew install portaudio
```

On Windows, if the PyAudio installation fails, use a compatible wheel or installation method for your Python version.

### 4. Train the voice command model

Run this once before the first full demo, or whenever the command vocabulary in `house_data.py` changes:

```bash
python train_voice_model.py
```

This creates `voice_model.joblib`.

### 5. Run the simulation

To view the house without starting the voice-controlled workflow:

```bash
python view_house.py
```

To train the optional DQN navigation policy:

```bash
python train_navigation.py
```

This generates `navigation_policy.npz`.

To start the full demo:

```bash
python main.py
```

Press `Ctrl+C` in the terminal to stop the program.

## Example Commands

Try commands such as:

```text
go to the bedroom
go to the bed
go to room 2
pick up the cup and bring it to me
get the glass
bring the book here
```

The available locations and objects are defined in `house_data.py`.

## Project Structure

| File | Responsibility |
|---|---|
| `main.py` | Main simulation loop and voice-command task state machine |
| `environment.py` | Builds the virtual house and its objects |
| `robot.py` | Robot model, joints, arm, and gripper helpers |
| `house_data.py` | Waypoints, destination aliases, object metadata, and scale |
| `voice_control.py` | Microphone listener and command parsing |
| `generate_training_data.py` | Generates synthetic command examples |
| `train_voice_model.py` | Trains and saves the voice command classifiers |
| `navigation.py` | Waypoint-based navigation |
| `path_planner.py` | Route planning through the house |
| `vision.py` | Camera-based object confirmation |
| `house_nav_env.py` | Lightweight navigation training environment |
| `tiny_nn.py` | NumPy neural-network components for the DQN policy |
| `train_navigation.py` | Trains the navigation policy |
| `navigation_rl.py` | Runs the trained policy in the PyBullet scene |
| `requirements.txt` | Python dependencies |

## How It Works

1. The voice listener receives a spoken or typed command.
2. The trained classifiers identify the command intent and destination/object.
3. The robot navigates to the relevant location.
4. For a fetch command, camera-based color segmentation checks the target.
5. The arm reaches the object using inverse kinematics, closes the gripper, and simulates attachment.
6. The robot navigates to the delivery point and releases the object.

## Important Notes and Limitations

- This project is a **simulation**, not a tested physical robot controller.
- Object recognition is based on color segmentation in the simulated scene; it is not a general-purpose object detector.
- Arm manipulation uses scripted inverse kinematics and grasp logic, not a learned reinforcement-learning grasp policy.
- The voice recognition path uses Google's Web Speech API through SpeechRecognition, so spoken commands require an internet connection. Typed fallback is available.
- The navigation DQN policy is optional; the project includes a scripted navigation fallback.
- Performance in simulation does not guarantee safe or reliable behavior on real hardware. Physical deployment would require additional sensors, safety controls, and extensive testing.

## Future Improvements

- Add offline speech recognition.
- Improve object detection beyond color thresholding.
- Evaluate navigation across more randomized layouts and obstacles.
- Add systematic tests and reproducible performance reports.
