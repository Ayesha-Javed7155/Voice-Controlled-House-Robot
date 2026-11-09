"""
view_house.py
Quick visual check: opens the house + robot in PyBullet with NO voice
or speech_recognition dependency needed. Use this first to confirm the
environment looks right before wiring up voice control.

Controls: W/A/S/D to drive the robot around manually (same as your
original demo) so you can walk through both rooms and see the doors
slide open.

Run:  python view_house.py
"""

import time
import math
import pybullet as p
import pybullet_data

import environment
import robot as robot_mod
from house_data import SCALE

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=10.0 * SCALE, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, -2 * SCALE, 1.0]
)

house = environment.build_house()
robot_mod.write_urdf()
robot_id, joints = robot_mod.spawn_robot(start_xy=environment.WAYPOINTS["entrance"])

left_door_state = {"x": -1.0 * SCALE}
right_door_state = {"x": 1.0 * SCALE}

MOVE_SPEED = 2.0 * SCALE
TURN_SPEED = 1.8
dt = 1.0 / 60.0

print("\nHouse loaded. Use W/A/S/D to drive the robot around. Ctrl+C to quit.\n")

while p.isConnected():
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    x, y, z = pos
    _, _, yaw = p.getEulerFromQuaternion(orn)

    keys = p.getKeyboardEvents()
    forward = 0.0
    turn = 0.0

    if ord("w") in keys and keys[ord("w")] & p.KEY_IS_DOWN:
        forward = MOVE_SPEED
    if ord("s") in keys and keys[ord("s")] & p.KEY_IS_DOWN:
        forward = -MOVE_SPEED
    if ord("a") in keys and keys[ord("a")] & p.KEY_IS_DOWN:
        turn = TURN_SPEED
    if ord("d") in keys and keys[ord("d")] & p.KEY_IS_DOWN:
        turn = -TURN_SPEED

    new_yaw = yaw + turn * dt
    new_x = x + math.cos(new_yaw) * forward * dt
    new_y = y + math.sin(new_yaw) * forward * dt
    new_x = max(-7.5 * SCALE, min(7.5 * SCALE, new_x))
    new_y = max(-8.4 * SCALE, min(8.4 * SCALE, new_y))

    p.resetBasePositionAndOrientation(robot_id, [new_x, new_y, z], p.getQuaternionFromEuler([0, 0, new_yaw]))

    environment.update_doors(house, (new_x, new_y), left_door_state, right_door_state)

    p.stepSimulation()
    time.sleep(dt)
