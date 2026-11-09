"""
main.py
Voice-controlled house robot demo (v2 — generalized + trained voice model).

Two kinds of commands now work for ANYTHING in the house:

  GOTO  — "go to the bed", "go to room 1", "go to the sofa", "go to room 2"
  FETCH — "pick up the cup and bring it to me", "get the glass",
          "bring the book here", "grab the plate"

The robot will:
  1. Understand the command using a trained voice model (voice_model.joblib)
  2. Navigate through the hallway/doors to the right room and spot
  3. For FETCH: confirm the object with its camera, reach with IK, grasp
     it (constraint-based pickup), carry it back to the fixed delivery
     point (the entrance), and release it
  4. For GOTO: just drive there and stop

Run with:  python main.py
(Run `python train_voice_model.py` first if voice_model.joblib doesn't exist yet.)
Press Ctrl+C in the terminal to stop.
"""

import time
import pybullet as p
import pybullet_data

import environment
import robot as robot_mod
import navigation
import path_planner
import vision
import voice_control
from house_data import WAYPOINTS, DELIVERY_POINT, DOCKING_POINTS, SCALE

STATE_IDLE = "IDLE"
STATE_NAV_ONLY = "NAV_ONLY"                # GOTO command, no pickup
STATE_NAV_TO_OBJECT = "NAV_TO_OBJECT"      # FETCH command, heading to the object's room/area
STATE_FINAL_APPROACH = "FINAL_APPROACH"    # short precise move to within real arm's reach
STATE_LOOK_AND_PICK = "LOOK_AND_PICK"
STATE_NAV_TO_DELIVERY = "NAV_TO_DELIVERY"
STATE_PLACE = "PLACE"


def main():
    # Start the voice listener FIRST, before any PyBullet/house setup.
    # Microphone device initialization can take a couple of seconds on
    # some systems; starting it now lets that happen in parallel with
    # the slower house-building/URDF-loading below instead of after it
    # — this is what was making "Listening for commands" feel like it
    # took forever to appear.
    listener = voice_control.VoiceListener()
    listener.start()

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
    robot_id, joints = robot_mod.spawn_robot(start_xy=WAYPOINTS["entrance"])

    camera_link_index = joints["head_to_camera"]
    gripper_link_index = joints["wrist_to_gripper"]

    nav = navigation.Navigator(robot_id)
    dock_nav = navigation.Navigator(robot_id)  # short, reliable final-approach hops
    planner = path_planner.HousePlanner()
    print("[main] Using A* path planning for navigation — direct, deterministic, no wasted time.")

    left_door_state = {"x": -1.0 * SCALE}
    right_door_state = {"x": 1.0 * SCALE}

    state = STATE_IDLE
    target_object_key = None
    held_object_id = [None]   # mutable container so the sync step below can read/clear it each frame
    held_offset = [None]      # (rel_pos, rel_orn) of the object relative to the gripper, set at grasp time
    state_timer = 0.0

    print("\n================ READY ================")
    print("Try saying: 'go to the bed'  or  'pick up the cup and bring it to me'")
    print("=========================================\n")

    dt = 1.0 / 60.0

    def _region_of(xy):
        """Rooms are the upper area (y > 0), split left/right; the
        hallway/entrance is the lower area (y <= 0)."""
        x, y = xy
        if y > 0:
            return "left_room" if x < 0 else "right_room"
        return "hall"

    def start_route(goal_xy):
        current_pos, _ = p.getBasePositionAndOrientation(robot_id)
        current_xy = (current_pos[0], current_pos[1])

        start_region = _region_of(current_xy)
        goal_region = _region_of(goal_xy)

        if start_region != goal_region and ("room" in start_region or "room" in goal_region):
            # Crossing between the hallway and a room — force the path
            # through the actual door location instead of letting A*
            # cut through whatever part of the wall opening happens to
            # be shortest. The wall's opening is architecturally wider
            # than the visible door itself, so without this the planner
            # could path through a section of the gap the door prop
            # isn't even in, which looked like walking through the wall
            # rather than through the door.
            room_side = start_region if "room" in start_region else goal_region
            door_xy = WAYPOINTS["door_left"] if room_side == "left_room" else WAYPOINTS["door_right"]
            leg1 = planner.plan(current_xy, door_xy)
            leg2 = planner.plan(door_xy, goal_xy)
            path = (leg1 if leg1 else [current_xy, door_xy]) + (leg2[1:] if leg2 else [goal_xy])
        else:
            path = planner.plan(current_xy, goal_xy)

        nav.start_route_points(path if path else [goal_xy])

    def start_delivery_route():
        start_route(DELIVERY_POINT)

    while p.isConnected():
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        environment.update_doors(house, (pos[0], pos[1]), left_door_state, right_door_state)

        # -------------------- KEEP THE HELD OBJECT GLUED TO THE GRIPPER --------------------
        # Runs every frame, independent of which state we're in, for as
        # long as something is held. Directly sets the object's world
        # pose from the gripper's current pose + the fixed offset
        # recorded at grasp time — kinematic, not physics-based, so
        # there's no lag/float regardless of how fast the robot moves.
        if held_object_id[0] is not None:
            gripper_pos, gripper_orn = p.getLinkState(robot_id, gripper_link_index)[:2]
            rel_pos, rel_orn = held_offset[0]
            world_pos, world_orn = p.multiplyTransforms(gripper_pos, gripper_orn, rel_pos, rel_orn)
            p.resetBasePositionAndOrientation(held_object_id[0], world_pos, world_orn)
            p.resetBaseVelocity(held_object_id[0], linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # -------------------- VOICE INPUT (only while idle) --------------------
        if state == STATE_IDLE:
            command = listener.get_command_nonblocking()
            if command:
                if command["intent"] == "STOP":
                    print("[main] Stop command received.")

                elif command["intent"] == "GOTO":
                    dest_key = command["target_key"]
                    print(f"[main] Command understood: go to the {command['label']}.")
                    start_route(WAYPOINTS[dest_key])
                    state = STATE_NAV_ONLY

                elif command["intent"] == "FETCH":
                    obj_key = command["target_key"]
                    if obj_key in house["objects"]:
                        target_object_key = obj_key
                        obj_info = house["objects"][obj_key]
                        print(f"[main] Command understood: fetch the {obj_info['label']}.")
                        start_route(WAYPOINTS[obj_info["destination_key"]])
                        state = STATE_NAV_TO_OBJECT
                    else:
                        print(f"[main] Didn't recognize that object in: \"{command['raw']}\".")
                else:
                    print(f"[main] Didn't understand: \"{command['raw']}\".")

        # -------------------- GOTO-ONLY NAVIGATION --------------------
        elif state == STATE_NAV_ONLY:
            still_going = nav.step(dt)
            if not still_going:
                print("[main] Arrived. Ready for the next command.\n")
                state = STATE_IDLE

        # -------------------- NAVIGATE TO OBJECT'S ROOM/AREA --------------------
        elif state == STATE_NAV_TO_OBJECT:
            still_going = nav.step(dt)
            if not still_going:
                print("[main] In the area. Moving in close for a real grasp...")
                obj_info = house["objects"][target_object_key]
                current_pos, _ = p.getBasePositionAndOrientation(robot_id)
                dock_path = planner.plan((current_pos[0], current_pos[1]), DOCKING_POINTS[target_object_key])
                dock_nav.start_route_points(dock_path if dock_path else [DOCKING_POINTS[target_object_key]])
                state = STATE_FINAL_APPROACH

        # -------------------- FINAL PRECISE APPROACH --------------------
        # A short, always-reliable scripted hop from the general room area
        # to a point within the arm's real reach of the object. This is
        # what was missing before — the robot would attempt to grasp from
        # wherever room-level navigation happened to stop, sometimes over
        # a meter away, which looked like "grabbing from outside the room."
        elif state == STATE_FINAL_APPROACH:
            still_going = dock_nav.step(dt)
            if not still_going:
                print("[main] In position. Looking for the object...")
                state = STATE_LOOK_AND_PICK
                state_timer = 0.0

        # -------------------- VISION CHECK + PICK --------------------
        elif state == STATE_LOOK_AND_PICK:
            state_timer += dt
            vision.confirm_object_visible(
                robot_id, camera_link_index, target_object_key, verbose=(state_timer < dt * 2)
            )

            obj_info = house["objects"][target_object_key]
            obj_id = obj_info["body_id"]
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id)

            ik_solution = p.calculateInverseKinematics(
                robot_id, gripper_link_index,
                targetPosition=[obj_pos[0], obj_pos[1], obj_pos[2] + 0.02],
            )
            movable_joints = [joints[n] for n in [
                "right_shoulder_pitch", "right_elbow", "right_wrist_pitch"
            ]]
            for j_idx, angle in zip(movable_joints, ik_solution[:len(movable_joints)]):
                p.setJointMotorControl2(robot_id, j_idx, p.POSITION_CONTROL, targetPosition=angle, force=80, maxVelocity=6.0)

            if state_timer > 0.3:
                robot_mod.set_joint_targets(robot_id, joints, robot_mod.GRIPPER_CLOSED)

            if state_timer > 0.45 and held_object_id[0] is None:
                # Record the object's exact position relative to the
                # gripper right now, then hold it there KINEMATICALLY
                # every frame from here on (see the "if held_object_id[0]"
                # block near the top of the main loop) — not via a
                # PyBullet physics constraint. The robot's whole body
                # moves by teleporting each frame (not real physics), and
                # a force/velocity-based constraint can't perfectly keep
                # up with a teleporting parent, which was causing the
                # object to visibly lag/float behind the hand. Directly
                # setting the object's pose every frame is exact, with
                # nothing for physics to get wrong.
                gripper_pos, gripper_orn = p.getLinkState(robot_id, gripper_link_index)[:2]
                obj_pos_now, obj_orn_now = p.getBasePositionAndOrientation(obj_id)
                inv_pos, inv_orn = p.invertTransform(gripper_pos, gripper_orn)
                rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, obj_pos_now, obj_orn_now)

                held_object_id[0] = obj_id
                held_offset[0] = (rel_pos, rel_orn)
                print(f"[main] Grasped the {obj_info['label']}.")
                robot_mod.set_joint_targets(robot_id, joints, robot_mod.ARM_CARRY_POSE)

            if state_timer > 0.65:
                print("[main] Heading back to deliver it...")
                start_delivery_route()
                state = STATE_NAV_TO_DELIVERY

        # -------------------- NAVIGATE BACK TO DELIVERY POINT --------------------
        elif state == STATE_NAV_TO_DELIVERY:
            still_going = nav.step(dt)
            if not still_going:
                state = STATE_PLACE
                state_timer = 0.0

        # -------------------- RELEASE THE OBJECT --------------------
        elif state == STATE_PLACE:
            state_timer += dt
            robot_mod.set_joint_targets(robot_id, joints, robot_mod.ARM_REST_POSE)
            if state_timer > 0.2 and held_object_id[0] is not None:
                held_object_id[0] = None  # stop the per-frame gripper-sync from running
                held_offset[0] = None
                robot_mod.set_joint_targets(robot_id, joints, robot_mod.GRIPPER_OPEN)

                # Place directly on the delivery tray instead of letting it
                # free-fall from arm height (which caused bouncing/rolling).
                obj_info = house["objects"][target_object_key]
                tray_x, tray_y = house["delivery_tray_xy"]
                tray_top_z = house["delivery_tray_top_z"]
                p.resetBasePositionAndOrientation(
                    obj_info["body_id"], [tray_x, tray_y, tray_top_z + obj_info["rest_half_height"]], [0, 0, 0, 1]
                )
                p.resetBaseVelocity(obj_info["body_id"], linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                print(f"[main] Delivered the {obj_info['label']}! Ready for the next command.\n")
            if state_timer > 0.35:
                target_object_key = None
                state = STATE_IDLE

        p.stepSimulation()
        time.sleep(dt)

    listener.stop()
    p.disconnect()


if __name__ == "__main__":
    main()
