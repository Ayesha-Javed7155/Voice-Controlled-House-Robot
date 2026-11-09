"""
robot.py
Defines the service robot: a mobile base + torso + head (with camera)
+ a decorative left arm + a FUNCTIONAL right arm with a 3-DOF chain
(shoulder pitch, elbow, wrist) ending in a simple two-finger gripper.
The right arm is what actually performs pick-and-place via IK.

The URDF is generated from a template rather than a fixed string, so
ROBOT_SCALE (tied to house_data.SCALE) resizes every link/joint origin
together, keeping proportions correct as the house grows or shrinks.
"""

import pybullet as p

from house_data import SCALE

# The robot grows a bit less than the house itself so it doesn't
# dominate the rooms — tune independently if you want a bigger/smaller
# robot relative to the house.
ROBOT_SCALE = 1.0 + (SCALE - 1.0) * 0.6


def _urdf(rs):
    """Builds the URDF XML string with every dimension scaled by rs."""
    return f"""<?xml version="1.0"?>
<robot name="service_bot">

  <material name="pink"><color rgba="1.0 0.55 0.75 1.0"/></material>
  <material name="dark_pink"><color rgba="0.85 0.25 0.55 1.0"/></material>
  <material name="black"><color rgba="0.1 0.1 0.1 1.0"/></material>
  <material name="silver"><color rgba="0.75 0.75 0.78 1.0"/></material>

  <!-- BASE -->
  <link name="base_link">
    <visual><geometry><cylinder radius="{0.35*rs}" length="{0.5*rs}"/></geometry><material name="pink"/></visual>
    <collision><geometry><cylinder radius="{0.35*rs}" length="{0.5*rs}"/></geometry></collision>
    <inertial><mass value="20.0"/><inertia ixx="0.3" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.3"/></inertial>
  </link>

  <!-- TORSO -->
  <link name="torso">
    <visual><geometry><cylinder radius="{0.25*rs}" length="{0.7*rs}"/></geometry><material name="pink"/></visual>
    <collision><geometry><cylinder radius="{0.25*rs}" length="{0.7*rs}"/></geometry></collision>
    <inertial><mass value="8.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
  </link>
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/><child link="torso"/>
    <origin xyz="0 0 {0.6*rs}" rpy="0 0 0"/>
  </joint>

  <!-- HEAD -->
  <link name="head">
    <visual><geometry><sphere radius="{0.16*rs}"/></geometry><material name="pink"/></visual>
    <collision><geometry><sphere radius="{0.16*rs}"/></geometry></collision>
    <inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="head_yaw_joint" type="revolute">
    <parent link="torso"/><child link="head"/>
    <origin xyz="0 0 {0.5*rs}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>

  <!-- CAMERA (fixed to head, used for vision) -->
  <link name="camera_link">
    <visual><geometry><box size="{0.03*rs} {0.08*rs} {0.03*rs}"/></geometry><material name="black"/></visual>
  </link>
  <joint name="head_to_camera" type="fixed">
    <parent link="head"/><child link="camera_link"/>
    <origin xyz="{0.16*rs} 0 0" rpy="0 0 0"/>
  </joint>

  <!-- LEFT ARM (decorative) -->
  <link name="left_arm">
    <visual><geometry><capsule radius="{0.05*rs}" length="{0.35*rs}"/></geometry><material name="pink"/></visual>
  </link>
  <joint name="left_shoulder" type="revolute">
    <parent link="torso"/><child link="left_arm"/>
    <origin xyz="0 {0.28*rs} {0.2*rs}" rpy="0.3 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="2"/>
  </joint>

  <!-- RIGHT ARM (functional: shoulder -> upper_arm -> elbow -> forearm -> wrist -> gripper) -->
  <link name="right_upper_arm">
    <visual><geometry><capsule radius="{0.045*rs}" length="{0.28*rs}"/></geometry><material name="silver"/></visual>
    <collision><geometry><capsule radius="{0.045*rs}" length="{0.28*rs}"/></geometry></collision>
    <inertial><mass value="0.6"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
  </link>
  <joint name="right_shoulder_pitch" type="revolute">
    <parent link="torso"/><child link="right_upper_arm"/>
    <origin xyz="0 {-0.28*rs} {0.2*rs}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.2" upper="1.2" effort="30" velocity="2"/>
  </joint>

  <link name="right_forearm">
    <visual><geometry><capsule radius="{0.04*rs}" length="{0.26*rs}"/></geometry><material name="silver"/></visual>
    <collision><geometry><capsule radius="{0.04*rs}" length="{0.26*rs}"/></geometry></collision>
    <inertial><mass value="0.4"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.003"/></inertial>
  </link>
  <joint name="right_elbow" type="revolute">
    <parent link="right_upper_arm"/><child link="right_forearm"/>
    <origin xyz="0 0 {-0.28*rs}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="0.2" effort="30" velocity="2"/>
  </joint>

  <link name="right_wrist">
    <visual><geometry><sphere radius="{0.035*rs}"/></geometry><material name="dark_pink"/></visual>
    <collision><geometry><sphere radius="{0.035*rs}"/></geometry></collision>
    <inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>
  <joint name="right_wrist_pitch" type="revolute">
    <parent link="right_forearm"/><child link="right_wrist"/>
    <origin xyz="0 0 {-0.26*rs}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.6" upper="1.6" effort="15" velocity="2"/>
  </joint>

  <!-- GRIPPER: palm + two fingers (prismatic, open/close) -->
  <link name="gripper_link">
    <visual><geometry><box size="{0.05*rs} {0.09*rs} {0.03*rs}"/></geometry><material name="black"/></visual>
    <collision><geometry><box size="{0.05*rs} {0.09*rs} {0.03*rs}"/></geometry></collision>
    <inertial><mass value="0.15"/><inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/></inertial>
  </link>
  <joint name="wrist_to_gripper" type="fixed">
    <parent link="right_wrist"/><child link="gripper_link"/>
    <origin xyz="0 0 {-0.045*rs}" rpy="0 1.5708 0"/>
  </joint>

  <link name="finger_left">
    <visual><geometry><box size="{0.02*rs} {0.02*rs} {0.06*rs}"/></geometry><material name="black"/></visual>
    <collision><geometry><box size="{0.02*rs} {0.02*rs} {0.06*rs}"/></geometry></collision>
    <inertial><mass value="0.03"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  </link>
  <joint name="finger_left_joint" type="prismatic">
    <parent link="gripper_link"/><child link="finger_left"/>
    <origin xyz="0 {0.045*rs} {-0.03*rs}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{-0.03*rs}" upper="{0.015*rs}" effort="10" velocity="1"/>
  </joint>

  <link name="finger_right">
    <visual><geometry><box size="{0.02*rs} {0.02*rs} {0.06*rs}"/></geometry><material name="black"/></visual>
    <collision><geometry><box size="{0.02*rs} {0.02*rs} {0.06*rs}"/></geometry></collision>
    <inertial><mass value="0.03"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  </link>
  <joint name="finger_right_joint" type="prismatic">
    <parent link="gripper_link"/><child link="finger_right"/>
    <origin xyz="0 {-0.045*rs} {-0.03*rs}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="{-0.015*rs}" upper="{0.03*rs}" effort="10" velocity="1"/>
  </joint>

  <!-- WHEELS (visual only, movement is kinematic like the base house demo) -->
  <link name="left_wheel">
    <visual><geometry><cylinder radius="{0.12*rs}" length="{0.06*rs}"/></geometry><material name="dark_pink"/></visual>
  </link>
  <joint name="left_wheel_joint" type="fixed">
    <parent link="base_link"/><child link="left_wheel"/>
    <origin xyz="0 {0.32*rs} {-0.2*rs}" rpy="1.5708 0 0"/>
  </joint>

  <link name="right_wheel">
    <visual><geometry><cylinder radius="{0.12*rs}" length="{0.06*rs}"/></geometry><material name="dark_pink"/></visual>
  </link>
  <joint name="right_wheel_joint" type="fixed">
    <parent link="base_link"/><child link="right_wheel"/>
    <origin xyz="0 {-0.32*rs} {-0.2*rs}" rpy="1.5708 0 0"/>
  </joint>

</robot>
"""


ARM_REST_POSE = {
    "right_shoulder_pitch": -0.3,
    "right_elbow": -0.3,
    "right_wrist_pitch": 0.0,
}

ARM_CARRY_POSE = {
    "right_shoulder_pitch": -1.6,
    "right_elbow": -1.8,
    "right_wrist_pitch": 0.6,
}

GRIPPER_OPEN = {"finger_left_joint": 0.015 * ROBOT_SCALE, "finger_right_joint": -0.015 * ROBOT_SCALE}
GRIPPER_CLOSED = {"finger_left_joint": -0.005 * ROBOT_SCALE, "finger_right_joint": 0.005 * ROBOT_SCALE}


def write_urdf(path="service_bot.urdf"):
    with open(path, "w") as f:
        f.write(_urdf(ROBOT_SCALE))
    return path


def spawn_robot(start_xy=(0.0, -3.0), start_z=None, urdf_path="service_bot.urdf"):
    if start_z is None:
        # Wheel bottom sits at (base_z - 0.2*rs - 0.12*rs) = base_z - 0.32*rs
        # relative to the base origin (see the wheel joint origin + radius in
        # the URDF template above). Setting base_z = 0.32*rs makes the wheel
        # bottoms touch the floor exactly, instead of hovering above it.
        start_z = 0.32 * ROBOT_SCALE
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[start_xy[0], start_xy[1], start_z],
        useFixedBase=False,
    )

    joint_map = {}
    for i in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, i)[1].decode("utf-8")
        joint_map[name] = i

    p.changeDynamics(robot_id, -1, linearDamping=0.5, angularDamping=0.8)

    # Start arm at rest and gripper open
    set_joint_targets(robot_id, joint_map, ARM_REST_POSE)
    set_joint_targets(robot_id, joint_map, GRIPPER_OPEN)

    return robot_id, joint_map


def set_joint_targets(robot_id, joint_map, targets, force=40, max_velocity=2.0):
    for name, angle in targets.items():
        p.setJointMotorControl2(
            robot_id, joint_map[name], p.POSITION_CONTROL,
            targetPosition=angle, force=force, maxVelocity=max_velocity,
        )


def get_link_state_pos(robot_id, link_index):
    return p.getLinkState(robot_id, link_index)[0]
