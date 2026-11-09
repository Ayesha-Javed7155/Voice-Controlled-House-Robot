"""
vision.py
Captures an image from the robot's onboard camera link and runs a
color-based detector (OpenCV) to confirm the target object is visible
before the arm attempts to pick it up. This is a real perception step
on a rendered image (not a lookup), which is what makes the "vision"
part of "vision + manipulation" genuine.
"""

import math
import numpy as np
import pybullet as p

from house_data import SCALE

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
FOV = 70
CAMERA_FAR = 12.0 * SCALE

# HSV color ranges for the objects we placed in the house
COLOR_RANGES = {
    "cup": {  # red cup
        "lower1": np.array([0, 100, 80]), "upper1": np.array([8, 255, 255]),
        "lower2": np.array([172, 100, 80]), "upper2": np.array([180, 255, 255]),
    },
    "bottle": {  # blue bottle
        "lower1": np.array([100, 100, 60]), "upper1": np.array([130, 255, 255]),
        "lower2": None, "upper2": None,
    },
}


def get_camera_image(robot_id, camera_link_index):
    cam_state = p.getLinkState(robot_id, camera_link_index)
    cam_pos = cam_state[0]
    cam_orn = cam_state[1]

    rot_matrix = p.getMatrixFromQuaternion(cam_orn)
    forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
    up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]

    target = [cam_pos[i] + forward[i] * 3.0 * SCALE for i in range(3)]

    view_matrix = p.computeViewMatrix(cam_pos, target, up)
    proj_matrix = p.computeProjectionMatrixFOV(FOV, CAMERA_WIDTH / CAMERA_HEIGHT, 0.05, CAMERA_FAR)

    img = p.getCameraImage(
        CAMERA_WIDTH, CAMERA_HEIGHT, view_matrix, proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb = np.reshape(img[2], (CAMERA_HEIGHT, CAMERA_WIDTH, 4))[:, :, :3]
    return rgb.astype(np.uint8)


def detect_object(rgb_image, object_key):
    """Returns (found: bool, pixel_centroid: (x, y) or None, coverage_pct: float)."""
    import cv2

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    ranges = COLOR_RANGES.get(object_key)
    if ranges is None:
        return False, None, 0.0

    mask = cv2.inRange(hsv, ranges["lower1"], ranges["upper1"])
    if ranges["lower2"] is not None:
        mask2 = cv2.inRange(hsv, ranges["lower2"], ranges["upper2"])
        mask = mask | mask2

    coverage = (mask > 0).sum() / mask.size * 100.0
    if coverage < 0.15:
        return False, None, coverage

    ys, xs = np.where(mask > 0)
    centroid = (float(xs.mean()), float(ys.mean()))
    return True, centroid, coverage


def confirm_object_visible(robot_id, camera_link_index, object_key, verbose=True):
    """High-level helper used by main.py: captures a frame and checks
    whether the target object is visible in view."""
    frame = get_camera_image(robot_id, camera_link_index)
    found, centroid, coverage = detect_object(frame, object_key)
    if verbose:
        if found:
            print(f"[vision] Detected '{object_key}' at pixel {centroid}, coverage {coverage:.2f}%")
        else:
            print(f"[vision] '{object_key}' not clearly visible yet (coverage {coverage:.2f}%)")
    return found
