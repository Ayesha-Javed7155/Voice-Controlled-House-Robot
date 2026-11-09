"""
environment.py
Builds the 2-room house with hallway, furniture, sliding doors,
pickable objects, and defines the navigation waypoints + fixed
delivery point used by navigation.py and main.py.

Everything below is scaled by house_data.SCALE, so bumping that one
number resizes the whole house, furniture, and object layout together
without breaking wall/door/waypoint alignment.
"""

import math
import pybullet as p

from house_data import (
    SCALE, WAYPOINTS, DELIVERY_POINT, DESTINATIONS, WAYPOINT_ROOM,
    OBJECTS_META, build_route_to,
)

# Kept for backward compatibility with earlier navigation calls.
ROOM_ROUTES = {
    "living_room": build_route_to("living_room"),
    "bedroom": build_route_to("bedroom"),
}


def build_house():
    """Builds the ground, walls, doors, and furniture. Returns a dict
    with door body ids (needed for the auto-open animation) and object
    body ids for anything the robot can pick up."""

    s = SCALE  # short alias, used a lot below

    # -------------------- GROUND --------------------
    floor_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[8 * s, 9 * s, 0.01], rgbaColor=[0.88, 0.82, 0.75, 1]
    )
    floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[8 * s, 9 * s, 0.01])
    p.createMultiBody(0, floor_col, floor_vis, [0, 0, 0])

    # -------------------- WALLS --------------------
    wall_height = 0.8 * s

    def wall(pos, half_extents, color=(0.95, 0.92, 0.88, 1)):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=list(color))
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        return p.createMultiBody(0, col, vis, pos)

    # Outer walls
    wall([0, 8 * s, wall_height], [8 * s, 0.1 * s, wall_height])
    wall([0, -8 * s, wall_height], [8 * s, 0.1 * s, wall_height])
    wall([-8 * s, 0, wall_height], [0.1 * s, 8 * s, wall_height])
    wall([8 * s, 0, wall_height], [0.1 * s, 8 * s, wall_height])

    # Divider walls (separate upper rooms from lower hall area)
    wall([-4.8 * s, 0, wall_height], [3.2 * s, 0.1 * s, wall_height])
    wall([4.8 * s, 0, wall_height], [3.2 * s, 0.1 * s, wall_height])
    wall([0, 4 * s, wall_height], [0.1 * s, 4 * s, wall_height])

    # -------------------- DOORS (slide open automatically) --------------------
    door_color = [0.65, 0.45, 0.25, 1.0]
    door_half = [0.6 * s, 0.05 * s, 0.8 * s]

    def door(pos):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=door_half, rgbaColor=door_color)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=door_half)
        return p.createMultiBody(0, col, vis, pos)

    left_door_id = door([-1.0 * s, 0, 0.8 * s])
    right_door_id = door([1.0 * s, 0, 0.8 * s])

    # -------------------- FURNITURE --------------------
    def table(pos, half_extents=(0.8, 0.6, 0.05), color=(0.40, 0.25, 0.15, 1)):
        half_extents = [half_extents[0] * s, half_extents[1] * s, half_extents[2] * s]
        x, y = pos[0], pos[1]
        z = pos[2] if len(pos) > 2 else 0.0

        top_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=list(color))
        top_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        top_center_z = z + 0.6 * s
        p.createMultiBody(0, top_col, top_vis, [x, y, top_center_z])

        leg_radius = 0.04 * s
        leg_length = 0.6 * s
        for dx in [-half_extents[0] + leg_radius * 2.5, half_extents[0] - leg_radius * 2.5]:
            for dy in [-half_extents[1] + leg_radius * 2.5, half_extents[1] - leg_radius * 2.5]:
                leg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=leg_radius, length=leg_length, rgbaColor=[0.2, 0.2, 0.2, 1])
                leg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=leg_radius, height=leg_length)
                p.createMultiBody(0, leg_col, leg_vis, [x + dx, y + dy, z + leg_length / 2])
        return top_center_z + half_extents[2]  # top surface height, for placing objects

    def sofa(pos, color=(0.35, 0.50, 0.45, 1)):
        x, y = pos[0], pos[1]
        z = pos[2] if len(pos) > 2 else 0.0
        seat_half = [1.0 * s, 0.4 * s, 0.2 * s]
        back_half = [1.0 * s, 0.1 * s, 0.4 * s]

        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=seat_half, rgbaColor=list(color))
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=seat_half)
        p.createMultiBody(0, col, vis, [x, y, z + seat_half[2]])

        back_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=back_half, rgbaColor=list(color))
        back_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=back_half)
        p.createMultiBody(0, back_col, back_vis, [x, y + 0.3 * s, z + 0.5 * s])
        return z + 2 * seat_half[2]  # seat top height, for placing objects

    def plant(pos):
        pot_radius, pot_len = 0.25 * s, 0.4 * s
        pot_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=pot_radius, length=pot_len, rgbaColor=[0.9, 0.9, 0.9, 1])
        pot_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=pot_radius, height=pot_len)
        p.createMultiBody(0, pot_col, pot_vis, [pos[0], pos[1], pot_len / 2])
        leaf_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.35 * s, rgbaColor=[0.20, 0.65, 0.30, 1])
        p.createMultiBody(0, -1, leaf_vis, [pos[0], pos[1], pot_len + 0.15 * s])

    def bed(pos):
        base_half = [1.2 * s, 0.9 * s, 0.25 * s]
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.88, 0.90, 0.95, 1])
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
        p.createMultiBody(0, base_col, base_vis, [pos[0], pos[1], base_half[2]])

        pil_half = [0.3 * s, 0.35 * s, 0.08 * s]
        pil_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=pil_half, rgbaColor=[0.98, 0.98, 0.98, 1])
        pil_z = 2 * base_half[2] + pil_half[2]
        p.createMultiBody(0, -1, pil_vis, [pos[0] + 0.8 * s, pos[1], pil_z])
        return 2 * base_half[2]  # top-of-mattress height, for placing objects

    # Living room (left)
    sofa([-5 * s, 5 * s, 0])
    living_table_top_z = table([-5 * s, 3.2 * s, 0], half_extents=(0.6, 0.4, 0.05), color=(0.3, 0.3, 0.3, 1))
    plant([-7.2 * s, 7.0 * s])

    # Bedroom (right)
    bed_top_z = bed([5.5 * s, 5 * s])
    bedroom_table_top_z = table([5.5 * s, 2 * s, 0], half_extents=(0.8, 0.5, 0.05), color=(0.55, 0.35, 0.20, 1))
    plant([7.2 * s, 7.0 * s])

    # Front hall / entrance
    hall_sofa_top_z = sofa([0, -6 * s, 0], color=(0.75, 0.35, 0.35, 1))
    plant([-7.0 * s, -7.0 * s])
    plant([7.0 * s, -7.0 * s])

    # Delivery tray: a small low platform right at the delivery point so
    # objects have a real, fixed surface to be placed on instead of being
    # dropped from arm height (which caused bouncing/rolling before).
    # Offset sideways (+x) rather than toward the sofa (+y), since the
    # sofa's footprint extends close enough to the entrance that a
    # y-offset placement was overlapping it.
    tray_half = [0.35 * s, 0.35 * s, 0.03 * s]
    tray_pos = [DELIVERY_POINT[0] + 1.8 * s, DELIVERY_POINT[1], tray_half[2]]
    tray_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=tray_half, rgbaColor=[0.55, 0.55, 0.58, 1])
    tray_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=tray_half)
    p.createMultiBody(0, tray_col, tray_vis, tray_pos)
    delivery_tray_top_z = tray_pos[2] + tray_half[2]
    delivery_tray_xy = (tray_pos[0], tray_pos[1])

    # -------------------- PICKABLE OBJECTS --------------------
    # Sizes are large (roughly 3x realistic scale, on top of the house
    # scale-up) so they read clearly in the viewport and on a projector.
    obj_scale = s * 1.4  # objects grow a bit more than the house itself, for visibility

    def make_pickable_cylinder(pos, color, radius, height):
        radius, height = radius * obj_scale, height * obj_scale
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=list(color))
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        body_id = p.createMultiBody(0.2, col, vis, pos)
        p.changeDynamics(body_id, -1, lateralFriction=1.2)
        return body_id, radius, height

    def make_pickable_box(pos, color, half_extents):
        half_extents = [v * obj_scale for v in half_extents]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=list(color))
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        body_id = p.createMultiBody(0.2, col, vis, pos)
        p.changeDynamics(body_id, -1, lateralFriction=1.2)
        return body_id, half_extents

    # Living room table: cup + glass
    cup_id, cup_r, cup_h = make_pickable_cylinder(
        [-5.25 * s, 3.2 * s, living_table_top_z], color=[0.85, 0.15, 0.15, 1], radius=0.11, height=0.18
    )
    p.resetBasePositionAndOrientation(cup_id, [-5.25 * s, 3.2 * s, living_table_top_z + cup_h / 2], [0, 0, 0, 1])

    glass_id, glass_r, glass_h = make_pickable_cylinder(
        [-4.75 * s, 3.2 * s, living_table_top_z], color=[0.55, 0.80, 0.90, 0.55], radius=0.09, height=0.20
    )
    p.resetBasePositionAndOrientation(glass_id, [-4.75 * s, 3.2 * s, living_table_top_z + glass_h / 2], [0, 0, 0, 1])

    # Bedroom table: bottle — placed near the table's front (south) edge,
    # not the center, so it's within the robot's real arm reach from the
    # docking point (see house_data.DOCKING_POINTS).
    bottle_id, bottle_r, bottle_h = make_pickable_cylinder(
        [5.5 * s, 1.7 * s, bedroom_table_top_z], color=[0.15, 0.35, 0.85, 1], radius=0.08, height=0.32
    )
    p.resetBasePositionAndOrientation(bottle_id, [5.5 * s, 1.7 * s, bedroom_table_top_z + bottle_h / 2], [0, 0, 0, 1])

    # Bed: book — placed near the bed's front (south) edge, not the
    # middle of the mattress, so it's actually reachable.
    book_id, book_half = make_pickable_box(
        [5.5 * s, 4.3 * s, 0], color=[0.65, 0.20, 0.15, 1], half_extents=[0.14, 0.18, 0.03]
    )
    p.resetBasePositionAndOrientation(book_id, [5.5 * s, 4.3 * s, bed_top_z + book_half[2]], [0, 0, 0, 1])

    # Hall sofa: plate
    plate_id, plate_r, plate_h = make_pickable_cylinder(
        [0.0, -5.9 * s, 0], color=[0.95, 0.95, 0.92, 1], radius=0.14, height=0.03
    )
    p.resetBasePositionAndOrientation(plate_id, [0.0, -5.9 * s, hall_sofa_top_z + plate_h / 2], [0, 0, 0, 1])

    # Half-height of each object, along its own up-axis, so main.py can
    # rest it flush on the delivery tray regardless of shape/size instead
    # of using a guessed fixed offset (which was the cause of objects
    # sinking into or floating above the tray at these larger sizes).
    rest_half_heights = {
        "cup": cup_h / 2, "glass": glass_h / 2, "bottle": bottle_h / 2,
        "book": book_half[2], "plate": plate_h / 2,
    }

    body_ids = {"cup": cup_id, "glass": glass_id, "bottle": bottle_id, "book": book_id, "plate": plate_id}
    objects = {
        key: {"body_id": body_ids[key], "rest_half_height": rest_half_heights[key], **meta}
        for key, meta in OBJECTS_META.items()
    }

    return {
        "left_door_id": left_door_id,
        "right_door_id": right_door_id,
        "objects": objects,
        "delivery_tray_top_z": delivery_tray_top_z,
        "delivery_tray_xy": delivery_tray_xy,
    }


def update_doors(house, robot_xy, left_door_state, right_door_state):
    """Slides doors open when the robot is near, closed otherwise.
    left_door_state / right_door_state are dicts holding the current
    x position, mutated and returned each call (call every sim step)."""
    s = SCALE
    x, y = robot_xy

    dist_left = math.hypot(x - (-1.0 * s), y)
    target_left = -2.3 * s if dist_left < 2.2 * s else -1.0 * s
    left_door_state["x"] += (target_left - left_door_state["x"]) * 0.1
    p.resetBasePositionAndOrientation(house["left_door_id"], [left_door_state["x"], 0, 0.8 * s], [0, 0, 0, 1])

    dist_right = math.hypot(x - 1.0 * s, y)
    target_right = 2.3 * s if dist_right < 2.2 * s else 1.0 * s
    right_door_state["x"] += (target_right - right_door_state["x"]) * 0.1
    p.resetBasePositionAndOrientation(house["right_door_id"], [right_door_state["x"], 0, 0.8 * s], [0, 0, 0, 1])

    return left_door_state, right_door_state
