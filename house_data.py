"""
house_data.py
Pure data: waypoints, destination aliases, and object names. No
pybullet import here on purpose, so train_voice_model.py and
generate_training_data.py can run anywhere (even without pybullet
installed) to train/test the voice models independently of the
simulation.
"""

# Global scale factor for the whole house + furniture + robot. Bump
# this up/down to resize everything together without breaking wall
# alignment, door positions, or navigation waypoints.
SCALE = 1.5

WAYPOINTS = {
    "entrance": (0.0 * SCALE, -7.2 * SCALE),   # moved further from the hall sofa to avoid overlap
    "hallway_mid": (0.0 * SCALE, -2.0 * SCALE),
    "door_left": (-1.0 * SCALE, -0.2 * SCALE),
    "door_right": (1.0 * SCALE, -0.2 * SCALE),
    "living_room": (-5.0 * SCALE, 4.2 * SCALE),
    "bedroom": (5.5 * SCALE, 3.0 * SCALE),
    "hall_sofa": (0.0 * SCALE, -5.0 * SCALE),   # just in front of the sofa, not inside it
    "bed": (5.5 * SCALE, 3.6 * SCALE),          # just in front of the bed, not inside it
    "living_sofa": (-5.0 * SCALE, 4.0 * SCALE), # just in front of the sofa, not inside it
}

DELIVERY_POINT = WAYPOINTS["entrance"]

DESTINATIONS = {
    "room 1": "living_room", "room1": "living_room", "living room": "living_room",
    "room 2": "bedroom", "room2": "bedroom", "bedroom": "bedroom",
    "bed": "bed",
    "sofa": "hall_sofa", "hall sofa": "hall_sofa", "couch": "hall_sofa",
    "living room sofa": "living_sofa", "living room couch": "living_sofa",
    "entrance": "entrance", "hall": "entrance", "front hall": "entrance",
}

WAYPOINT_ROOM = {
    "living_room": "living_room", "living_sofa": "living_room",
    "bedroom": "bedroom", "bed": "bedroom",
    "hall_sofa": "entrance", "entrance": "entrance",
}

OBJECTS_META = {
    "cup": {"destination_key": "living_room", "label": "red cup"},
    "glass": {"destination_key": "living_room", "label": "glass"},
    "bottle": {"destination_key": "bedroom", "label": "blue bottle"},
    "book": {"destination_key": "bed", "label": "book"},
    "plate": {"destination_key": "hall_sofa", "label": "plate"},
}

# Precise close-approach points for each object, computed so the final
# distance from dock -> object is within the robot's real arm reach
# (~0.76m — see robot.py's arm link lengths). Room/furniture-level
# navigation (WAYPOINTS above) gets the robot into the right area;
# these get it close enough for an actual, physically real grasp,
# instead of reaching across the room at it.
DOCKING_POINTS = {
    "cup": (-5.25 * SCALE, 2.7 * SCALE),
    "glass": (-4.75 * SCALE, 2.7 * SCALE),
    "bottle": (5.5 * SCALE, 1.4 * SCALE),
    "book": (5.5 * SCALE, 4.0 * SCALE),
    "plate": (0.0 * SCALE, -5.4 * SCALE),
}


def build_route_to(destination_key):
    room = WAYPOINT_ROOM[destination_key]
    if room == "entrance":
        return ["entrance", destination_key] if destination_key != "entrance" else ["entrance"]
    if room == "living_room":
        return ["entrance", "hallway_mid", "door_left", destination_key]
    if room == "bedroom":
        return ["entrance", "hallway_mid", "door_right", destination_key]
    raise ValueError(f"Unknown room for destination '{destination_key}'")
