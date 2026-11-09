"""
generate_training_data.py
Builds a labeled synthetic dataset of spoken commands covering every
destination ("go to the bed", "room 1", "sofa"...) and every pickable
object ("pick up the cup and bring it to me", "get the glass"...).

Run standalone to inspect the dataset:
    python generate_training_data.py
"""

import random

from house_data import DESTINATIONS, OBJECTS_META

OBJECT_LABELS = {
    "cup": ["cup", "the cup", "red cup", "a cup"],
    "glass": ["glass", "the glass", "drinking glass", "a glass"],
    "bottle": ["bottle", "the bottle", "blue bottle", "water bottle", "a bottle"],
    "book": ["book", "the book", "a book"],
    "plate": ["plate", "the plate", "a plate"],
}

GOTO_TEMPLATES = [
    "go to the {dest}",
    "go to {dest}",
    "move to the {dest}",
    "head to the {dest}",
    "navigate to {dest}",
    "walk to the {dest}",
    "go towards the {dest}",
    "can you go to {dest}",
    "please go to the {dest}",
    "robot go to {dest}",
]

FETCH_TEMPLATES = [
    "pick up the {obj} and bring it to me",
    "pick the {obj} and bring it to me",
    "get the {obj} and bring it here",
    "bring me the {obj}",
    "fetch the {obj}",
    "get me the {obj}",
    "can you get the {obj}",
    "please bring the {obj} to me",
    "grab the {obj} for me",
    "go get the {obj}",
]

# Rebuild a clean unique destination phrase list from DESTINATIONS keys
DEST_PHRASES = sorted(set(DESTINATIONS.keys()))


def build_dataset(seed=42):
    random.seed(seed)
    rows = []  # (text, intent, target_key)

    for dest_phrase in DEST_PHRASES:
        dest_key = DESTINATIONS[dest_phrase]
        for template in GOTO_TEMPLATES:
            text = template.format(dest=dest_phrase)
            rows.append((text, "GOTO", dest_key))

    for obj_key, phrasings in OBJECT_LABELS.items():
        for obj_phrase in phrasings:
            for template in FETCH_TEMPLATES:
                text = template.format(obj=obj_phrase)
                rows.append((text, "FETCH", obj_key))

    # Stop/cancel examples — kept to a comparable count as the other
    # intents (via templates), not just a handful of raw strings, so
    # the classifier doesn't under-weight this class.
    STOP_TEMPLATES = [
        "stop", "cancel", "stop the robot", "cancel that", "never mind",
        "exit", "quit", "stop right there", "please stop", "hold on stop",
        "cancel the command", "abort", "stop now", "never mind that",
        "forget it", "stop moving", "halt", "that's enough stop",
        "cancel this task", "never mind the command",
        "halt", "please halt", "halt now", "halt right there", "please halt now",
    ]
    for text in STOP_TEMPLATES:
        rows.append((text, "STOP", None))

    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    data = build_dataset()
    print(f"Generated {len(data)} labeled examples.\n")
    for text, intent, target in data[:15]:
        print(f"  [{intent:5s} -> {target}]  \"{text}\"")
