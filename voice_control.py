"""
voice_control.py (v2 — trained model)
Listens on the microphone in a background thread and pushes recognized
text into a thread-safe queue that main.py polls every frame.

Command understanding now uses a TRAINED model (voice_model.joblib,
produced by train_voice_model.py) instead of hardcoded keyword
matching:
  - intent_model: GOTO / FETCH / STOP
  - target_model: which destination (GOTO) or object (FETCH) is meant

This generalizes to phrasings the model never saw verbatim during
training (e.g. "take me to the bedroom" even though training only saw
"go to the bedroom" style templates), because it's trained on
character n-grams across hundreds of synthetic phrasings per
destination/object rather than a fixed keyword list.

A keyword safety-net still catches an explicit "stop"/"cancel"/"halt"
even in the rare case the model misses it — standard practice for a
safety-critical intent like stopping a moving robot.

Uses Google's free Web Speech API via SpeechRecognition (needs
internet). Falls back to typed console input if no microphone is found.
"""

import queue
import threading
import os

import joblib
import speech_recognition as sr

from house_data import DESTINATIONS, OBJECTS_META

MODEL_PATH = os.path.join(os.path.dirname(__file__), "voice_model.joblib")

_SAFETY_STOP_WORDS = ["stop", "cancel", "halt", "abort", "never mind", "forget it"]


class CommandParser:
    """Wraps the trained models. Raises a clear error if you forgot to
    run train_voice_model.py first."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"'{model_path}' not found. Run `python train_voice_model.py` first "
                f"to train the voice model."
            )
        bundle = joblib.load(model_path)
        self.intent_model = bundle["intent_model"]
        self.target_model = bundle["target_model"]

    def parse(self, text):
        """Returns a dict describing the command:
        {'intent': 'GOTO'|'FETCH'|'STOP', 'target_key': str|None,
         'label': str|None, 'raw': text}
        For GOTO, target_key is a WAYPOINTS key (e.g. 'bed', 'hall_sofa').
        For FETCH, target_key is an OBJECTS_META key (e.g. 'cup', 'glass')."""
        text_lower = text.lower().strip()

        # Safety-net: an explicit stop word always wins, regardless of
        # what the trained classifier predicts.
        if any(w in text_lower for w in _SAFETY_STOP_WORDS):
            return {"intent": "STOP", "target_key": None, "label": None, "raw": text}

        intent = self.intent_model.predict([text_lower])[0]

        if intent == "STOP":
            return {"intent": "STOP", "target_key": None, "label": None, "raw": text}

        target_key = self.target_model.predict([text_lower])[0]

        if intent == "GOTO":
            label = target_key.replace("_", " ")
        else:  # FETCH
            label = OBJECTS_META.get(target_key, {}).get("label", target_key)

        return {"intent": intent, "target_key": target_key, "label": label, "raw": text}


class VoiceListener:
    """Runs speech recognition on a background thread so the PyBullet
    main loop never blocks waiting on the microphone."""

    def __init__(self, energy_threshold=300, pause_threshold=0.8):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True  # self-adjusts while listening, no upfront pause needed
        self.text_queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._thread = None
        self.parser = CommandParser()

    def start(self):
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[voice] Listening for commands... "
              "(try 'go to the bed', 'pick up the glass and bring it to me', 'go to room 2')")

    def stop(self):
        self._stop_flag.set()

    def _listen_loop(self):
        try:
            mic = sr.Microphone()
        except Exception as e:
            print(f"[voice] No microphone available: {e}")
            print("[voice] Falling back to text input via console (type commands + Enter).")
            self._text_input_loop()
            return

        # No fixed ambient-noise calibration pause — dynamic_energy_threshold
        # (set in __init__) lets the recognizer continuously self-adjust to
        # background noise while actually listening, instead of a slow
        # upfront delay before it can respond at all. Worth the small
        # accuracy trade-off for a live demo where every second of "not
        # listening yet" costs you.


        while not self._stop_flag.is_set():
            try:
                with mic as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=6)
                text = self.recognizer.recognize_google(audio)
                print(f"[voice] Heard: \"{text}\"")
                self.text_queue.put(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[voice] Speech service error: {e}. Check your internet connection.")
            except Exception as e:
                print(f"[voice] Listener error: {e}")

    def _text_input_loop(self):
        while not self._stop_flag.is_set():
            try:
                text = input("[type a command] > ")
                if text.strip():
                    self.text_queue.put(text)
            except EOFError:
                break

    def get_command_nonblocking(self):
        try:
            text = self.text_queue.get_nowait()
            return self.parser.parse(text)
        except queue.Empty:
            return None
