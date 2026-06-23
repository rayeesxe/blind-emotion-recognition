"""
Main application for Blind Emotion Recognition.

This script captures video frames from the default webcam, detects emotions using a
pre‑trained model from the `fer` library and announces changes in the detected
emotion via a text‑to‑speech engine (`pyttsx3`).  The goal is to provide
real‑time emotional feedback for users who are visually impaired.

Usage:
    python3 main.py

Press 'q' in the video window to quit the program or use Ctrl+C in the terminal.

Note:
    The `fer` library uses the FER‑2013 dataset behind the scenes.  For more
    information about the model, see the accompanying README.md and research
    references.
"""
import time
import threading

import cv2  # type: ignore
import numpy as np  # type: ignore
from fer import FER  # type: ignore
import pyttsx3  # type: ignore


def speak_thread(engine: pyttsx3.Engine, text: str) -> None:
    """Speak the given text asynchronously using pyttsx3 in a separate thread.

    Parameters:
        engine: pyttsx3.Engine instance.
        text: The message to speak.
    """
    def _speak():
        engine.say(text)
        engine.runAndWait()

    t = threading.Thread(target=_speak, daemon=True)
    t.start()


def main() -> None:
    """Entry point of the blind emotion recognition application."""
    # Initialize the emotion detector and TTS engine
    detector = FER(mtcnn=True)  # use MTCNN for face detection accuracy
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)  # set speech rate slower for clarity

    # Open the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    last_emotion: str = ""
    last_spoken_time = 0.0
    cooldown_seconds = 2.0  # speak no more than once every 2 seconds to avoid repetition

    print("Starting emotion detection. Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            break

        # Convert frame to RGB for FER
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect emotions; returns list of dicts with bounding boxes and scores
        results = detector.detect_emotions(rgb_frame)

        # Determine the dominant emotion on the largest face
        dominant_emotion = None
        largest_area = 0
        for result in results:
            (x, y, w, h) = result["box"]
            area = w * h
            emotions = result["emotions"]
            # Get the emotion with highest probability
            emotion_label = max(emotions, key=emotions.get)
            if area > largest_area:
                largest_area = area
                dominant_emotion = emotion_label
            # Draw bounding box and label (optional visual feedback)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{emotion_label}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # If a face was detected and the dominant emotion changed, speak it
        current_time = time.time()
        if dominant_emotion and dominant_emotion != last_emotion:
            if current_time - last_spoken_time > cooldown_seconds:
                message = f"The person appears {dominant_emotion}"
                print(message)
                speak_thread(engine, message)
                last_emotion = dominant_emotion
                last_spoken_time = current_time

        # Show the frame for users with vision (can be disabled for blind users)
        cv2.imshow('Emotion Recognition', frame)
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
