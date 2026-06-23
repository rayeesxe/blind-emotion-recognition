# Blind Emotion Recognition

This project implements a **blind‑friendly facial emotion recognition** application using Python. The goal is to help visually impaired users understand the emotional state of people around them by providing **real‑time audio feedback**. The system captures frames from a webcam, detects facial expressions using a pre‑trained model and then uses a text‑to‑speech (TTS) engine to announce the detected emotion.

## Background

Individuals with visual impairment often struggle to interpret facial cues during social interactions. Academic research has proposed using convolutional neural networks and transfer learning to build facial emotion recognition systems for the visually impaired. For example, studies on facial emotion recognition using transfer learning highlight how using the FER‑2013 dataset and transfer learning can accurately detect emotions while providing audio output so that visually impaired users can understand social cues. Modern emotion recognition software relies on computer vision and machine learning techniques such as facial landmark detection and neural networks. Open‑source libraries like **OpenCV** and **TensorFlow** are common tools for building such systems.

## Features

* **Real‑time video capture:** captures frames from the default webcam.
* **Emotion detection:** uses the [`fer`](https://github.com/justinshenk/fer) library’s pre‑trained model (trained on the FER‑2013 dataset) to recognize emotions such as happy, sad, angry, fearful, surprised and neutral.
* **Audio feedback:** speaks the detected emotion aloud using the `pyttsx3` text‑to‑speech library, making the output accessible to blind users.
* **Minimal dependencies:** relies on Python libraries available via `pip`.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Clone this repository:

   git clone https://github.com/YOUR_USERNAME/blind-emotion-recognition.git
   cd blind-emotion-recognition

3. Install the required dependencies:

   pip install -r requirements.txt

   The key libraries include:

   - `opencv-python` for capturing video from the webcam.
   - `fer` for facial emotion recognition.
   - `pyttsx3` for text‑to‑speech.
   - `numpy` for array operations.

## Usage

Run the application from the command line:

   python3 main.py

By default, the program opens your primary webcam, detects faces in each frame and announces the dominant emotion whenever it changes. The live video feed will also display bounding boxes with labels on the screen if a monitor is available, but audio feedback makes the system usable without visual output.

To exit the program, press `q` while the video window is focused or use `Ctrl+C` in the terminal.

## Notes

* **Camera permissions:** The application requires access to a webcam. Make sure your system allows access.
* **Performance:** Real‑time emotion detection may be computationally intensive. Lower frame sizes or frame skipping may help on low‑power devices.
* **Bias and accuracy:** Emotion recognition models can exhibit bias and may perform differently across demographics. Use this system as an assistive tool, not a definitive predictor of emotional state.
* **Privacy:** Be mindful of privacy when capturing video and audio. Collect consent from individuals whose faces are being analyzed.

## License

This project is released under the MIT License.
