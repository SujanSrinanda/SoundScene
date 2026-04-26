# Project Report: SoundScene – Context-Aware Visual Sound Alert System

## 1. Problem Statement
Many individuals, particularly the hearing-impaired or those wearing noise-canceling headphones, miss critical environmental audio cues (e.g., doorbells, alarms, crying babies). Existing sound detection systems often lack context—they simply translate a sound to text without assessing the situation. SoundScene bridges this gap by combining an offline-first Machine Learning sound classifier with a rule-based Context Engine to provide immediate, context-aware visual alerts prioritizing user safety and awareness.

## 2. System Architecture
SoundScene employs a decoupled, offline-first architecture:
- **Backend (Python FastAPI):** Serves as the core logic hub. It handles API requests, interfaces with the ML model for inference, and processes the output through the Context Engine.
- **Frontend (Vanilla HTML/CSS/JS):** A lightweight, responsive UI that runs directly in the browser. It communicates with the backend via REST APIs to provide real-time visual situation cards.
- **ML Pipeline (TensorFlow Lite):** A lightweight Convolutional Neural Network (CNN) optimized for local inference.

## 3. Machine Learning Model Explanation
The core model is a lightweight Convolutional Neural Network (CNN) designed to classify environmental sounds. 
- **Architecture:** `Conv2D -> ReLU -> MaxPooling -> Conv2D -> ReLU -> MaxPooling -> Dense -> Softmax`
- **Training Dataset:** A curated subset of UrbanSound8K / ESC-50, focusing on relevant classes (doorbell, dog bark, smoke alarm, car horn, baby cry, knock).
- **Loss & Optimizer:** Trained using Categorical Crossentropy and the Adam optimizer for efficient convergence.

## 4. Feature Extraction: MFCC & 16kHz Sampling
Raw audio waveforms are complex and high-dimensional. We extract Mel-Frequency Cepstral Coefficients (MFCCs) using `librosa`.
- **Why MFCCs?** They represent the short-term power spectrum of a sound, mirroring human auditory perception, making them ideal for identifying distinct environmental events. We extract 13 coefficients per frame.
- **Why 16 kHz Mono?** Downsampling audio to 16 kHz significantly reduces computational overhead while preserving the critical frequency bands needed to identify common environmental sounds. It ensures the model remains lightweight and capable of real-time (<100ms) inference on edge devices.

## 5. Context Engine Logic
The Context Engine elevates simple classification to situational awareness. It ingests the sound label, confidence score, time of day, location, and repetition count to determine the UI state.
- **Rules Mapping:** 
  - *Smoke Alarm:* Always mapped to "Critical" (Red) urgency.
  - *Knock at Night:* Escalated from "Normal" (Green) to "Warning" (Yellow).
  - *Repeated Sounds:* If a sound (e.g., dog barking) occurs repeatedly, the engine escalates its urgency level.
- **Output:** Returns a human-readable description, color-coded urgency level, and a recommended action.

## 6. UI Design (Situation Cards)
The frontend relies on dynamic **Situation Cards** designed for immediate visual comprehension:
- **Accessibility:** Uses distinct colors (Green, Yellow, Red), glowing shadows, and large emojis/SVG icons so the alert level can be understood instantly.
- **Clean Aesthetics:** Implements a modern dark theme with soft glassmorphism effects, avoiding clutter and prioritizing the actionable information.
- **Animations:** Smooth transitions prevent jarring visual changes, offering a premium user experience.

## 7. Limitations and Solutions
- **Limitation:** Background noise can reduce model accuracy.
  - *Solution:* Implement spectral gating or noise reduction as a preprocessing step before MFCC extraction.
- **Limitation:** Running deep learning models in the browser can be resource-intensive.
  - *Solution:* The system uses an optimized TensorFlow Lite model running on the local backend, keeping the frontend extremely lightweight.

## 8. Future Scope
- **Continuous Learning:** Allow users to flag false positives to fine-tune the model locally over time.
- **Hardware Integration:** Deploy the system on edge devices like Raspberry Pi with attached LED light strips for physical visual alerts.
- **Wearable Extension:** Push critical context-aware alerts directly to smartwatches (e.g., Apple Watch, Wear OS) via Bluetooth.
