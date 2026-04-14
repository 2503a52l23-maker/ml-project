"""
==================================================
  SIGN LANGUAGE PREDICTOR - STEP 1: COLLECT DATA
==================================================
This script opens your camera and helps you collect
hand landmark data for each sign language letter.

HOW TO USE:
  python collect_data.py

- Press the KEY shown on screen to start collecting for that letter
- Hold your sign clearly in front of the camera
- Press 'q' to quit at any time
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# ─── Configuration ──────────────────────────────────────────
DATASET_DIR     = "dataset"
SAMPLES_PER_CLASS = 100          # how many samples to collect per sign
SIGNS           = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["HELLO", "THANKS", "YES", "NO", "PLEASE"]

# ─── MediaPipe Setup ────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils

# ─── Helpers ────────────────────────────────────────────────
def extract_landmarks(hand_landmarks):
    """Flatten 21 hand landmarks (x, y, z) into a 63-element vector."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def normalize_landmarks(coords):
    """Normalize landmarks relative to wrist so position doesn't matter."""
    coords = np.array(coords).reshape(21, 3)
    wrist   = coords[0]
    coords -= wrist                        # shift to wrist origin
    scale   = np.max(np.abs(coords)) + 1e-6
    coords /= scale                        # scale to [-1, 1]
    return coords.flatten().tolist()

# ─── Main Collection Loop ────────────────────────────────────
def collect_data():
    os.makedirs(DATASET_DIR, exist_ok=True)

    data   = []
    labels = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open camera. Check your camera connection.")
        return

    print("\n" + "="*55)
    print("  SIGN LANGUAGE DATA COLLECTOR")
    print("="*55)
    print(f"  Signs to collect : {', '.join(SIGNS)}")
    print(f"  Samples per sign : {SAMPLES_PER_CLASS}")
    print("="*55)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        for sign in SIGNS:
            print(f"\n🔵  Get ready to show sign: '{sign}'")
            print(f"    Press ENTER in the terminal when ready...")
            input()

            collected = 0

            while collected < SAMPLES_PER_CLASS:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                # Draw landmarks
                if result.multi_hand_landmarks:
                    for hl in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                    # Extract & save
                    raw  = extract_landmarks(result.multi_hand_landmarks[0])
                    norm = normalize_landmarks(raw)
                    data.append(norm)
                    labels.append(sign)
                    collected += 1

                # HUD
                pct    = int((collected / SAMPLES_PER_CLASS) * 100)
                bar    = "█" * (pct // 5) + "░" * (20 - pct // 5)
                status = f"Sign: {sign} | [{bar}] {pct}% ({collected}/{SAMPLES_PER_CLASS})"
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Data Collector - Press Q to quit", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Collection stopped early.")
                    cap.release()
                    cv2.destroyAllWindows()
                    _save(data, labels)
                    return

            print(f"  ✅  Collected {collected} samples for '{sign}'")

    cap.release()
    cv2.destroyAllWindows()
    _save(data, labels)


def _save(data, labels):
    path = os.path.join(DATASET_DIR, "landmarks.pkl")
    with open(path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    print(f"\n💾  Dataset saved  →  {path}")
    print(f"    Total samples  :  {len(data)}")
    unique = set(labels)
    print(f"    Classes        :  {len(unique)} ({', '.join(sorted(unique))})")
    print("\n➡️  Next step: run  python train_model.py")


if __name__ == "__main__":
    collect_data()