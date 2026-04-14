"""
==================================================
  SIGN LANGUAGE PREDICTOR - STEP 3: PREDICT
==================================================
Opens your camera, detects hand signs in real-time
and prints the predicted text to terminal.

HOW TO USE:
  python predict.py

CONTROLS (while camera window is open):
  SPACE  →  Confirm current prediction (add to sentence)
  ENTER  →  Print full sentence to terminal & clear
  C      →  Clear last character
  R      →  Reset sentence
  Q      →  Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys
import time
from collections import deque, Counter

# ─── Config ─────────────────────────────────────────────────
MODEL_PATH       = "dataset/model.pkl"
CONFIDENCE_THRESH = 0.65      # minimum confidence to show prediction
SMOOTHING_FRAMES  = 15        # how many frames to average for stability
HOLD_SECONDS      = 1.5       # seconds to hold sign before auto-confirm

# ─── Colors (BGR) ───────────────────────────────────────────
GREEN  = (0, 220, 80)
RED    = (0, 60, 220)
BLUE   = (220, 100, 0)
YELLOW = (0, 220, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (60, 60, 60)

# ─── Load Model ─────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌  Model not found at '{MODEL_PATH}'")
        print("    Please run  train_model.py  first.\n")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    le    = bundle["label_encoder"]
    acc   = bundle.get("accuracy", 0)
    print(f"\n✅  Model loaded  (trained accuracy: {acc*100:.1f}%)")
    print(f"    Signs supported: {', '.join(le.classes_)}\n")
    return model, le

# ─── Landmark Helpers ────────────────────────────────────────
def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def normalize_landmarks(coords):
    coords = np.array(coords).reshape(21, 3)
    wrist  = coords[0]
    coords -= wrist
    scale  = np.max(np.abs(coords)) + 1e-6
    coords /= scale
    return coords.flatten()

# ─── Draw Helpers ────────────────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0,
                    {(x1+r,y1+r):180, (x2-r,y1+r):270, (x1+r,y2-r):90, (x2-r,y2-r):0}[(cx,cy)],
                    {(x1+r,y1+r):270, (x2-r,y1+r):360, (x1+r,y2-r):180, (x2-r,y2-r):90}[(cx,cy)],
                    color, thickness)

def draw_hud(frame, prediction, confidence, sentence, hold_ratio, fps):
    h, w = frame.shape[:2]

    # ── Top bar ──
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.putText(frame, "SIGN LANGUAGE PREDICTOR", (10, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

    # ── Prediction box ──
    box_y = h - 220
    cv2.rectangle(frame, (0, box_y), (w, h), (15, 15, 15), -1)

    if prediction:
        color = GREEN if confidence >= CONFIDENCE_THRESH else YELLOW
        conf_pct = int(confidence * 100)
        bar_w    = int((w - 40) * confidence)

        cv2.putText(frame, "DETECTED:", (15, box_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        cv2.putText(frame, prediction, (15, box_y + 80),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, color, 3)
        cv2.putText(frame, f"{conf_pct}%", (w - 95, box_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Confidence bar
        cv2.rectangle(frame, (15, box_y + 95), (w - 15, box_y + 110), GRAY, -1)
        cv2.rectangle(frame, (15, box_y + 95), (15 + bar_w, box_y + 110), color, -1)

        # Hold progress bar
        if hold_ratio > 0:
            hold_w = int((w - 40) * hold_ratio)
            cv2.rectangle(frame, (15, box_y + 118), (w - 15, box_y + 130), GRAY, -1)
            cv2.rectangle(frame, (15, box_y + 118), (15 + hold_w, box_y + 130), BLUE, -1)
            cv2.putText(frame, "Hold...", (15, box_y + 148),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
    else:
        cv2.putText(frame, "No hand detected", (15, box_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)

    # ── Sentence bar ──
    cv2.rectangle(frame, (0, box_y + 155), (w, h), (10, 10, 40), -1)
    disp = sentence if sentence else "_"
    cv2.putText(frame, f"Sentence: {disp}", (15, box_y + 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)

    # ── Controls bar ──
    ctrl = "[SPACE] Confirm   [ENTER] Print   [C] Delete   [R] Reset   [Q] Quit"
    cv2.putText(frame, ctrl, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

# ─── Main Predict Loop ───────────────────────────────────────
def predict():
    model, le = load_model()

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)
    sentence          = []
    hold_start        = None
    prev_pred         = None
    fps_timer         = time.time()
    fps               = 0
    frame_count       = 0

    print("\n" + "="*55)
    print("  📷  Camera is open. Show your hand sign!")
    print("="*55)
    print("  SPACE  → Add current sign to sentence")
    print("  ENTER  → Print sentence & clear")
    print("  C      → Delete last character")
    print("  R      → Reset sentence")
    print("  Q      → Quit")
    print("="*55 + "\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame      = cv2.flip(frame, 1)
            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result     = hands.process(rgb)

            # FPS
            frame_count += 1
            if frame_count % 15 == 0:
                fps       = 15 / (time.time() - fps_timer)
                fps_timer = time.time()

            prediction = None
            confidence = 0.0
            hold_ratio = 0.0

            if result.multi_hand_landmarks:
                hl   = result.multi_hand_landmarks[0]
                raw  = extract_landmarks(hl)
                feat = normalize_landmarks(raw).reshape(1, -1)

                probs      = model.predict_proba(feat)[0]
                idx        = np.argmax(probs)
                confidence = probs[idx]
                label      = le.classes_[idx]

                if confidence >= CONFIDENCE_THRESH:
                    prediction_buffer.append(label)
                else:
                    prediction_buffer.append(None)

                if prediction_buffer:
                    counts  = Counter(x for x in prediction_buffer if x is not None)
                    if counts:
                        prediction = counts.most_common(1)[0][0]

                # Auto-hold logic
                if prediction == prev_pred and prediction is not None:
                    if hold_start is None:
                        hold_start = time.time()
                    elapsed    = time.time() - hold_start
                    hold_ratio = min(elapsed / HOLD_SECONDS, 1.0)

                    if elapsed >= HOLD_SECONDS:
                        sentence.append(prediction)
                        hold_start = None
                        prev_pred  = None
                        prediction_buffer.clear()
                        print(f"  ✋  Auto-confirmed: '{prediction}'  →  Sentence: {''.join(sentence)}")
                else:
                    hold_start = None
                    prev_pred  = prediction

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=GREEN, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=WHITE, thickness=2)
                )
            else:
                prediction_buffer.clear()
                hold_start = None
                prev_pred  = None

            draw_hud(frame, prediction, confidence, " ".join(sentence), hold_ratio, fps)
            cv2.imshow("Sign Language Predictor", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n👋  Quitting. Goodbye!")
                break

            elif key == 32:  # SPACE — manual confirm
                if prediction:
                    sentence.append(prediction)
                    hold_start = None
                    prediction_buffer.clear()
                    print(f"  ✋  Confirmed: '{prediction}'  →  Sentence: {''.join(sentence)}")

            elif key == 13:  # ENTER — print sentence
                full = " ".join(sentence)
                print("\n" + "="*55)
                print(f"  📝  SENTENCE: {full}")
                print("="*55 + "\n")
                sentence.clear()

            elif key == ord('c'):  # delete last
                if sentence:
                    removed = sentence.pop()
                    print(f"  ⌫  Removed '{removed}'  →  Sentence: {''.join(sentence)}")

            elif key == ord('r'):  # reset
                sentence.clear()
                print("  🔄  Sentence reset.")

    cap.release()
    cv2.destroyAllWindows()

# ─── Entry ──────────────────────────────────────────────────
if __name__ == "__main__":
    predict()