"""
==================================================
  SIGN LANGUAGE PREDICTOR - STEP 2: TRAIN MODEL
==================================================
Loads landmark data, trains a Random Forest pipeline
with StandardScaler, evaluates with cross-validation,
and saves confusion matrix plot + trained model.

HOW TO USE:
  python train_model.py

Requires dataset/landmarks.pkl from collect_data.py
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

# --- Config ---
DATASET_PATH = "dataset/landmarks.pkl"
MODEL_OUT_PATH = "dataset/model.pkl"
CONFUSION_MATRIX_PATH = "dataset/confusion_matrix.png"

def train():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}. Run collect_data.py first!")
        return

    # Load data
    with open(DATASET_PATH, "rb") as f:
        data_dict = pickle.load(f)
    
    X = np.array(data_dict["data"])
    labels = np.array(data_dict["labels"])

    # Encode text labels (A, B, C) to numbers
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # Build pipeline with StandardScaler + RandomForest (200 trees)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Training complete! Test Accuracy: {acc:.2%}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"📈 5-Fold CV: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Classification report
    print(f"\n📋 Per-class metrics:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Sign Language - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"📉 Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # Save model and encoder
    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "accuracy": acc}, f)
    
    print(f"💾 Model saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    train()