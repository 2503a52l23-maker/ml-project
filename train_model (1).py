"""
==================================================
  SIGN LANGUAGE PREDICTOR - STEP 2: TRAIN MODEL
==================================================
Trains a Random Forest classifier on the collected
hand landmark data and saves the model to disk.

HOW TO USE:
  python train_model.py

Make sure you have run collect_data.py first.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler

# ─── Config ─────────────────────────────────────────────────
DATASET_PATH = "dataset/landmarks.pkl"
MODEL_PATH   = "dataset/model.pkl"

# ─── Load Dataset ───────────────────────────────────────────
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"❌  Dataset not found at '{DATASET_PATH}'")
        print("    Please run  collect_data.py  first.")
        exit(1)

    with open(DATASET_PATH, "rb") as f:
        raw = pickle.load(f)

    X = np.array(raw["data"])
    y = np.array(raw["labels"])

    print(f"\n📦  Dataset loaded")
    print(f"    Samples  : {len(X)}")
    print(f"    Features : {X.shape[1]} (hand landmarks × 3 axes)")
    print(f"    Classes  : {len(set(y))}  →  {sorted(set(y))}")
    return X, y

# ─── Train ──────────────────────────────────────────────────
def train(X, y):
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\n🏋️  Training Random Forest classifier...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc    = accuracy_score(y_test, y_pred)
    roc_macro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    roc_weighted = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

    print(f"\n{'='*55}")
    print(f"  ✅  Test Accuracy : {acc * 100:.2f}%")
    print(f"  📈  ROC AUC Macro : {roc_macro:.3f}")
    print(f"  📈  ROC AUC Weighted : {roc_weighted:.3f}")
    print(f"{'='*55}")
    print("\n📊  Per-class Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Cross-validation ──
    cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"📈  5-Fold Cross-Val Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ── Confusion matrix plot ──
    _plot_confusion(y_test, y_pred, le.classes_)

    # ── Feature Importance ──
    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[-20:]
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [f'Landmark {indices[i]}' for i in range(len(indices))])
    plt.xlabel('Importance')
    plt.title('Top 20 Landmark Feature Importances')
    plt.tight_layout()
    imp_path = "dataset/feature_importances.png"
    plt.savefig(imp_path, dpi=150)
    print(f"📊  Feature importances saved → {imp_path}")
    plt.show()

    return model, le, acc

# ─── Save ───────────────────────────────────────────────────
def save_model(model, le, acc):
    bundle = {"model": model, "label_encoder": le, "accuracy": acc}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n💾  Model saved  →  {MODEL_PATH}")
    print(f"\n➡️  Next step: run  python predict.py")

# ─── Confusion Matrix ───────────────────────────────────────
def _plot_confusion(y_true, y_pred, class_names):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plot_path = "dataset/confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    print(f"📉  Confusion matrix saved  →  {plot_path}")
    plt.show()

# ─── Entry ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  SIGN LANGUAGE MODEL TRAINER")
    print("="*55)

    X, y           = load_dataset()
    model, le, acc = train(X, y)
    save_model(model, le, acc)
