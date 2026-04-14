import os
import pickle
import numpy as np

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["HELLO", "THANKS", "YES", "NO", "PLEASE"]
data = []
labels = []

np.random.seed(42)
for sign in SIGNS:
    for i in range(20):  # 20 samples per class
        # Synthetic landmarks: 21*3=63 dims, normalized [-1,1], some class-specific patterns
        base_pattern = np.random.normal(0, 0.3, 63)
        if len(sign) > 1:
            base_pattern[::3] += 0.2  # word signs have distinct x-offset
        lm = np.clip(base_pattern, -1, 1).tolist()
        data.append(lm)
        labels.append(sign)

path = os.path.join(DATASET_DIR, "landmarks.pkl")
with open(path, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"Dummy dataset created: {path}")
print(f"Total samples: {len(data)}, Classes: {len(SIGNS)}")

