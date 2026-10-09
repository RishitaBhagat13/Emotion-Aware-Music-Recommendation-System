import os
import json
import numpy as np

# =====================================
# PATHS
# =====================================

JSON_FILE = r"datasets\nslt_100.json"

FEATURE_DIR = r"data_preprocessing\datasets\motion_features"

SAVE_X = r"training\X.npy"
SAVE_Y = r"training\y.npy"

# =====================================
# DEBUG
# =====================================

print("Current Directory:", os.getcwd())

print("JSON exists:",
      os.path.exists(JSON_FILE))

print("Feature dir exists:",
      os.path.exists(FEATURE_DIR))

if os.path.exists(FEATURE_DIR):
    print("Feature files:",
          len(os.listdir(FEATURE_DIR)))

# =====================================
# LOAD ANNOTATIONS
# =====================================

with open(JSON_FILE, "r") as f:
    annotations = json.load(f)

print("Total JSON entries:",
      len(annotations))

# =====================================
# BUILD DATASET
# =====================================

X = []
y = []

found = 0
missing = 0

for video_id, info in annotations.items():

    feature_file = os.path.join(
        FEATURE_DIR,
        f"{video_id}.npy"
    )

    if not os.path.exists(feature_file):
        missing += 1
        continue

    try:

        features = np.load(feature_file)

        # Motion features should be (30,252)

        if len(features.shape) != 2:
            print(
                f"Skipping {video_id}: "
                f"invalid shape {features.shape}"
            )
            continue

        if features.shape[0] != 30:
            print(
                f"Skipping {video_id}: "
                f"expected 30 frames, got {features.shape}"
            )
            continue

        label = info["action"][0]

        X.append(features)
        y.append(label)

        found += 1

    except Exception as e:

        print(
            f"Error loading {video_id}: {e}"
        )

# =====================================
# CONVERT TO NUMPY
# =====================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int32
)

# =====================================
# RESULTS
# =====================================

print("\n========== RESULTS ==========")

print("Found files:", found)

print("Missing files:", missing)

print("X shape:", X.shape)

print("y shape:", y.shape)

if len(y) > 0:

    print(
        "Unique classes:",
        len(set(y))
    )

    print(
        "Min label:",
        np.min(y)
    )

    print(
        "Max label:",
        np.max(y)
    )

# =====================================
# SAVE
# =====================================

np.save(
    SAVE_X,
    X
)

np.save(
    SAVE_Y,
    y
)

print("\nDataset saved!")

print("Saved:")
print(SAVE_X)
print(SAVE_Y)