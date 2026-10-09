import json
import numpy as np

from tensorflow.keras.models import load_model

# ==========================
# LOAD MODEL
# ==========================

model = load_model(
    r"trained_modal\final_motion_attention_model.keras"
)

# ==========================
# LOAD DATA
# ==========================

X = np.load(
    r"training\X.npy"
)

# ==========================
# LOAD SIGN NAMES
# ==========================

with open(
    r"training\nslt100_sign_names.json",
    "r"
) as f:

    sign_names = json.load(f)

# ==========================
# TEST SAMPLE
# ==========================

sample = X[0]

prediction = model.predict(
    np.expand_dims(sample, axis=0),
    verbose=0
)

pred_class = np.argmax(prediction)

print("\nPrediction ID:",
      pred_class)

print(
    "Sign:",
    sign_names[str(pred_class)]
)

print(
    "Confidence:",
    np.max(prediction)
)