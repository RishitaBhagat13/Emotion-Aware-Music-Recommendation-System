import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================
# LOAD MODEL
# =====================================

MODEL_PATH = r"C:\Users\BIT\Desktop\Deaf_care_Proj\trained_modal\final_motion_attention_model.keras"

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# =====================================
# LOAD DATA (Memory Mapping)
# =====================================

X = np.load("training/X.npy", mmap_mode="r")
y = np.load("training/y.npy", mmap_mode="r")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# =====================================
# LABEL ENCODING
# =====================================

classes = sorted(list(set(y)))

label_map = {c: i for i, c in enumerate(classes)}

y_encoded = np.array([label_map[c] for c in y])

y_cat = to_categorical(
    y_encoded,
    num_classes=len(classes)
)

# =====================================
# SAME TRAIN/TEST SPLIT AS TRAINING
# =====================================

_, X_test, _, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Convert memmap slice to ndarray
X_test = np.asarray(X_test)

# =====================================
# EVALUATE
# =====================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==========================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Loss     : {loss:.4f}")
print("==========================")

# =====================================
# PREDICTIONS
# =====================================

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report\n")

print(classification_report(
    y_true,
    y_pred,
    digits=4
))

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix\n")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Sign Language Confusion Matrix")
plt.show()