import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# =====================================
# PATHS
# =====================================

MODEL_PATH = r"models\rafdb_emotion_model.keras"
TEST_DIR = r"emotion_dataset\RAF-DB\DATASET\test"

IMG_SIZE = 224
BATCH_SIZE = 16

# =====================================
# LOAD MODEL
# =====================================

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# =====================================
# LOAD TEST DATA
# =====================================

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================
# SHOW CLASS MAPPING
# =====================================

print("\nClass Mapping:")
print(test_data.class_indices)

class_names = list(test_data.class_indices.keys())

# =====================================
# EVALUATE MODEL
# =====================================

loss, accuracy = model.evaluate(test_data, verbose=1)

print("\n==============================")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")
print("==============================")

# =====================================
# PREDICTIONS
# =====================================

test_data.reset()

predictions = model.predict(test_data, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix\n")
print(cm)

# =====================================
# PLOT CONFUSION MATRIX
# =====================================

fig, ax = plt.subplots(figsize=(9,8))

im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

plt.colorbar(im)

ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Predicted Emotion",
    ylabel="True Emotion",
    title="RAF-DB Confusion Matrix"
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor"
)

# Write numbers inside each cell

thresh = cm.max() / 2.0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=11
        )

plt.tight_layout()
plt.show()