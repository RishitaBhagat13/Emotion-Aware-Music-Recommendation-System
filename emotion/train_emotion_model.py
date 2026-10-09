import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================
# PATHS
# =====================================

TRAIN_DIR = r"emotion_dataset\FER_2013_emodataset\train"

TEST_DIR = r"emotion_dataset\FER_2013_emodataset\test"

MODEL_PATH = r"models\emotion_model.keras"

# =====================================
# SETTINGS
# =====================================

IMG_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 30

# =====================================
# DATA GENERATORS
# =====================================

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

test_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =====================================
# CLASS INFO
# =====================================

print("\nClasses:")

print(train_data.class_indices)

# =====================================
# MODEL
# =====================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

outputs = Dense(
    7,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=outputs
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# CALLBACKS
# =====================================

callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =====================================
# TRAIN
# =====================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================
# TEST
# =====================================

loss, acc = model.evaluate(
    test_data
)

print("\n===================")
print("Emotion Accuracy:", acc)
print("===================\n")

# =====================================
# SAVE
# =====================================

model.save(
    MODEL_PATH
)

print("Model saved!")

print(MODEL_PATH)