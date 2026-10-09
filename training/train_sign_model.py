import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# =====================================
# LOAD DATA
# =====================================

X = np.load("training/X.npy")
y = np.load("training/y.npy")

print("X:", X.shape)
print("y:", y.shape)

# =====================================
# LABEL ENCODING
# =====================================

unique_classes = sorted(list(set(y)))

label_map = {
    label: idx
    for idx, label in enumerate(unique_classes)
}

y_encoded = np.array(
    [label_map[label] for label in y]
)

num_classes = len(unique_classes)

y_cat = to_categorical(
    y_encoded,
    num_classes=num_classes
)

print("Classes:", num_classes)

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =====================================
# MODEL
# =====================================

model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        ),
        input_shape=(30, 126)
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Bidirectional(
        LSTM(
            64
        )
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dense(
        num_classes,
        activation="softmax"
    )
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
        patience=10,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "best_sign_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =====================================
# TRAIN
# =====================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=100,

    batch_size=32,

    callbacks=callbacks
)

# =====================================
# EVALUATE
# =====================================

loss, acc = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", acc)

model.save(
    "final_sign_model.keras"
)

print(
    "Model Saved!"
)