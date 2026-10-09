import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D,
    LayerNormalization
)

from tensorflow.keras.models import Model

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================
# LOAD DATA
# =====================================

X = np.load("training/X.npy")
y = np.load("training/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# =====================================
# LABEL ENCODING
# =====================================

classes = sorted(list(set(y)))

label_map = {
    c: i
    for i, c in enumerate(classes)
}

y_encoded = np.array(
    [label_map[c] for c in y]
)

num_classes = len(classes)

y_cat = to_categorical(
    y_encoded,
    num_classes
)

print("Classes:", num_classes)

# =====================================
# SPLIT
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

inputs = Input(
    shape=(30, 252)
)

x = Bidirectional(
    LSTM(
        128,
        return_sequences=True
    )
)(inputs)

x = Dropout(0.3)(x)

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(x)

x = LayerNormalization()(x)

attention = Attention()([x, x])

x = GlobalAveragePooling1D()(attention)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

x = Dense(
    128,
    activation="relu"
)(x)

outputs = Dense(
    num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs
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
        patience=12,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    ),

    ModelCheckpoint(
        "best_motion_attention_model.keras",
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

print("\nFinal Accuracy:", acc)

model.save(
    "final_motion_attention_model.keras"
)

print("Model Saved!")