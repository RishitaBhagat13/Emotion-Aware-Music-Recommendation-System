import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from tensorflow.keras.models import Model

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ======================
# LOAD DATA
# ======================

X = np.load("training/X.npy")

y = np.load("training/y.npy")

classes = sorted(list(set(y)))

label_map = {
    c:i
    for i,c in enumerate(classes)
}

y = np.array(
    [label_map[c] for c in y]
)

num_classes = len(classes)

y = to_categorical(
    y,
    num_classes
)

# ======================
# SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================
# MODEL
# ======================

inputs = Input(
    shape=(30,126)
)

x = Bidirectional(
    LSTM(
        128,
        return_sequences=True
    )
)(inputs)

x = Dropout(0.3)(x)

attention = Attention()(
    [x,x]
)

x = GlobalAveragePooling1D()(
    attention
)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

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

callbacks = [

    EarlyStopping(
        patience=10,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "best_attention_model.keras",
        save_best_only=True
    )
]

model.fit(

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

loss, acc = model.evaluate(
    X_test,
    y_test
)

print("Accuracy:", acc)