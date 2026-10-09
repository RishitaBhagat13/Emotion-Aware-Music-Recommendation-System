import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# LOAD MODEL
# ==========================

model = load_model(
    r"models\rafdb_emotion_model.keras"
)

# ==========================
# EMOTION LABELS
# ==========================

emotion_map = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

# ==========================
# FACE DETECTOR
# ==========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

print("Starting Emotion Detection...")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        try:

            face = cv2.resize(
                face,
                (224, 224)
            )

            face = np.array(
                face,
                dtype=np.float32
            )

            face = preprocess_input(face)

            face = np.expand_dims(
                face,
                axis=0
            )

            prediction = model.predict(
                face,
                verbose=0
            )

            emotion_id = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            )

            emotion = emotion_map[
                emotion_id
            ]

            label = (
                f"{emotion} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        except Exception:
            pass

    cv2.imshow(
        "RAF-DB Emotion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()