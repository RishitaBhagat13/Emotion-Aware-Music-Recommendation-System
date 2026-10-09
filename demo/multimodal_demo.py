import cv2
import json
import numpy as np
import mediapipe as mp

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================
# LOAD SIGN MODEL
# =====================================

sign_model = load_model(
    r"trained_modal\final_motion_attention_model.keras"
)

# =====================================
# LOAD EMOTION MODEL
# =====================================

emotion_model = load_model(
    r"models\rafdb_emotion_model.keras"
)

# =====================================
# LOAD SIGN NAMES
# =====================================

with open(
    r"training\nslt100_sign_names.json",
    "r"
) as f:

    sign_names = json.load(f)

# =====================================
# EMOTION LABELS
# =====================================

emotion_map = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

# =====================================
# FACE DETECTOR
# =====================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =====================================
# MEDIAPIPE HAND LANDMARKER
# =====================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = r"models\hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(
    options
)

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(result):

    features = np.zeros(
        126,
        dtype=np.float32
    )

    idx = 0

    if result.hand_landmarks:

        for hand in result.hand_landmarks[:2]:

            for lm in hand:

                if idx < 126:

                    features[idx] = lm.x
                    idx += 1

                    features[idx] = lm.y
                    idx += 1

                    features[idx] = lm.z
                    idx += 1

    return features

# =====================================
# NORMALIZATION
# =====================================

def normalize_frame(frame):

    frame = frame.copy()

    if np.sum(frame[:63]) != 0:

        wrist_x = frame[0]
        wrist_y = frame[1]
        wrist_z = frame[2]

        for i in range(0, 63, 3):

            frame[i] -= wrist_x
            frame[i + 1] -= wrist_y
            frame[i + 2] -= wrist_z

    if np.sum(frame[63:]) != 0:

        wrist_x = frame[63]
        wrist_y = frame[64]
        wrist_z = frame[65]

        for i in range(63, 126, 3):

            frame[i] -= wrist_x
            frame[i + 1] -= wrist_y
            frame[i + 2] -= wrist_z

    return frame

# =====================================
# FUSION LOGIC
# =====================================

def generate_message(sign, emotion):

    sign = sign.lower()
    emotion = emotion.lower()

    if sign == "help" and emotion == "sad":
        return "User may be requesting help and appears sad."

    elif sign == "help" and emotion == "angry":
        return "User may urgently need assistance."

    elif sign == "help":
        return "User is requesting help."

    elif sign == "yes":
        return "User is expressing agreement."

    elif sign == "no":
        return "User is expressing disagreement."

    elif sign == "drink":
        return "User may want a drink."

    elif sign == "book":
        return "User is referring to a book."

    return f"Sign={sign}, Emotion={emotion}"

# =====================================
# VARIABLES
# =====================================

sequence = []

prev_frame = np.zeros(
    126,
    dtype=np.float32
)

current_sign = "Waiting..."
current_conf = 0.0

current_emotion = "Neutral"
emotion_conf = 0.0

# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

print("Multimodal Demo Started")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =====================================
    # EMOTION DETECTION
    # =====================================

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

        try:

            face = frame[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (224, 224)
            )

            face = preprocess_input(
                face.astype(np.float32)
            )

            face = np.expand_dims(
                face,
                axis=0
            )

            prediction = emotion_model.predict(
                face,
                verbose=0
            )[0]

            emotion_id = np.argmax(
                prediction
            )

            emotion_conf = float(
                np.max(prediction)
            )

            current_emotion = emotion_map[
                emotion_id
            ]

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

        except Exception:
            pass

    # =====================================
    # SIGN DETECTION
    # =====================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape

            for lm in hand_landmarks:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    frame,
                    (cx, cy),
                    3,
                    (0, 255, 0),
                    -1
                )

    current_frame = extract_features(
        result
    )

    current_frame = normalize_frame(
        current_frame
    )

    velocity = current_frame - prev_frame

    combined = np.concatenate(
        [current_frame, velocity]
    )

    prev_frame = current_frame

    sequence.append(combined)

    if len(sequence) > 30:
        sequence.pop(0)

    if len(sequence) == 30:

        x = np.array(
            sequence,
            dtype=np.float32
        )

        x = np.expand_dims(
            x,
            axis=0
        )

        prediction = sign_model.predict(
            x,
            verbose=0
        )[0]

        pred_class = int(
            np.argmax(prediction)
        )

        current_conf = float(
            np.max(prediction)
        )

        if current_conf > 0.60:

            current_sign = sign_names[
                str(pred_class)
            ]

    # =====================================
    # FUSION MESSAGE
    # =====================================

    message = generate_message(
        current_sign,
        current_emotion
    )

    # =====================================
    # DISPLAY
    # =====================================

    cv2.putText(
        frame,
        f"Sign: {current_sign}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sign Conf: {current_conf:.2f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Emotion: {current_emotion}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Emotion Conf: {emotion_conf:.2f}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        message,
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Multimodal Deaf Care System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()