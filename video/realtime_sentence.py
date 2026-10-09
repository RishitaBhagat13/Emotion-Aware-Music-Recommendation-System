import cv2
import json
import numpy as np
import mediapipe as mp

from collections import deque, Counter
from tensorflow.keras.models import load_model

from sentence_builder import SentenceBuilder


# ===========================================================
# LOAD MODEL
# ===========================================================

model = load_model(
    r"trained_modal\final_motion_attention_model.keras"
)


# ===========================================================
# LOAD SIGN NAMES
# ===========================================================

with open(
    r"training\nslt100_sign_names.json",
    "r"
) as f:

    sign_names = json.load(f)


# ===========================================================
# MEDIAPIPE
# ===========================================================

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


# ===========================================================
# FEATURE EXTRACTION
# ===========================================================

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


# ===========================================================
# NORMALIZATION
# ===========================================================

def normalize_frame(frame):

    frame = frame.copy()

    # LEFT HAND

    if np.sum(frame[:63]) != 0:

        wrist_x = frame[0]
        wrist_y = frame[1]
        wrist_z = frame[2]

        for i in range(0,63,3):

            frame[i] -= wrist_x
            frame[i+1] -= wrist_y
            frame[i+2] -= wrist_z

    # RIGHT HAND

    if np.sum(frame[63:]) != 0:

        wrist_x = frame[63]
        wrist_y = frame[64]
        wrist_z = frame[65]

        for i in range(63,126,3):

            frame[i] -= wrist_x
            frame[i+1] -= wrist_y
            frame[i+2] -= wrist_z

    return frame


# ===========================================================
# VARIABLES
# ===========================================================

sequence = []

prev_frame = np.zeros(
    126,
    dtype=np.float32
)

builder = SentenceBuilder()

prediction_window = deque(maxlen=10)

current_sign = "Waiting..."
current_conf = 0.0

stable_word = ""
stable_count = 0

last_added_word = ""

CONFIDENCE_THRESHOLD = 0.60

STABLE_THRESHOLD = 8


# ===========================================================
# CAMERA
# ===========================================================

cap = cv2.VideoCapture(0)

print("Realtime Sentence Recognition Started")
print("Press C to clear sentence")
print("Press Q to quit")

# ===========================================================
# MAIN LOOP
# ===========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    # -----------------------------------------
    # Draw Hand Landmarks
    # -----------------------------------------

    if result.hand_landmarks:

        h, w, _ = frame.shape

        for hand in result.hand_landmarks:

            for lm in hand:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    frame,
                    (cx, cy),
                    3,
                    (0,255,0),
                    -1
                )

    # -----------------------------------------
    # Feature Extraction
    # -----------------------------------------

    current_frame = extract_features(result)

    current_frame = normalize_frame(current_frame)

    velocity = current_frame - prev_frame

    combined = np.concatenate(
        [current_frame, velocity]
    )

    prev_frame = current_frame

    sequence.append(combined)

    if len(sequence) > 30:
        sequence.pop(0)

    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    if len(sequence) == 30:

        x = np.array(
            sequence,
            dtype=np.float32
        )

        x = np.expand_dims(
            x,
            axis=0
        )

        prediction = model.predict(
            x,
            verbose=0
        )[0]

        pred_class = int(
            np.argmax(prediction)
        )

        current_conf = float(
            np.max(prediction)
        )

        if current_conf >= CONFIDENCE_THRESHOLD:

            predicted_word = sign_names[
                str(pred_class)
            ]

            prediction_window.append(
                predicted_word
            )

            stable_prediction = Counter(
                prediction_window
            ).most_common(1)[0][0]

            # -------------------------------
            # Stable Prediction Logic
            # -------------------------------

            if stable_prediction == stable_word:

                stable_count += 1

            else:

                stable_word = stable_prediction

                stable_count = 1

            current_sign = stable_word

            # -------------------------------
            # Accept New Word
            # -------------------------------

            if (
                stable_count >= STABLE_THRESHOLD
                and stable_word != last_added_word
            ):

                builder.add_word(
                    stable_word
                )

                last_added_word = stable_word

                print(
                    "Added:",
                    stable_word
                )

                print(
                    "Sentence:",
                    builder.get_sentence()
                )

                    # -----------------------------------------
    # Display
    # -----------------------------------------

    cv2.putText(
        frame,
        f"Current Sign : {current_sign}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {current_conf:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Sentence:",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        builder.get_sentence(),
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Realtime Sentence Recognition",
        frame
    )

    # -----------------------------------------
    # Keyboard Controls
    # -----------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Clear sentence
    elif key == ord("c"):

        builder.clear()

        prediction_window.clear()

        stable_word = ""

        stable_count = 0

        last_added_word = ""

        print("\nSentence Cleared\n")

    # Print sentence
    elif key == 13:      # Enter key

        print("\n==========================")
        print("Final Sentence :")
        print(builder.get_sentence())
        print("==========================\n")


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()