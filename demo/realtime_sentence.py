import cv2
import json
import time
import numpy as np
import mediapipe as mp

from collections import deque, Counter
from tensorflow.keras.models import load_model

# =====================================
# LOAD MODEL
# =====================================

model = load_model(
    r"trained_modal\final_motion_attention_model.keras"
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

landmarker = HandLandmarker.create_from_options(options)

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(result):

    features = np.zeros(126, dtype=np.float32)

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

    # LEFT HAND
    if np.sum(frame[:63]) != 0:

        wrist_x = frame[0]
        wrist_y = frame[1]
        wrist_z = frame[2]

        for i in range(0, 63, 3):

            frame[i] -= wrist_x
            frame[i + 1] -= wrist_y
            frame[i + 2] -= wrist_z

    # RIGHT HAND
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
# VARIABLES
# =====================================

sequence = []

prev_frame = np.zeros(
    126,
    dtype=np.float32
)

current_sign = "Waiting..."
current_conf = 0.0

# ---------- Sentence Builder ----------

prediction_window = deque(maxlen=10)

stable_sign = ""

same_count = 0

sentence = []

last_added = ""

last_word_time = time.time()

no_hand_frames = 0

WAITING_FOR_NEXT_SIGN = False

CONFIDENCE_THRESHOLD = 0.60

STABLE_COUNT = 8

HAND_GAP = 15        # frames with no hands

SENTENCE_TIMEOUT = 2 # seconds


# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

print("Realtime Sign Detection Started")
print("Press Q to quit")

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

    hands_present = result.hand_landmarks is not None and len(result.hand_landmarks) > 0

    # =====================================
    # DRAW HANDS
    # =====================================

    if hands_present:

        no_hand_frames = 0

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

    else:

        no_hand_frames += 1

        # Ready to accept next word after hand disappears
        if no_hand_frames > HAND_GAP:
            WAITING_FOR_NEXT_SIGN = False

    # =====================================
    # FEATURE EXTRACTION
    # =====================================

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

    # =====================================
    # PREDICTION
    # =====================================

    if len(sequence) == 30 and hands_present:

        x = np.array(sequence, dtype=np.float32)

        x = np.expand_dims(
            x,
            axis=0
        )

        prediction = model.predict(
            x,
            verbose=0
        )[0]

        pred_class = np.argmax(prediction)

        current_conf = float(np.max(prediction))

        if current_conf > CONFIDENCE_THRESHOLD:

            predicted_word = sign_names[str(int(pred_class))]

            prediction_window.append(predicted_word)

            stable_word = Counter(
                prediction_window
            ).most_common(1)[0][0]

            if stable_word == stable_sign:
                same_count += 1
            else:
                stable_sign = stable_word
                same_count = 1

            current_sign = stable_sign

            # ---------- ADD WORD ----------

            if (
                same_count >= STABLE_COUNT
                and not WAITING_FOR_NEXT_SIGN
            ):

                if stable_sign != last_added:

                    sentence.append(stable_sign)

                    last_added = stable_sign

                    WAITING_FOR_NEXT_SIGN = True

                    last_word_time = time.time()

                    print("Added:", stable_sign)

    # =====================================
    # SENTENCE COMPLETE
    # =====================================

    if len(sentence) > 0:

        if (
            time.time() - last_word_time
            > SENTENCE_TIMEOUT
            and no_hand_frames > HAND_GAP
        ):

            final_sentence = " ".join(sentence)

            print("\n====================")
            print("Sentence:", final_sentence)
            print("====================\n")

            # -----------------------------------
            # TODO:
            # Send final_sentence to
            # your recommender system
            # -----------------------------------

            sentence.clear()

            prediction_window.clear()

            stable_sign = ""

            same_count = 0

            last_added = ""

            WAITING_FOR_NEXT_SIGN = False

    # =====================================
    # DISPLAY
    # =====================================

    cv2.putText(
        frame,
        f"Sign: {current_sign}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Conf: {current_conf:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Sentence:",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        " ".join(sentence),
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Realtime Sign Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()