import cv2
import json
import numpy as np
import mediapipe as mp

from collections import Counter, deque


from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sentence_builder import SentenceBuilder
from emotion_tracker import EmotionTracker


# ==========================================================
# LOAD MODELS
# ==========================================================

sign_model = load_model(
    r"trained_modal\final_motion_attention_model.keras"
)

emotion_model = load_model(
    r"models\rafdb_emotion_model.keras"
)


# ==========================================================
# LOAD SIGN LABELS
# ==========================================================

with open(
    r"training\nslt100_sign_names.json",
    "r"
) as f:

    sign_names = json.load(f)


# ==========================================================
# EMOTION LABELS
# ==========================================================

emotion_map = {

    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"

}


# ==========================================================
# FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(

    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"

)


# ==========================================================
# MEDIAPIPE
# ==========================================================

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

# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

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


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_frame(frame):

    frame = frame.copy()

    # Left Hand

    if np.sum(frame[:63]) != 0:

        wrist_x = frame[0]
        wrist_y = frame[1]
        wrist_z = frame[2]

        for i in range(0, 63, 3):

            frame[i] -= wrist_x
            frame[i + 1] -= wrist_y
            frame[i + 2] -= wrist_z

    # Right Hand

    if np.sum(frame[63:]) != 0:

        wrist_x = frame[63]
        wrist_y = frame[64]
        wrist_z = frame[65]

        for i in range(63, 126, 3):

            frame[i] -= wrist_x
            frame[i + 1] -= wrist_y
            frame[i + 2] -= wrist_z

    return frame


# ==========================================================
# TEMPORARY FUSION MESSAGE
# (Later replaced by intent_parser.py)
# ==========================================================

def generate_message(sentence, emotion):

    if sentence.strip() == "":

        return "Waiting for sign..."

    return f"{sentence} | Emotion: {emotion}"


# ==========================================================
# VARIABLES
# ==========================================================

sequence = []

prev_frame = np.zeros(
    126,
    dtype=np.float32
)

# --------------------------
# Emotion
# --------------------------

current_emotion = "Neutral"
emotion_conf = 0.0

emotion_tracker = EmotionTracker()

# --------------------------
# Sign
# --------------------------

current_sign = "Waiting..."
current_conf = 0.0

prediction_window = deque(maxlen=10)

stable_word = ""

stable_count = 0

last_added = ""

CONFIDENCE_THRESHOLD = 0.60

STABLE_THRESHOLD = 8

# --------------------------
# Sentence Builder
# --------------------------

builder = SentenceBuilder()

# ==========================================================
# CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

print("=" * 60)
print("Realtime Multimodal Recommendation System")
print("=" * 60)
print("Press C -> Clear Sentence")
print("Press ENTER -> Print Sentence")
print("Press Q -> Quit")
print("=" * 60)

# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ======================================================
    # EMOTION DETECTION
    # ======================================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.2,

        minNeighbors=5,

        minSize=(80,80)

    )

    for (x,y,w,h) in faces:

        try:

            face = frame[
                y:y+h,
                x:x+w
            ]

            face = cv2.resize(
                face,
                (224,224)
            )

            face = np.array(
                face,
                dtype=np.float32
            )

            face = preprocess_input(
                face
            )

            face = np.expand_dims(
                face,
                axis=0
            )

            prediction = emotion_model.predict(
                face,
                verbose=0
            )[0]

            emotion_id = int(
                np.argmax(prediction)
            )

            confidence = float(
                np.max(prediction)
            )

            emotion = emotion_map[
                emotion_id
            ]

            # ----------------------------
            # Stable Emotion
            # ----------------------------

            emotion_tracker.update(

                emotion,

                confidence

            )

            current_emotion = emotion_tracker.get_emotion()

            emotion_conf = emotion_tracker.get_confidence()

            # ----------------------------
            # Draw Face Box
            # ----------------------------

            cv2.rectangle(

                frame,

                (x,y),

                (x+w,y+h),

                (255,0,0),

                2

            )

            cv2.putText(

                frame,

                current_emotion,

                (x,y-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255,0,0),

                2

            )

        except Exception as e:

            print(
                "Emotion Error:",
                e
            )

    # ======================================================
    # HAND LANDMARK DETECTION
    # ======================================================

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

        h,w,_ = frame.shape

        for hand in result.hand_landmarks:

            for lm in hand:

                cx = int(lm.x * w)

                cy = int(lm.y * h)

                cv2.circle(

                    frame,

                    (cx,cy),

                    3,

                    (0,255,0),

                    -1

                )

    # ======================================================
    # FEATURE EXTRACTION
    # ======================================================

    current_frame = extract_features(
        result
    )

    current_frame = normalize_frame(
        current_frame
    )

    velocity = current_frame - prev_frame

    combined = np.concatenate(

        [

            current_frame,

            velocity

        ]

    )

    prev_frame = current_frame

    sequence.append(
        combined
    )

    if len(sequence) > 30:

        sequence.pop(0)

        # ======================================================
    # SIGN PREDICTION
    # ======================================================

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

        if current_conf >= CONFIDENCE_THRESHOLD:

            predicted_word = sign_names[
                str(pred_class)
            ]

            # ------------------------------------
            # Smooth Prediction
            # ------------------------------------

            prediction_window.append(
                predicted_word
            )

            stable_prediction = Counter(
                prediction_window
            ).most_common(1)[0][0]

            if stable_prediction == stable_word:

                stable_count += 1

            else:

                stable_word = stable_prediction

                stable_count = 1

            current_sign = stable_word

            # ------------------------------------
            # Add to Sentence
            # ------------------------------------

            if (
                stable_count >= STABLE_THRESHOLD
                and stable_word != last_added
            ):

                builder.add_word(
                    stable_word
                )

                last_added = stable_word
                print("\n" + "="*50)
                print("Added :", stable_word)
                print(
                        "Sentence :",
                        builder.get_sentence()
                )
                print(
                        "Emotion :",
                        current_emotion
                )
                print("="*50)
    # ======================================================
    # CURRENT MESSAGE
    # ======================================================

    message = generate_message(

        builder.get_sentence(),

        current_emotion

    )

        # ======================================================
    # DISPLAY
    # ======================================================

    cv2.putText(
        frame,
        f"Emotion : {current_emotion}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Emotion Confidence : {emotion_conf:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Current Sign : {current_sign}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Sign Confidence : {current_conf:.2f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Sentence :",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        builder.get_sentence(),
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        message,
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Multimodal Deaf Care System",
        frame
    )

    # ======================================================
    # KEYBOARD
    # ======================================================

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

        last_added = ""

        current_sign = "Waiting..."

        emotion_tracker.reset()

        current_emotion = "Neutral"

        emotion_conf = 0.0

        print("\n==========================")
        print("Sentence Cleared")
        print("==========================\n")

    # Print final sentence

    elif key == 13:

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)

        print(
            "Sentence :",
            builder.get_sentence()
        )

        print(
            "Emotion :",
            current_emotion
        )

        print("=" * 60 + "\n")


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()