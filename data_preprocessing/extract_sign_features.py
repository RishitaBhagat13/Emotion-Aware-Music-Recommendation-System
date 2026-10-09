import os
import numpy as np
import mediapipe as mp

# =====================================
# PATHS
# =====================================

FRAMES_DIR = r"C:\Users\BIT\Desktop\Deaf_care_Proj\data_preprocessing\datasets\extracted_frames"

OUTPUT_DIR = r"C:\Users\BIT\Desktop\Deaf_care_Proj\data_preprocessing\datasets\extracted_features"

MODEL_PATH = r"C:\Users\BIT\Desktop\Deaf_care_Proj\models\hand_landmarker.task"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# MEDIAPIPE SETUP
# =====================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

landmarker = HandLandmarker.create_from_options(options)

# =====================================
# PROCESS VIDEOS
# =====================================

video_folders = sorted(os.listdir(FRAMES_DIR))

print(f"Total video folders: {len(video_folders)}")

for video_folder in video_folders:

    folder_path = os.path.join(
        FRAMES_DIR,
        video_folder
    )

    if not os.path.isdir(folder_path):
        continue

    frame_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".jpg")
    ])

    sequence = []

    for frame_file in frame_files:

        frame_path = os.path.join(
            folder_path,
            frame_file
        )

        try:

            image = mp.Image.create_from_file(
                frame_path
            )

            result = landmarker.detect(
                image
            )

            features = np.zeros(
                126,
                dtype=np.float32
            )

            idx = 0

            if result.hand_landmarks:

                for hand in result.hand_landmarks[:2]:

                    for lm in hand:

                        features[idx] = lm.x
                        idx += 1

                        features[idx] = lm.y
                        idx += 1

                        features[idx] = lm.z
                        idx += 1

            sequence.append(features)

        except Exception as e:

            print(
                f"Error in {frame_path}: {e}"
            )

    # Ensure exactly 30 frames

    while len(sequence) < 30:
        sequence.append(
            np.zeros(
                126,
                dtype=np.float32
            )
        )

    sequence = sequence[:30]

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{video_folder}.npy"
    )

    np.save(
        save_path,
        sequence
    )

    print(
        f"Saved: {video_folder}"
    )

print("\nFeature Extraction Complete!")