import os
import numpy as np

INPUT_DIR = r"data_preprocessing\datasets\extracted_features"
OUTPUT_DIR = r"data_preprocessing\datasets\normalized_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")]

for file in files:

    data = np.load(
        os.path.join(INPUT_DIR, file)
    )

    normalized = []

    for frame in data:

        frame = frame.copy()

        # LEFT HAND
        if np.sum(frame[:63]) != 0:

            wrist_x = frame[0]
            wrist_y = frame[1]
            wrist_z = frame[2]

            for i in range(0, 63, 3):

                frame[i] -= wrist_x
                frame[i+1] -= wrist_y
                frame[i+2] -= wrist_z

        # RIGHT HAND
        if np.sum(frame[63:]) != 0:

            wrist_x = frame[63]
            wrist_y = frame[64]
            wrist_z = frame[65]

            for i in range(63, 126, 3):

                frame[i] -= wrist_x
                frame[i+1] -= wrist_y
                frame[i+2] -= wrist_z

        normalized.append(frame)

    normalized = np.array(
        normalized,
        dtype=np.float32
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            file
        ),
        normalized
    )

print("Normalization complete!")