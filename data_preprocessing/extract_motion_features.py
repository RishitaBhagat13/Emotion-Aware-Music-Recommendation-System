import os
import numpy as np

INPUT_DIR = r"data_preprocessing\datasets\normalized_features"

OUTPUT_DIR = r"data_preprocessing\datasets\motion_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".npy")
]

print("Files:", len(files))

for file in files:

    data = np.load(
        os.path.join(INPUT_DIR, file)
    )

    motion_sequence = []

    prev_frame = np.zeros(
        126,
        dtype=np.float32
    )

    for frame in data:

        velocity = frame - prev_frame

        combined = np.concatenate(
            [frame, velocity]
        )

        motion_sequence.append(
            combined
        )

        prev_frame = frame

    motion_sequence = np.array(
        motion_sequence,
        dtype=np.float32
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            file
        ),
        motion_sequence
    )

print("Motion Feature Extraction Complete!")