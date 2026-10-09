import os
import json
import cv2

VIDEO_DIR = r"datasets\videos"
OUTPUT_DIR = r"data_preprocessing\datasets\extracted_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load NSLT-100 annotations
with open(r"datasets\nslt_100.json", "r") as f:
    annotations = json.load(f)

video_ids = set(annotations.keys())

print("Videos to process:", len(video_ids))

for video_id in video_ids:

    video_file = os.path.join(
        VIDEO_DIR,
        f"{video_id}.mp4"
    )

    if not os.path.exists(video_file):
        continue

    save_folder = os.path.join(
        OUTPUT_DIR,
        video_id
    )

    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_file)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Uniform sampling of 30 frames
    frame_indices = []

    if total_frames > 30:
        step = total_frames / 30

        frame_indices = [
            int(i * step)
            for i in range(30)
        ]
    else:
        frame_indices = list(range(total_frames))

    count = 0

    current_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if current_frame in frame_indices:

            frame_path = os.path.join(
                save_folder,
                f"frame_{count:03d}.jpg"
            )

            cv2.imwrite(frame_path, frame)

            count += 1

        current_frame += 1

    cap.release()

    print(
        f"{video_id}: {count} frames"
    )

print("Frame extraction complete!")