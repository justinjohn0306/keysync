import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from landmarks_extractor import LandmarksExtractor


def process_video(video_path, landmarks_extractor, output_dir, batch_size=32):
    video_parent_dir = os.path.dirname(video_path).split("/")[-1]

    # Get video name without extension
    output_path = video_path.replace(video_parent_dir, output_dir).replace(
        ".mp4", ".npy"
    )

    if os.path.exists(output_path):
        print(f"Landmarks already exist for {video_path}, skipping...")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize list to store landmarks for all frames
    all_landmarks = []

    # Process frames in batches
    frames_batch = []
    for _ in tqdm(range(total_frames), desc=f"Processing {video_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_batch.append(frame_rgb)

        # Process batch when it reaches the batch size or at the end
        if len(frames_batch) == batch_size or _ == total_frames - 1:
            # Convert list of frames to numpy array
            frames_array = torch.from_numpy(np.array(frames_batch)).permute(0, 3, 1, 2)

            # Extract landmarks for the batch
            batch_landmarks = landmarks_extractor.extract_landmarks(frames_array)

            all_landmarks.extend(batch_landmarks)

            # Clear the batch
            frames_batch = []

    # Release video capture
    cap.release()

    # Save landmarks
    np.save(output_path, np.array(all_landmarks))
    print(f"Saved landmarks to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract facial landmarks from videos")
    parser.add_argument("video_dir", help="Directory containing video files")
    parser.add_argument(
        "--output_dir", default="landmarks", help="Directory to save landmarks"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run face alignment on (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for processing frames"
    )
    args = parser.parse_args()

    # Initialize landmarks extractor
    landmarks_extractor = LandmarksExtractor(device=args.device)

    # Process all video files in the directory
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    for filename in os.listdir(args.video_dir):
        if filename.lower().endswith(video_extensions):
            video_path = os.path.join(args.video_dir, filename)
            process_video(
                video_path, landmarks_extractor, args.output_dir, args.batch_size
            )


if __name__ == "__main__":
    main()
