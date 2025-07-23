import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

# --- Configuration ---
video_path = "./assets/video/vid_2.mp4" # Video Path
base_output_dir = "./frames" # Output Path
frame_interval = 120 # Number of frames between capture
capture_cap = 1000 # Number of captures before stopping
max_workers = os.cpu_count() # dont change this shit

# ---------------------

def save_frame(frame_data, output_path):
    try:
        cv2.imwrite(output_path, frame_data)
    except Exception as e:
        print(f"Error saving frame to {output_path}: {e}")

def main():
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"{video_basename}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    frame_count = 0
    saved_count = 0
    start_time = time.time()

    print(f"Starting frame extraction...")
    print(f"   Video: {video_path}")
    print(f"   Output: {output_dir}")
    print(f"   Interval: Every {frame_interval} frames")
    print(f"   Workers: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_to_save = frame.copy()
                filename = f"{video_basename}_frame_{saved_count:05d}.jpg"
                output_path = os.path.join(output_dir, filename)
                executor.submit(save_frame, frame_to_save, output_path)
                saved_count += 1

            if saved_count >= capture_cap:
                break

            frame_count += 1

    cap.release()

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*40)
    print(f"Extraction Complete!")
    print(f"   Extracted {saved_count} frames.")
    print(f"   Total time: {duration:.2f} seconds.")
    print(f"   Frames saved in '{os.path.abspath(output_dir)}'")
    print("="*40)

if __name__ == "__main__":
    main()