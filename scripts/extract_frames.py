import cv2
import os

champion = 'lux'
video_path = "./video/lux.mp4"
output_dir = "./frames"
frame_interval = 10

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"{champion}_{saved_count:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames for {champion}")
