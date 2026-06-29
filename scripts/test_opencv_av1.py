import cv2

video_path = "outputs/temp/police chase 720p.mp4"
print(f"Opening {video_path}...")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video. Maybe it's named something else?")
    exit(1)

frames = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Reached end of video after {frames} frames.")
        break
    frames += 1

print("Releasing cap...")
cap.release()
print("Done. Look above to see if the AV1 warning printed purely from OpenCV!")
