import cv2
from src.pipeline import VehicleReIDPipeline

pipeline = VehicleReIDPipeline(config_path="configs/pipeline.yaml")
cap = cv2.VideoCapture("/home/youssef/Desktop/188613-883402208.mp4")

if not cap.isOpened():
    print("Failed to open video")
    exit(1)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    print(f"Processing frame {i}...")
    annotated_frame, track_records = pipeline.process_frame(frame)
    print(f"Frame {i} processed. Found {len(track_records)} tracks.")

cap.release()
print("Pipeline test successful!")
