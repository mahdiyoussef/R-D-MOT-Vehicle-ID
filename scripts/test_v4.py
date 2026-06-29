import cv2
import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import VehicleReIDPipeline

# Ensure pipeline.yaml uses dinov2 and new backends
cfg_path = "configs/pipeline.yaml"
pipeline = VehicleReIDPipeline(config_path=cfg_path)

video_path = "/home/youssef/Desktop/Highway Chinese 1080p.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

print("Starting pipeline test...")
for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error.")
        break
    
    # Process the frame
    try:
        annotated_frame, track_records = pipeline.process_frame(frame)
        print(f"Frame {i+1}: processed successfully. Found {len(track_records)} tracks.")
    except Exception as e:
        print(f"Error processing frame {i+1}: {e}")
        import traceback
        traceback.print_exc()
        break

cap.release()
print("Test finished.")
