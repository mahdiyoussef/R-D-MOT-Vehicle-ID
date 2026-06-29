import cv2
import yaml
import sys
from src.pipeline import VehicleReIDPipeline

def test_pipeline():
    config_path = "configs/pipeline.yaml"
    print("Initializing pipeline...")
    try:
        pipeline = VehicleReIDPipeline(config_path=config_path)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    video_path = "/home/youssef/Desktop/police_chase_h264.mp4"
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        sys.exit(1)

    frame_count = 0
    print("Starting frame processing...")
    while cap.isOpened() and frame_count < 10:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or failed to read frame.")
            break
        
        try:
            results = pipeline.process_frame(frame)
            print(f"Frame {frame_count}: Processed successfully. Detections/Tracks: {len(results)}")
        except Exception as e:
            print(f"Failed to process frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
        frame_count += 1

    cap.release()
    print("Test completed successfully.")

if __name__ == "__main__":
    test_pipeline()
