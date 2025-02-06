import cv2
import numpy as np
from ultralytics import YOLO
import supertracker

def process_custom_bytetrack(frame, model, tracker, results):
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    detections = supertracker.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids
    )
    
    return tracker.update_with_detections(detections)

def process_ultralytics_tracker(frame, model, results):
    return results[0].tracker_id, results[0]

def process_video(source_path, model_path, output_path, use_custom_tracker=True):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Initialize video capture and writer
    cap = cv2.VideoCapture(source_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    # Initialize tracker based on selection
    tracker = supertracker.ByteTrack(frame_rate=fps) if use_custom_tracker else None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection with or without built-in tracker
        if use_custom_tracker:
            results = model(frame)
            tracked_detections = process_custom_bytetrack(frame, model, tracker, results[0])
        else:
            results = model.track(frame, tracker="bytetrack.yaml")
            tracked_detections = results[0]
        
        # Draw boxes and labels
        if use_custom_tracker:
            for i in range(len(tracked_detections)):
                x1, y1, x2, y2 = tracked_detections.xyxy[i].astype(int)
                conf = tracked_detections.confidence[i]
                class_id = tracked_detections.class_id[i]
                tracker_id = tracked_detections.tracker_id[i]
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"#{tracker_id} {model.names[class_id]} {conf:.2f}"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width + 10, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
        else:
            # Draw using Ultralytics results
            annotated_frame = results[0].plot()
            frame = annotated_frame
        
        writer.write(frame)
    
    cap.release()
    writer.release()

def main():
    VIDEO_PATH = "cr.mp4"
    MODEL_PATH = "person_v3.pt" 
    OUTPUT_PATH = "output.mp4"
    USE_CUSTOM_TRACKER = True  # Set to False to use Ultralytics tracker
    
    process_video(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH, USE_CUSTOM_TRACKER)

if __name__ == "__main__":
    main()