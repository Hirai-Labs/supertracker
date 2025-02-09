# supertracker

**An easy-to-use library for implementing various multi-object tracking algorithms.**

## Features

- Simple and intuitive interface for multi-object tracking
- Multiple tracking algorithms supported:
  - ByteTrack
  - DeepSORT (coming soon)
  - SORT (coming soon)
  - OCSORT (coming soon)
  - BoT-SORT (coming soon)
- Support for custom detection formats
- Easy integration with existing detection pipelines
- Modular design for adding new trackers

## Installation

```bash
pip install supertracker
```

## Examples

```python
import cv2
from ultralytics import YOLO
from supertracker import ByteTrack
from supertracker import Detections

# Initialize YOLO model and tracker
model = YOLO('yolov8n.pt')  # or your custom model
tracker = ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    frame_rate=30
)

# Initialize video capture
cap = cv2.VideoCapture(0)  # or video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)[0]
    
    # Convert YOLO results to Detections format
    detections = Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )
    
    # Update tracker
    tracked_objects = tracker.update_with_detections(detections)

    # Visualize results
    for i in range(len(tracked_objects)):
        box = tracked_objects.xyxy[i].astype(int)
        track_id = tracked_objects.tracker_id[i]
        class_id = tracked_objects.class_id[i]
        conf = tracked_objects.confidence[i]
        
        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Draw label with class name, track ID and confidence
        label = f"#{track_id} {model.names[class_id]} {conf:.2f}"
        cv2.putText(frame, label, (box[0], box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Documentation
Visit our [documentation](https://Hirai-Labs.github.io/supertracker) for more details

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
