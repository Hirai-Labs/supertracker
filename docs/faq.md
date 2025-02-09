# Frequently Asked Questions

## General Questions

### What is supertracker?
Supertracker is a Python library that provides a unified interface for multiple object tracking algorithms, designed to work seamlessly with various object detection models.

### Which tracking algorithms are supported?
Currently, ByteTrack is fully implemented. DeepSORT, SORT, OCSORT, and BoT-SORT are planned for future releases.

### What detection formats are supported?
Supertracker works with any detection format that can provide bounding boxes (xyxy format), confidence scores, and optional class IDs. It has built-in support for YOLO format detections.

## Installation

### Why can't I install supertracker?
Make sure you have Python 3.7 or newer installed. Also check that you have numpy and opencv-python in your environment:
```bash
pip install numpy opencv-python
pip install supertracker
```

## Usage

### How do I switch between different trackers?
Currently, only ByteTrack is available. When other trackers are implemented, you can switch by importing the desired tracker:
```python
from supertracker import ByteTrack  # Currently available
# from supertracker import DeepSORT  # Coming soon
# from supertracker import SORT      # Coming soon
```

### Why are my tracks getting lost quickly?
Check these common issues:

1. Adjust `lost_track_buffer` for longer track retention
2. Lower `track_activation_threshold` if detections are weak
3. Ensure consistent frame rate processing

### How do I optimize for speed vs accuracy?
- For speed: Lower `lost_track_buffer`, increase `track_activation_threshold`
- For accuracy: Increase `lost_track_buffer`, lower `track_activation_threshold`

## Performance

### What's the expected FPS?
Performance depends on:

- Detection model speed
- Image resolution
- Hardware capabilities
- Number of objects

Typical performance on modern hardware:

- 30+ FPS at 720p
- 20+ FPS at 1080p

### Memory Usage
Typical memory usage:

- Base: ~100MB
- Per track: negligible (~1KB)
- Total: Depends on number of active tracks

## Integration

### Can I use it with custom detection models?
Yes, as long as you can provide detections in the format:
```python
detections = Detections(
    xyxy=boxes,          # numpy array of shape (N, 4)
    confidence=scores,    # numpy array of shape (N,)
    class_id=class_ids   # optional, numpy array of shape (N,)
)
```

### Does it work with TensorRT?
Yes, supertracker works with any detection model, including TensorRT optimized ones. Just convert the detections to our format.

## Troubleshooting

### Common Error Messages

#### "Dimension mismatch in detections"
Check that your detection format matches:

- xyxy: (N, 4) shape
- confidence: (N,) shape
- class_id: (N,) shape

#### "No tracks found"
Common causes:

1. Detection confidence too low
2. `track_activation_threshold` too high
3. No detections being passed to tracker

## Contributing

### How can I add a new tracker?
1. Fork the repository
2. Implement the tracker following our template
3. Ensure tests pass
4. Submit a pull request

### Where can I report bugs?
Please report bugs on our [GitHub Issues](https://github.com/Hirai-Labs/supertracker/issues) page with:

1. Minimal reproduction code
2. Error message
3. Environment details
