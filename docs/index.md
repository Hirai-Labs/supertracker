# Welcome to supertracker

[![image](https://img.shields.io/pypi/v/supertracker.svg)](https://pypi.python.org/pypi/supertracker)

**An easy-to-use library for implementing various multi-object tracking algorithms.**

## Overview

Supertracker provides a unified interface for multiple object tracking algorithms, making it easy to implement and switch between different tracking approaches in your computer vision applications.

## Available Trackers

### ByteTrack
- High-performance multi-object tracking
- Robust occlusion handling
- Configurable parameters for different scenarios
- Ideal for real-time applications

### Coming Soon
- DeepSORT: Deep learning enhanced tracking
- SORT: Simple online realtime tracking
- OCSORT: Observation-centric SORT
- BoT-SORT: Bootstrap your own SORT

## Quick Installation

```bash
pip install supertracker
```

## Basic Usage

```python
from supertracker import ByteTrack
from supertracker import Detections

# Initialize tracker
tracker = ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    frame_rate=30
)

# Process detections
tracked_objects = tracker.update_with_detections(detections)
```

## Documentation Structure

- **Getting Started**: Basic installation and usage
- **Tutorials**: Step-by-step guides for common scenarios
- **API Reference**: Detailed documentation of all classes and methods
- **Examples**: Real-world implementation examples
- **Configuration**: Tracker-specific parameter tuning
- **Contributing**: Guidelines for contributing to the project

## Performance

Each tracker implementation is optimized for:
- Real-time processing
- Memory efficiency
- Accuracy in various scenarios
- Robust handling of occlusions

## Integration Examples

We provide examples for integration with popular detection frameworks:
- YOLO (v5, v6, v7, v8)
- Detectron2
- TensorFlow Object Detection API
- Custom detection models

## Support

- [GitHub Issues](https://github.com/Hirai-Labs/supertracker/issues)
- [Documentation](https://Hirai-Labs.github.io/supertracker)
- [Examples Repository](https://github.com/Hirai-Labs/supertracker/tree/main/examples)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
