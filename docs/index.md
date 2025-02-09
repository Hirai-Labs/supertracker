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

## Support

- [GitHub Issues](https://github.com/Hirai-Labs/supertracker/issues)
- [Documentation](https://Hirai-Labs.github.io/supertracker)
- [Examples Repository](https://github.com/Hirai-Labs/supertracker/tree/main/examples)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
