import pytest
import numpy as np
from supertracker.bytetrack.core import ByteTrack
from supertracker.detection.core import Detections
from supertracker.bytetrack.kalman_filter import KalmanFilter

@pytest.fixture
def tracker():
    return ByteTrack()

@pytest.fixture
def kalman():
    return KalmanFilter()

@pytest.fixture
def sample_detections():
    xyxy = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
    confidence = np.array([0.9, 0.8], dtype=np.float32)
    class_id = np.array([1, 2], dtype=int)
    return Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
