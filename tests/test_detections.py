import pytest
import numpy as np
from supertracker.detection.core import Detections

@pytest.fixture
def sample_detections():
    xyxy = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
    confidence = np.array([0.9, 0.8], dtype=np.float32)
    class_id = np.array([1, 2], dtype=int)
    return Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id
    )

def test_detections_init(sample_detections):
    assert sample_detections.xyxy.shape == (2, 4)
    assert sample_detections.confidence.shape == (2,)
    assert sample_detections.class_id.shape == (2,)

def test_detections_len(sample_detections):
    assert len(sample_detections) == 2

def test_detections_iter(sample_detections):
    items = list(sample_detections)
    assert len(items) == 2
    assert np.array_equal(items[0][0], sample_detections.xyxy[0])
    assert items[0][1] == sample_detections.confidence[0]
    assert items[0][2] == sample_detections.class_id[0]

def test_detections_getitem_slice(sample_detections):
    sliced = sample_detections[0:2]
    assert np.array_equal(sliced.xyxy, sample_detections.xyxy)
    assert np.array_equal(sliced.confidence, sample_detections.confidence)
    assert np.array_equal(sliced.class_id, sample_detections.class_id)

def test_detections_empty():
    empty = Detections.empty()
    assert len(empty) == 0
    assert empty.xyxy.shape == (0, 4)
    assert empty.confidence.shape == (0,)
    assert empty.class_id.shape == (0,)

@pytest.mark.parametrize("invalid_index", ["string", 1.5, None])
def test_detections_invalid_index(sample_detections, invalid_index):
    with pytest.raises(TypeError):
        _ = sample_detections[invalid_index]
