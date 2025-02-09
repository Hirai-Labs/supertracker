import numpy as np
import pytest
from supertracker.bytetrack.core import ByteTrack
from supertracker.detection.core import Detections
from supertracker.bytetrack.single_object_track import STrack

@pytest.fixture
def tracker():
    return ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=1
    )

@pytest.fixture
def sample_detections():
    return Detections(
        xyxy=np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([1, 1], dtype=int)
    )

def test_bytetrack_initialization(tracker):
    """Test ByteTrack initialization with default parameters"""
    assert tracker.track_activation_threshold == 0.25
    assert tracker.minimum_matching_threshold == 0.8
    assert tracker.frame_id == 0
    assert tracker.max_time_lost == 30
    assert len(tracker.tracked_tracks) == 0
    assert len(tracker.lost_tracks) == 0
    assert len(tracker.removed_tracks) == 0

def test_bytetrack_reset(tracker):
    """Test ByteTrack reset functionality"""
    # Setup tracker with some data
    tracker.frame_id = 10
    tracker.tracked_tracks = [1, 2, 3]
    tracker.lost_tracks = [4, 5]
    tracker.removed_tracks = [6]
    
    tracker.reset()
    
    assert tracker.frame_id == 0
    assert len(tracker.tracked_tracks) == 0
    assert len(tracker.lost_tracks) == 0
    assert len(tracker.removed_tracks) == 0

def test_bytetrack_update_with_empty_detections(tracker):
    """Test ByteTrack update with empty detections"""
    empty_detections = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty(0, dtype=np.float32),
        class_id=np.empty(0, dtype=int)
    )
    result = tracker.update_with_detections(empty_detections)
    assert len(result) == 0
    assert hasattr(result, 'tracker_id')

def test_bytetrack_update_with_single_detection(tracker):
    """Test ByteTrack update with a single detection"""
    single_detection = Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([1], dtype=int)
    )
    result = tracker.update_with_detections(single_detection)
    assert len(result) == 1
    assert hasattr(result, 'tracker_id')
    assert result.tracker_id[0] >= 0  # Valid track ID assigned

def test_bytetrack_update_with_multiple_detections(tracker, sample_detections):
    """Test ByteTrack update with multiple detections"""
    result = tracker.update_with_detections(sample_detections)
    assert len(result) == 2
    assert hasattr(result, 'tracker_id')
    assert all(tid >= 0 for tid in result.tracker_id)  # All valid track IDs

def test_bytetrack_track_continuity(tracker):
    """Test if ByteTrack maintains track continuity"""
    # First frame
    det1 = Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([1], dtype=int)
    )
    result1 = tracker.update_with_detections(det1)
    first_track_id = result1.tracker_id[0]
    
    # Second frame (same object moved slightly)
    det2 = Detections(
        xyxy=np.array([[110, 110, 210, 210]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([1], dtype=int)
    )
    result2 = tracker.update_with_detections(det2)
    second_track_id = result2.tracker_id[0]
    
    assert first_track_id == second_track_id  # Same object should maintain ID

def test_bytetrack_low_confidence_detections(tracker):
    """Test ByteTrack handling of low confidence detections"""
    low_conf_det = Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        confidence=np.array([0.2], dtype=np.float32),  # Below activation threshold
        class_id=np.array([1], dtype=int)
    )
    result = tracker.update_with_detections(low_conf_det)
    assert len(result) == 0  # Should not create track for low confidence detection

def test_bytetrack_track_termination(tracker):
    """Test if ByteTrack properly terminates lost tracks"""
    # Create initial track
    det = Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([1], dtype=int)
    )
    tracker.update_with_detections(det)
    
    # Simulate missing detections for max_time_lost + 1 frames
    empty_det = Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty(0, dtype=np.float32),
        class_id=np.empty(0, dtype=int)
    )
    
    for _ in range(tracker.max_time_lost + 1):
        result = tracker.update_with_detections(empty_det)
    
    assert len(tracker.tracked_tracks) == 0
    assert len(result) == 0

def test_bytetrack_multiple_tracks_matching(tracker):
    """Test ByteTrack matching with multiple concurrent tracks"""
    # First frame with two objects
    det1 = Detections(
        xyxy=np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400]
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.9], dtype=np.float32),
        class_id=np.array([1, 1], dtype=int)
    )
    result1 = tracker.update_with_detections(det1)
    initial_ids = set(result1.tracker_id)
    
    # Second frame with same objects slightly moved
    det2 = Detections(
        xyxy=np.array([
            [110, 110, 210, 210],
            [310, 310, 410, 410]
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.9], dtype=np.float32),
        class_id=np.array([1, 1], dtype=int)
    )
    result2 = tracker.update_with_detections(det2)
    new_ids = set(result2.tracker_id)
    
    assert initial_ids == new_ids  # Track IDs should be maintained
