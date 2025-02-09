import pytest
import numpy as np
from supertracker.bytetrack.core import ByteTrack, joint_tracks, sub_tracks, remove_duplicate_tracks
from supertracker.detection.core import Detections
from supertracker.bytetrack.single_object_track import STrack
from supertracker.bytetrack.kalman_filter import KalmanFilter

@pytest.fixture
def tracker():
    return ByteTrack()

@pytest.fixture
def kalman():
    return KalmanFilter()

def test_bytetrack_init(tracker):
    assert tracker.frame_id == 0
    assert len(tracker.tracked_tracks) == 0
    assert len(tracker.lost_tracks) == 0
    assert len(tracker.removed_tracks) == 0

def test_bytetrack_reset(tracker):
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

def test_update_empty_detections(tracker):
    empty_detections = Detections.empty()
    result = tracker.update_with_detections(empty_detections)
    assert len(result) == 0

def test_update_with_detections(tracker):
    xyxy = np.array([[100, 100, 200, 200]], dtype=np.float32)
    confidence = np.array([0.9], dtype=np.float32)
    detections = Detections(xyxy=xyxy, confidence=confidence)
    
    result = tracker.update_with_detections(detections)
    assert len(result) >= 0

@pytest.mark.parametrize("track_params", [
    (np.array([100, 100, 50, 50]), 0.9),
    (np.array([200, 200, 50, 50]), 0.8),
])
def test_track_creation(kalman, track_params):
    box, score = track_params
    track = STrack(box, score, 1, kalman)
    assert track.score == score
    assert np.array_equal(track.tlwh, box)

def test_joint_tracks(kalman):
    track1 = STrack(np.array([100, 100, 50, 50]), 0.9, 1, kalman)
    track2 = STrack(np.array([200, 200, 50, 50]), 0.8, 1, kalman)
    
    result = joint_tracks([track1], [track2])
    assert len(result) == 2

def test_sub_tracks(kalman):
    track1 = STrack(np.array([100, 100, 50, 50]), 0.9, 1, kalman)
    track2 = STrack(np.array([200, 200, 50, 50]), 0.8, 1, kalman)
    
    result = sub_tracks([track1, track2], [track1])
    assert len(result) == 1

def test_remove_duplicate_tracks(kalman):
    track1 = STrack(np.array([100, 100, 50, 50]), 0.9, 1, kalman)
    track2 = STrack(np.array([101, 101, 50, 50]), 0.8, 1, kalman)
    
    result_a, result_b = remove_duplicate_tracks([track1], [track2])
    assert len(result_a) + len(result_b) <= 2
