import unittest
import numpy as np
from supertracker.detection.core import Detections

class TestDetections(unittest.TestCase):
    def test_empty_detections(self):
        detections = Detections.empty()
        self.assertEqual(len(detections), 0)
        self.assertEqual(detections.xyxy.shape, (0, 4))
        
    def test_detections_creation(self):
        xyxy = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        confidence = np.array([0.9, 0.8])
        detections = Detections(xyxy=xyxy, confidence=confidence)
        self.assertEqual(len(detections), 2)
