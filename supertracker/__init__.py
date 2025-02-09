"""Top-level package for supertracker."""

__author__ = """Syahvan Alviansyah"""
__email__ = "syahvanalviansyah91@gmail.com"
__version__ = "0.0.3"

from supertracker.detection.core import Detections
from supertracker.detection.utils import box_iou_batch
from supertracker.bytetrack import matching
from supertracker.bytetrack.kalman_filter import KalmanFilter
from supertracker.bytetrack.single_object_track import STrack, TrackState
from supertracker.bytetrack.utils import IdCounter
from supertracker.bytetrack.core import ByteTrack
