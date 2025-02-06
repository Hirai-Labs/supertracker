from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

@dataclass
class Detections:
    xyxy: np.ndarray
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.xyxy)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[float], Optional[int], Optional[int]]]:
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None
            )

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray]
    ) -> Detections:
        """
        Get a subset of the Detections object based on the provided index/indices.
        
        Args:
            index: Integer, slice, list of integers or numpy array for indexing
            
        Returns:
            Detections: A new Detections object containing the subset
        """
        if isinstance(index, (int, slice, list, np.ndarray)):
            return Detections(
                xyxy=self.xyxy[index],
                confidence=self.confidence[index] if self.confidence is not None else None, 
                class_id=self.class_id[index] if self.class_id is not None else None,
                tracker_id=self.tracker_id[index] if self.tracker_id is not None else None
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    @classmethod
    def empty(cls) -> Detections:
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )
