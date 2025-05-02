from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class Detections:
    """
    The Detections class standardizes results from
    various object detection and segmentation models into a consistent format.

    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        mask: (Optional[np.ndarray]): An array of shape
            `(n, H, W)` containing the segmentation masks (`bool` data type).
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the tracker ids of the detections.
        data (Dict[str, Union[np.ndarray, List]]): A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data.
    """

    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray]
    ) -> Detections:
        """
        Get a subset of the Detections object.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new Detections object that represents a subset of the original
        detections.

        Args:
            index (Union[int, slice, List[int], np.ndarray]): The index or indices
                to access a subset of the Detections.

        Returns:
            Detections: A subset of the Detections object.

        Raises:
            TypeError: If the index is not an integer, slice, list of integers, or numpy array.
        """
        if self.is_empty():
            return self
            
        # Validate index type
        if not isinstance(index, (int, slice, list, np.ndarray)):
            raise TypeError(f"Invalid index type: {type(index)}. Expected int, slice, list, or numpy.ndarray.")
            
        if isinstance(index, int):
            index = [index]
        
        return Detections(
            xyxy=self.xyxy[index],
            mask=self.mask[index] if self.mask is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            data={k: v[index] if isinstance(v, np.ndarray) else [v[i] for i in index] 
                  for k, v in self.data.items()} if self.data else {},
        )

    def __iter__(self):
        """
        Iterate through the detections, yielding tuples of (xyxy, confidence, class_id).
        """
        for i in range(len(self)):
            yield (self.xyxy[i], 
                   self.confidence[i] if self.confidence is not None else None,
                   self.class_id[i] if self.class_id is not None else None)

    @classmethod
    def empty(cls) -> Detections:
        """
        Create an empty Detections object with no bounding boxes,
            confidences, or class IDs.

        Returns:
            (Detections): An empty Detections object.
        """
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

    def is_empty(self) -> bool:
        """
        Returns `True` if the `Detections` object is considered empty.
        """
        return len(self) == 0