from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from supertracker.detection.config import (
    CLASS_NAME_DATA_FIELD,
    ORIENTED_BOX_COORDINATES,
)

from supertracker.detection.utils import (
    get_data_item,
    is_data_equal,
    is_metadata_equal,
    mask_to_xyxy,
    validate_detections_fields
)

@dataclass
class Detections:
    """
    A class for storing and manipulating detection results in a standardized format.
    """
    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_detections_fields(
            xyxy=self.xyxy,
            mask=self.mask,
            confidence=self.confidence,
            class_id=self.class_id,
            tracker_id=self.tracker_id,
            data=self.data,
        )

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        """
        Iterates over the Detections object.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: Detections):
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                np.array_equal(self.mask, other.mask),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                np.array_equal(self.tracker_id, other.tracker_id),
                is_data_equal(self.data, other.data),
                is_metadata_equal(self.metadata, other.metadata),
            ]
        )

    @classmethod
    def from_yolo(cls, custom_results) -> Detections:
        """
        Creates a `sv.Detections` instance from a custom Results class that mimics
        the ultralytics Results format.

        Args:
            custom_results: The output Results instance from custom detector

        Returns:
            Detections: A new Detections object.
        """
        # Handle OBB case first if available
        if hasattr(custom_results, "obb") and custom_results.obb is not None:
            class_id = custom_results.obb.cls.astype(int)
            class_names = np.array([custom_results.names[i] for i in class_id])
            oriented_box_coordinates = custom_results.obb.xyxyxyxy
            
            return cls(
                xyxy=custom_results.obb.xyxy,
                confidence=custom_results.obb.conf,
                class_id=class_id,
                tracker_id=custom_results.obb.id.astype(int) if custom_results.obb.id is not None else None,
                data={
                    ORIENTED_BOX_COORDINATES: oriented_box_coordinates,
                    CLASS_NAME_DATA_FIELD: class_names,
                },
            )

        # Handle regular detection case
        if hasattr(custom_results, "boxes") and custom_results.boxes is not None:
            class_id = custom_results.boxes.cls.astype(int)
            class_names = np.array([custom_results.names[i] for i in class_id])
            
            return cls(
                xyxy=custom_results.boxes.xyxy,
                confidence=custom_results.boxes.conf,
                class_id=class_id,
                mask=custom_results.masks.data if hasattr(custom_results, "masks") and custom_results.masks is not None else None,
                tracker_id=custom_results.boxes.id.astype(int) if hasattr(custom_results.boxes, "id") and custom_results.boxes.id is not None else None,
                data={CLASS_NAME_DATA_FIELD: class_names},
            )
            
        # Handle case with only masks
        if hasattr(custom_results, "masks") and custom_results.masks is not None:
            masks = custom_results.masks.data
            return cls(
                xyxy=mask_to_xyxy(masks),
                mask=masks,
                class_id=np.arange(len(custom_results)),
            )

        # Return empty detections if no supported format is found
        return cls.empty()

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
        empty_detections = Detections.empty()
        empty_detections.data = self.data
        empty_detections.metadata = self.metadata
        return self == empty_detections

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray, str]
    ) -> Union[Detections, List, np.ndarray, None]:
        """
        Get a subset of the Detections object or access an item from its data field.
        """
        if isinstance(index, str):
            return self.data.get(index)
        if self.is_empty():
            return self
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            mask=self.mask[index] if self.mask is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            data=get_data_item(self.data, index),
            metadata=self.metadata,
        )

    def __setitem__(self, key: str, value: Union[np.ndarray, List]):
        """
        Set a value in the data dictionary of the Detections object.
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value
