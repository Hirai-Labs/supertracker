from typing import Any, Dict, List, Union

import numpy as np

def validate_detections_fields(
    xyxy: Any,
    mask: Any,
    confidence: Any,
    class_id: Any,
    tracker_id: Any,
    data: Dict[str, Any],
) -> None:
    """Simplified validation for detection fields"""
    if not isinstance(xyxy, np.ndarray) or xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError("xyxy must be a 2D numpy array with shape (N, 4)")
    
    n = len(xyxy)
    
    # Validate mask if provided
    if mask is not None and (not isinstance(mask, np.ndarray) or mask.ndim != 3 or mask.shape[0] != n):
        raise ValueError(f"mask must be a 3D numpy array with shape ({n}, H, W)")
    
    # Validate confidence if provided
    if confidence is not None and (not isinstance(confidence, np.ndarray) or confidence.shape != (n,)):
        raise ValueError(f"confidence must be a 1D numpy array with shape ({n},)")
    
    # Validate class_id if provided
    if class_id is not None and (not isinstance(class_id, np.ndarray) or class_id.shape != (n,)):
        raise ValueError(f"class_id must be a 1D numpy array with shape ({n},)")
    
    # Validate tracker_id if provided
    if tracker_id is not None and (not isinstance(tracker_id, np.ndarray) or tracker_id.shape != (n,)):
        raise ValueError(f"tracker_id must be a 1D numpy array with shape ({n},)")
    
    # Validate data dictionary
    if data:
        for key, value in data.items():
            if isinstance(value, list):
                if len(value) != n:
                    raise ValueError(f"Length of list for key '{key}' must be {n}")
            elif isinstance(value, np.ndarray):
                if value.ndim == 1 and value.shape[0] != n:
                    raise ValueError(f"Shape of np.ndarray for key '{key}' must be ({n},)")
                elif value.ndim > 1 and value.shape[0] != n:
                    raise ValueError(f"First dimension of np.ndarray for key '{key}' must have size {n}")
            else:
                raise ValueError(f"Value for key '{key}' must be a list or np.ndarray")


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `boxes_true` and `boxes_detection`. Both sets
        of boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.

    Args:
        boxes_true (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        boxes_detection (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `boxes_true` and `boxes_detection`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    ious = area_inter / (area_true[:, None] + area_detection - area_inter)
    ious = np.nan_to_num(ious)
    return ious


def mask_to_xyxy(masks: np.ndarray) -> np.ndarray:
    """
    Converts a 3D `np.array` of 2D bool masks into a 2D `np.array` of bounding boxes.

    Parameters:
        masks (np.ndarray): A 3D `np.array` of shape `(N, W, H)`
            containing 2D bool masks

    Returns:
        np.ndarray: A 2D `np.array` of shape `(N, 4)` containing the bounding boxes
            `(x_min, y_min, x_max, y_max)` for each mask
    """
    n = masks.shape[0]
    xyxy = np.zeros((n, 4), dtype=int)

    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)

        if len(rows) > 0 and len(cols) > 0:
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
            xyxy[i, :] = [x_min, y_min, x_max, y_max]

    return xyxy


def is_data_equal(data_a: Dict[str, np.ndarray], data_b: Dict[str, np.ndarray]) -> bool:
    """
    Compares the data payloads of two Detections instances.

    Args:
        data_a, data_b: The data payloads of the instances.

    Returns:
        True if the data payloads are equal, False otherwise.
    """
    return set(data_a.keys()) == set(data_b.keys()) and all(
        np.array_equal(data_a[key], data_b[key]) for key in data_a
    )


def is_metadata_equal(metadata_a: Dict[str, Any], metadata_b: Dict[str, Any]) -> bool:
    """
    Compares the metadata payloads of two Detections instances.

    Args:
        metadata_a, metadata_b: The metadata payloads of the instances.

    Returns:
        True if the metadata payloads are equal, False otherwise.
    """
    return set(metadata_a.keys()) == set(metadata_b.keys()) and all(
        np.array_equal(metadata_a[key], metadata_b[key])
        if (
            isinstance(metadata_a[key], np.ndarray)
            and isinstance(metadata_b[key], np.ndarray)
        )
        else metadata_a[key] == metadata_b[key]
        for key in metadata_a
    )


def get_data_item(
    data: Dict[str, Union[np.ndarray, List]],
    index: Union[int, slice, List[int], np.ndarray],
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Retrieve a subset of the data dictionary based on the given index.

    Args:
        data: The data dictionary of the Detections object.
        index: The index or indices specifying the subset to retrieve.

    Returns:
        A subset of the data dictionary corresponding to the specified index.
    """
    subset_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            subset_data[key] = value[index]
        elif isinstance(value, list):
            if isinstance(index, slice):
                subset_data[key] = value[index]
            elif isinstance(index, list):
                subset_data[key] = [value[i] for i in index]
            elif isinstance(index, np.ndarray):
                if index.dtype == bool:
                    subset_data[key] = [
                        value[i] for i, index_value in enumerate(index) if index_value
                    ]
                else:
                    subset_data[key] = [value[i] for i in index]
            elif isinstance(index, int):
                subset_data[key] = [value[index]]
            else:
                raise TypeError(f"Unsupported index type: {type(index)}")
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

    return subset_data