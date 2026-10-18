from __future__ import annotations

import numpy as np


def farthest_point_sample_positions(points: np.ndarray, num_samples: int) -> list[int]:
    """Production contract for the three fixed novel-camera anchors."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected (N,D) camera centers, got {points.shape}")
    if points.shape[0] == 0 or num_samples == 0:
        return []
    target = min(int(num_samples), int(points.shape[0]))
    centroid = np.mean(points, axis=0, dtype=np.float32)
    selected = [int(np.argmax(np.linalg.norm(points - centroid[None], axis=1)))]
    minimum = np.linalg.norm(points - points[selected[0]][None], axis=1)
    minimum[selected[0]] = -np.inf
    while len(selected) < target:
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(points - points[index][None], axis=1))
        minimum[np.asarray(selected, dtype=np.int64)] = -np.inf
    return sorted(selected)
