from __future__ import annotations

import numpy as np

from wfm.camera import farthest_point_sample_positions


def test_farthest_point_sampling_is_deterministic() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    assert farthest_point_sample_positions(points, 3) == [0, 1, 3]
    assert farthest_point_sample_positions(points, 3) == [0, 1, 3]


def test_farthest_point_sampling_caps_count() -> None:
    points = np.asarray([[0, 0], [1, 0]], dtype=np.float32)
    assert farthest_point_sample_positions(points, 9) == [0, 1]
