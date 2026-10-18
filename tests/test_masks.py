from __future__ import annotations

import numpy as np

from wfm.masks import mask_pixel_count, should_use_flow_fallback, union_mask_sequences


def test_semantic_class_masks_are_unioned_across_all_frames() -> None:
    human = [np.array([[1, 0], [0, 0]]), np.zeros((2, 2), dtype=np.uint8)]
    robot = [np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])]
    combined = union_mask_sequences(None, human)
    combined = union_mask_sequences(combined, robot)
    assert mask_pixel_count(combined) == 3
    assert combined[0].tolist() == [[True, True], [False, False]]


def test_waft_fallback_only_fires_for_completely_empty_union() -> None:
    empty = [np.zeros((2, 2), dtype=bool) for _ in range(3)]
    assert should_use_flow_fallback(empty, fallback_enabled=True)
    assert not should_use_flow_fallback(empty, fallback_enabled=False)
    nonempty = union_mask_sequences(empty, [np.eye(2, dtype=bool)] * 3)
    assert not should_use_flow_fallback(nonempty, fallback_enabled=True)
