from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def union_mask_sequences(
    combined: Sequence[np.ndarray] | None, incoming: Sequence[np.ndarray]
) -> list[np.ndarray]:
    incoming_bool = [np.asarray(mask).astype(bool) for mask in incoming]
    if combined is None:
        return [mask.copy() for mask in incoming_bool]
    if len(combined) != len(incoming_bool):
        raise ValueError(
            f"Semantic mask sequences have different lengths: {len(combined)} != {len(incoming_bool)}"
        )
    result = []
    for current, new in zip(combined, incoming_bool):
        current_bool = np.asarray(current).astype(bool)
        if current_bool.shape != new.shape:
            raise ValueError(
                f"Semantic masks have different shapes: {current_bool.shape} != {new.shape}"
            )
        result.append(current_bool | new)
    return result


def mask_pixel_count(masks: Sequence[np.ndarray] | None) -> int:
    return 0 if masks is None else sum(int(np.asarray(mask).astype(bool).sum()) for mask in masks)


def should_use_flow_fallback(
    masks: Sequence[np.ndarray] | None, *, fallback_enabled: bool
) -> bool:
    return bool(fallback_enabled and mask_pixel_count(masks) == 0)
