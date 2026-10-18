from __future__ import annotations

import sys
from pathlib import Path


VENDORED_MOSCA = Path(__file__).resolve().parents[1] / "vendor" / "mosca"
sys.path.insert(0, str(VENDORED_MOSCA))

from lib_gen.scene_benchmark import build_chunk_ranges  # noqa: E402


def test_overlapping_tail_chunk_policy_preserves_every_frame_once() -> None:
    ranges = build_chunk_ranges(total=150, chunk_size=73)
    assert ranges == [
        (0, 73, 0, 73),
        (73, 146, 0, 73),
        (77, 150, 69, 73),
    ]
    delivered = []
    for start, _, keep_start, keep_end in ranges:
        delivered.extend(range(start + keep_start, start + keep_end))
    assert delivered == list(range(150))
