"""Tests for scoring primitives — only the functions that are wired up in V2.

`srgb_to_lab`, `delta_e2000_image`, and `hungarian_match_blocks` are intentionally
stubbed with NotImplementedError for MVP-A — those tests confirm the stub contract
without exercising the unimplemented math.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.eval.eval_result import SummaryStats
from tests.eval.scoring import (
    delta_e2000_image,
    hungarian_match_blocks,
    iou_per_mask,
    srgb_to_lab,
    summarize,
)


class TestSummarize:
    def test_constant_array_has_zero_spread(self) -> None:
        arr = np.full((10, 10), 1.0, dtype=np.float32)
        s = summarize(arr)
        assert s.mean == pytest.approx(1.0)
        assert s.p50 == pytest.approx(1.0)
        assert s.p95 == pytest.approx(1.0)
        assert s.p99 == pytest.approx(1.0)
        assert s.max == pytest.approx(1.0)

    def test_returns_summary_stats_instance(self) -> None:
        arr = np.zeros((4, 4), dtype=np.float32)
        s = summarize(arr)
        assert isinstance(s, SummaryStats)

    def test_linear_ramp_percentiles(self) -> None:
        arr = np.arange(0, 100, dtype=np.float32)  # 0..99
        s = summarize(arr)
        assert s.mean == pytest.approx(49.5)
        assert s.p50 == pytest.approx(49.5, abs=1.0)
        assert s.p95 == pytest.approx(94.05, abs=1.0)
        assert s.p99 == pytest.approx(98.01, abs=1.0)
        assert s.max == pytest.approx(99.0)

    def test_handles_2d_input(self) -> None:
        arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        s = summarize(arr)
        assert s.mean == pytest.approx(1.5)
        assert s.max == pytest.approx(3.0)

    def test_handles_3d_input(self) -> None:
        # H, W, C — should still flatten correctly
        arr = np.ones((3, 3, 3), dtype=np.float32) * 2.0
        s = summarize(arr)
        assert s.mean == pytest.approx(2.0)
        assert s.max == pytest.approx(2.0)

    def test_negative_values_allowed(self) -> None:
        arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        s = summarize(arr)
        assert s.mean == pytest.approx(0.0)
        assert s.max == pytest.approx(1.0)

    def test_returns_python_floats_not_numpy_scalars(self) -> None:
        # Critical for json.dumps — numpy floats break JSON serialization
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        s = summarize(arr)
        assert type(s.mean) is float
        assert type(s.p50) is float
        assert type(s.p95) is float
        assert type(s.p99) is float
        assert type(s.max) is float


class TestIouPerMask:
    def test_both_empty_returns_one(self) -> None:
        # Convention: empty / empty = perfect match (no false positives, no misses)
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        assert iou_per_mask(pred, gt) == 1.0

    def test_perfect_overlap(self) -> None:
        m = np.ones((10, 10), dtype=bool)
        assert iou_per_mask(m, m) == 1.0

    def test_no_overlap(self) -> None:
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        pred[:5, :] = True
        gt[5:, :] = True
        assert iou_per_mask(pred, gt) == 0.0

    def test_half_overlap(self) -> None:
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        pred[:, :5] = True  # left half
        gt[:5, :] = True  # top half
        # intersection = top-left 5x5 = 25 px
        # union = top half (50) + bottom-left (25) = 75 px
        assert iou_per_mask(pred, gt) == pytest.approx(25.0 / 75.0)

    def test_accepts_uint8_masks(self) -> None:
        # callers may pass uint8 0/1 — function must cast to bool
        pred = np.ones((4, 4), dtype=np.uint8)
        gt = np.ones((4, 4), dtype=np.uint8)
        assert iou_per_mask(pred, gt) == 1.0

    def test_accepts_float_masks(self) -> None:
        pred = np.ones((4, 4), dtype=np.float32)
        gt = np.ones((4, 4), dtype=np.float32)
        assert iou_per_mask(pred, gt) == 1.0

    def test_pred_subset_of_gt(self) -> None:
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        pred[:2, :2] = True  # 4 px
        gt[:5, :5] = True  # 25 px
        # intersection = 4, union = 25
        assert iou_per_mask(pred, gt) == pytest.approx(4.0 / 25.0)

    def test_returns_python_float(self) -> None:
        m = np.ones((4, 4), dtype=bool)
        result = iou_per_mask(m, m)
        assert type(result) is float


class TestStubContract:
    """The unimplemented stubs MUST raise NotImplementedError until MVP-A wires them."""

    def test_srgb_to_lab_raises(self) -> None:
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            srgb_to_lab(arr)

    def test_delta_e2000_image_raises(self) -> None:
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(NotImplementedError):
            delta_e2000_image(a, b)

    def test_hungarian_match_blocks_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            hungarian_match_blocks({}, {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
