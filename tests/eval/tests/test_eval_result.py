"""Tests for the unified EvalResult schema.

Covers JSON roundtrip, the is_pass threshold rule, and default field behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from tests.eval.eval_result import EvalResult, SummaryStats


def _make_stats(
    mean: float = 1.0,
    p50: float = 1.0,
    p95: float = 2.0,
    p99: float = 2.5,
    max_: float = 3.0,
) -> SummaryStats:
    return SummaryStats(mean=mean, p50=p50, p95=p95, p99=p99, max=max_)


def _make_result(**kwargs) -> EvalResult:
    defaults = dict(
        fixture_id="test_fix_001",
        image_path="corpus/test_fix_001/original.png",
        recon_path="recon/test_fix_001.png",
        dE_heatmap_path="recon/test_fix_001_dE.png",
        dE2000=_make_stats(),
    )
    defaults.update(kwargs)
    return EvalResult(**defaults)


class TestSummaryStats:
    def test_construct_with_all_fields(self) -> None:
        s = SummaryStats(mean=0.5, p50=0.4, p95=1.2, p99=1.8, max=2.5)
        assert s.mean == 0.5
        assert s.p50 == 0.4
        assert s.p95 == 1.2
        assert s.p99 == 1.8
        assert s.max == 2.5

    def test_asdict_roundtrip(self) -> None:
        s = _make_stats(mean=0.7, p95=1.4)
        d = asdict(s)
        assert d == {"mean": 0.7, "p50": 1.0, "p95": 1.4, "p99": 2.5, "max": 3.0}


class TestEvalResultDefaults:
    def test_has_required_fields(self) -> None:
        r = _make_result()
        assert r.fixture_id == "test_fix_001"
        assert r.image_path == "corpus/test_fix_001/original.png"

    def test_defaults_applied(self) -> None:
        r = _make_result()
        assert r.pigment_iou is None
        assert r.block_iou is None
        assert r.chromatic_class_recovery is None
        assert r.block_count == 0
        assert r.pigment_count == 0
        assert r.print_order == []
        assert r.duration_ms == 0
        assert r.git_sha == ""
        assert r.engine == "stub"
        assert r.params == {}
        assert r.passed is False

    def test_timestamp_is_iso_string(self) -> None:
        r = _make_result()
        assert isinstance(r.timestamp, str)
        # ISO-8601: starts with year-month-day
        assert r.timestamp[:4].isdigit()
        assert r.timestamp[4] == "-"

    def test_default_lists_are_independent(self) -> None:
        # default_factory must not share state across instances
        a = _make_result(fixture_id="a")
        b = _make_result(fixture_id="b")
        a.print_order.append("red")
        assert b.print_order == []
        a.params["foo"] = 1
        assert b.params == {}


class TestIsPassThreshold:
    def test_passes_when_under_thresholds(self) -> None:
        r = _make_result(dE2000=_make_stats(mean=1.0, p95=2.0))
        assert r.is_pass is True

    def test_fails_when_mean_at_threshold(self) -> None:
        # threshold is strict <1.5
        r = _make_result(dE2000=_make_stats(mean=1.5, p95=2.0))
        assert r.is_pass is False

    def test_fails_when_mean_above_threshold(self) -> None:
        r = _make_result(dE2000=_make_stats(mean=1.6, p95=2.0))
        assert r.is_pass is False

    def test_fails_when_p95_at_threshold(self) -> None:
        r = _make_result(dE2000=_make_stats(mean=1.0, p95=3.0))
        assert r.is_pass is False

    def test_fails_when_p95_above_threshold(self) -> None:
        r = _make_result(dE2000=_make_stats(mean=1.0, p95=3.1))
        assert r.is_pass is False

    def test_passes_at_exact_lower_bound(self) -> None:
        r = _make_result(dE2000=_make_stats(mean=0.0, p95=0.0))
        assert r.is_pass is True


class TestJsonRoundtrip:
    def test_to_json_returns_string(self) -> None:
        r = _make_result()
        s = r.to_json()
        assert isinstance(s, str)
        # must be valid JSON
        parsed = json.loads(s)
        assert parsed["fixture_id"] == "test_fix_001"

    def test_from_json_reconstructs_equal_object(self) -> None:
        original = _make_result(
            block_count=4,
            pigment_count=5,
            print_order=["yellow", "red", "blue"],
            engine="tan",
            params={"k": 4, "smoothing": 0.1},
            passed=True,
        )
        s = original.to_json()
        restored = EvalResult.from_json(s)
        assert restored.fixture_id == original.fixture_id
        assert restored.block_count == original.block_count
        assert restored.pigment_count == original.pigment_count
        assert restored.print_order == original.print_order
        assert restored.engine == original.engine
        assert restored.params == original.params
        assert restored.passed == original.passed

    def test_summary_stats_survives_roundtrip(self) -> None:
        original = _make_result(dE2000=_make_stats(mean=0.85, p50=0.7, p95=2.1, p99=2.8, max_=4.0))
        restored = EvalResult.from_json(original.to_json())
        assert isinstance(restored.dE2000, SummaryStats)
        assert restored.dE2000.mean == 0.85
        assert restored.dE2000.p95 == 2.1
        assert restored.dE2000.max == 4.0

    def test_engine_literal_values_accepted(self) -> None:
        for engine in ("tan", "km_nnls", "qwen_layered", "stub"):
            r = _make_result(engine=engine)  # type: ignore[arg-type]
            restored = EvalResult.from_json(r.to_json())
            assert restored.engine == engine


class TestSchemaShape:
    """Guard against accidental field renames that would break downstream consumers."""

    def test_serialized_keys_match_contract(self) -> None:
        r = _make_result()
        parsed = json.loads(r.to_json())
        expected_keys = {
            "fixture_id",
            "image_path",
            "recon_path",
            "dE_heatmap_path",
            "dE2000",
            "pigment_iou",
            "block_iou",
            "chromatic_class_recovery",
            "block_count",
            "pigment_count",
            "print_order",
            "duration_ms",
            "git_sha",
            "engine",
            "params",
            "passed",
            "timestamp",
        }
        assert set(parsed.keys()) == expected_keys

    def test_dE2000_keys_match_contract(self) -> None:
        r = _make_result()
        parsed = json.loads(r.to_json())
        assert set(parsed["dE2000"].keys()) == {"mean", "p50", "p95", "p99", "max"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
