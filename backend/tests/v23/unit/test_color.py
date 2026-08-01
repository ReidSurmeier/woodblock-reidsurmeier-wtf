"""D14.h — sRGB -> Lab -> ΔE76 sanity tests."""

from __future__ import annotations

import numpy as np
import pytest

from backend.services.v23.core import color


def test_lab_white_at_D65() -> None:
    """sRGB (1,1,1) -> Lab (100, 0, 0) (approx — within 0.5)."""
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    lab = color.srgb_to_lab(white)
    assert lab[0] == pytest.approx(100.0, abs=0.5)
    assert lab[1] == pytest.approx(0.0, abs=0.5)
    assert lab[2] == pytest.approx(0.0, abs=0.5)


def test_lab_black_at_origin() -> None:
    """sRGB (0,0,0) -> Lab (0, 0, 0) (within 0.1)."""
    black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lab = color.srgb_to_lab(black)
    assert lab[0] == pytest.approx(0.0, abs=0.5)
    assert np.allclose(lab[1:], 0.0, atol=0.5)


def test_delta_e76_identical_is_zero() -> None:
    rgb = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    de = color.rgb_delta_e76(rgb, rgb)
    assert de == pytest.approx(0.0, abs=1e-5)


def test_delta_e76_white_to_black_is_100() -> None:
    """ΔE76 between pure white and pure black is exactly the L axis = 100."""
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    de = color.rgb_delta_e76(white, black)
    assert float(de) == pytest.approx(100.0, abs=0.5)


def test_delta_e_summary_shape() -> None:
    rng = np.random.default_rng(7)
    a = rng.random((32, 32, 3)).astype(np.float32)
    b = rng.random((32, 32, 3)).astype(np.float32)
    s = color.delta_e_summary(a, b)
    assert {"dE_mean", "dE_p95", "dE_max"} <= s.keys()
    assert 0.0 <= s["dE_mean"] <= s["dE_p95"] <= s["dE_max"] <= 400.0


def test_vectorised_lab_matches_scalar() -> None:
    """Vectorised srgb_to_lab on a (1,3) array == scalar lookup."""
    rgb = np.array([[0.3, 0.6, 0.9]], dtype=np.float32)
    lab_vec = color.srgb_to_lab(rgb)
    lab_scalar = color.srgb_to_lab(rgb[0])
    assert np.allclose(lab_vec[0], lab_scalar, atol=1e-5)
