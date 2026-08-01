"""D14.l — capture_swatch + fit_pigments real wiring."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    from backend.mcp import paths

    importlib.reload(paths)
    from backend.mcp.tools import calibration

    importlib.reload(calibration)


def _write_swatch_grid(p: Path) -> Path:
    """Build a 2x3 swatch grid with known colours: white, mid-gray, black on top;
    pure red, green, blue on bottom. Each cell 40x40, origin at (10, 10)."""
    img = np.zeros((100, 130, 3), dtype=np.uint8)
    palette = [
        [255, 255, 255],
        [128, 128, 128],
        [10, 10, 10],
        [200, 30, 30],
        [30, 200, 30],
        [30, 30, 200],
    ]
    for r in range(2):
        for c in range(3):
            idx = r * 3 + c
            x0, y0 = 10 + c * 40, 10 + r * 40
            img[y0 : y0 + 40, x0 : x0 + 40] = palette[idx]
    sw = p / "swatch.png"
    Image.fromarray(img, "RGB").save(sw)
    return sw


def test_capture_swatch_real_grid(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    sw = _write_swatch_grid(tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.capture_swatch(
        str(sw),
        layout={"origin_xy": [10, 10], "cell_px": [40, 40], "rows": 2, "cols": 3},
        colorchecker={
            "patches": [
                {"name": "white", "expected_lab": [100.0, 0.0, 0.0]},
                {"name": "gray_50", "expected_lab": [53.0, 0.0, 0.0]},
                {"name": "black", "expected_lab": [2.0, 0.0, 0.0]},
                {"name": "red", "expected_lab": [42.0, 65.0, 47.0]},
                {"name": "green", "expected_lab": [70.0, -70.0, 65.0]},
                {"name": "blue", "expected_lab": [25.0, 35.0, -80.0]},
            ]
        },
    )
    assert r.ok is True, r.errors
    assert r.data["detected_patches"] == 6
    # White patch should have L ~ 100
    white = next(p for p in r.data["patches"] if p["name"] == "white")
    assert white["mean_lab"][0] > 90.0
    # Persisted under calibrations dir
    cand_id = r.data["candidate_id"]
    assert cand_id.startswith("cal_candidate_")


def test_capture_swatch_invalid_layout(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    sw = _write_swatch_grid(tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.capture_swatch(str(sw), layout={"origin_xy": [0, 0]}, colorchecker={})
    assert r.ok is False
    assert r.errors[0].code == "INVALID_LAYOUT"


def test_fit_pigments_real(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    sw = _write_swatch_grid(tmp_path)
    from backend.mcp.tools import calibration

    cap = calibration.capture_swatch(
        str(sw),
        layout={"origin_xy": [10, 10], "cell_px": [40, 40], "rows": 2, "cols": 3},
        colorchecker={},
    )
    assert cap.ok is True
    cand_id = cap.data["candidate_id"]

    fit = calibration.fit_pigments(cand_id)
    assert fit.ok is True, fit.errors
    assert fit.data["pigments_fitted"] == 13
    assert isinstance(fit.data["swatch_fit_dE_median"], float)
    assert fit.data["calibration_id"].startswith("cal_fitted_")
    # All fit entries have delta_lab triples
    for f in fit.data["fits"]:
        assert len(f["delta_lab"]) == 3
        assert f["pigment_name"]


def test_fit_pigments_unknown_candidate(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.fit_pigments("cal_candidate_nope_999")
    assert r.ok is False
    assert r.errors[0].code == "CANDIDATE_NOT_FOUND"


def test_apply_then_list_real(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    sw = _write_swatch_grid(tmp_path)
    from backend.mcp.tools import calibration

    cap = calibration.capture_swatch(
        str(sw),
        layout={"origin_xy": [10, 10], "cell_px": [40, 40], "rows": 2, "cols": 3},
        colorchecker={},
    )
    fit = calibration.fit_pigments(cap.data["candidate_id"])
    cal_id = fit.data["calibration_id"]

    apply_r = calibration.apply_calibration(cal_id)
    assert apply_r.ok is True
    assert apply_r.data["applied"] is True

    lst = calibration.list_calibrations()
    assert lst.ok is True
    assert lst.data["active"] == cal_id
    assert any(c["calibration_id"] == cal_id for c in lst.data["calibrations"])
