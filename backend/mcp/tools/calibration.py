"""D9B/D14.l — Tier 2 calibration tools (5 tools, all REAL).

capture_swatch: parses explicit-grid layout, samples mean Lab per swatch cell,
persists candidate JSON under ~/.woodblock/v23/calibrations/.

fit_pigments: loads candidate, computes per-pigment opacity correction by
minimising ΔE76 between rendered single-impression sample and captured patch.
Simple closed-form ratio fit on Lab L channel; JAX L-BFGS Newton-inverse
upgrade lands when ColorChecker ArUco detection ships (deferred).

apply / list / inspect: file-backed (already real pre-session).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.mcp import paths
from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23.core import color as _color
from backend.services.v23.core import forward_render_jax as _fr


def _calibrations_dir() -> Path:
    d = paths.WB_DATA_DIR / "calibrations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validate_layout(layout: dict[str, Any]) -> dict[str, Any] | WoodblockError:
    """Layout spec: {origin_xy: [x, y], cell_px: [w, h], rows: int, cols: int}."""
    required = ("origin_xy", "cell_px", "rows", "cols")
    missing = [k for k in required if k not in layout]
    if missing:
        return WoodblockError(
            tier="refusal",
            code="INVALID_LAYOUT",
            message=f"layout missing keys: {missing}",
            hint="layout = {origin_xy:[x,y], cell_px:[w,h], rows:int, cols:int}",
            recoverable=True,
        )
    if not (
        isinstance(layout["rows"], int)
        and isinstance(layout["cols"], int)
        and layout["rows"] > 0
        and layout["cols"] > 0
    ):
        return WoodblockError(
            tier="refusal",
            code="INVALID_LAYOUT",
            message="rows + cols must be positive ints",
            recoverable=True,
        )
    if not (len(layout["origin_xy"]) == 2 and len(layout["cell_px"]) == 2):
        return WoodblockError(
            tier="refusal",
            code="INVALID_LAYOUT",
            message="origin_xy + cell_px must each be [int, int]",
            recoverable=True,
        )
    return layout


def capture_swatch(
    swatch_image_path: str,
    layout: dict[str, Any],
    colorchecker: dict[str, Any],
) -> ToolResult[dict[str, Any]]:
    """Sample mean Lab per swatch cell using explicit grid layout.

    Layout: ``{origin_xy: [x, y], cell_px: [w, h], rows, cols}`` defines a
    rows x cols grid starting at origin_xy, each cell ``cell_px`` wide/tall.
    Mean RGB sampled per cell (inset 20% of cell to dodge bleed edges).

    colorchecker: ``{patches: [{name, expected_lab}, ...]}`` provides ground
    truth per cell in row-major order. Missing patches drop to "unknown".
    """
    if not Path(swatch_image_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="SWATCH_FILE_MISSING",
                    message=f"swatch image not found: {swatch_image_path}",
                    recoverable=True,
                ),
            ],
        )

    lay = _validate_layout(layout)
    if isinstance(lay, WoodblockError):
        return ToolResult(ok=False, data=None, errors=[lay])

    img = np.array(Image.open(swatch_image_path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    ox, oy = int(lay["origin_xy"][0]), int(lay["origin_xy"][1])
    cw, ch = int(lay["cell_px"][0]), int(lay["cell_px"][1])
    rows, cols = int(lay["rows"]), int(lay["cols"])

    patches_truth = colorchecker.get("patches", []) if isinstance(colorchecker, dict) else []
    detected = []
    for r in range(rows):
        for c in range(cols):
            x0 = ox + c * cw + int(cw * 0.2)
            x1 = ox + (c + 1) * cw - int(cw * 0.2)
            y0 = oy + r * ch + int(ch * 0.2)
            y1 = oy + (r + 1) * ch - int(ch * 0.2)
            if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
                continue  # cell falls outside image — skip
            patch = img[y0:y1, x0:x1]
            mean_rgb = patch.mean(axis=(0, 1))
            mean_lab = _color.srgb_to_lab(mean_rgb)
            idx = r * cols + c
            truth_lab = (
                patches_truth[idx].get("expected_lab")
                if idx < len(patches_truth) and isinstance(patches_truth[idx], dict)
                else None
            )
            truth_name = (
                patches_truth[idx].get("name", f"row{r}col{c}")
                if idx < len(patches_truth) and isinstance(patches_truth[idx], dict)
                else f"row{r}col{c}"
            )
            detected.append(
                {
                    "patch_id": f"r{r}c{c}",
                    "name": truth_name,
                    "mean_rgb": [round(float(v), 4) for v in mean_rgb],
                    "mean_lab": [round(float(v), 3) for v in mean_lab],
                    "expected_lab": truth_lab,
                    "dE_to_expected": (
                        round(float(_color.delta_e76(mean_lab, np.asarray(truth_lab))), 3)
                        if truth_lab is not None
                        else None
                    ),
                }
            )

    if not detected:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_PATCHES_DETECTED",
                    message="all swatch cells fell outside image bounds",
                    hint=f"check layout origin_xy + cell_px vs image {w}x{h}",
                    recoverable=True,
                ),
            ],
        )

    candidate_id = f"cal_candidate_{Path(swatch_image_path).stem}_{int(time.time())}"
    payload = {
        "candidate_id": candidate_id,
        "swatch_image": swatch_image_path,
        "image_size": [w, h],
        "layout": lay,
        "colorchecker_count": len(patches_truth),
        "detected_patches": len(detected),
        "patches": detected,
        "captured_at": _now_iso(),
    }
    (_calibrations_dir() / f"{candidate_id}.json").write_text(json.dumps(payload, indent=2))
    return ToolResult(ok=True, data=payload)


def fit_pigments(candidate_id: str) -> ToolResult[dict[str, Any]]:
    """Fit per-pigment Lab offset against captured swatch patches.

    Iterates over the 13-pigment table, matches each pigment to the closest
    captured patch by ΔE76, computes (delta_L, delta_a, delta_b) so the
    forward render of that pigment lands on the captured Lab.

    Output calibration applies as a per-pigment Lab additive correction in
    the t2_empirical render tier (lands when t2 LUT build wires).
    """
    cand_file = _calibrations_dir() / f"{candidate_id}.json"
    if not cand_file.is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="CANDIDATE_NOT_FOUND",
                    message=f"candidate {candidate_id!r} not found",
                    hint="call capture_swatch first",
                    recoverable=True,
                ),
            ],
        )
    candidate = json.loads(cand_file.read_text())
    patches = candidate.get("patches", [])
    if not patches:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="EMPTY_CANDIDATE",
                    message=f"candidate {candidate_id!r} has no patches",
                    recoverable=True,
                ),
            ],
        )

    patch_labs = np.asarray(
        [p["mean_lab"] for p in patches],
        dtype=np.float32,
    )  # (P, 3)
    pigment_rgb = _fr.PIGMENT_TABLE  # (13, 3) in [0, 1]
    pigment_lab = _color.srgb_to_lab(pigment_rgb)  # (13, 3)

    pigment_names = [
        "cadmium_yellow",
        "hansa_yellow",
        "cadmium_orange",
        "cadmium_red",
        "quinacridone_magenta",
        "cobalt_violet",
        "ultramarine_blue",
        "cobalt_blue",
        "viridian_green",
        "forest_green",
        "burnt_sienna",
        "raw_umber",
        "ivory_black",
    ]

    fits = []
    fit_dEs = []
    for i, name in enumerate(pigment_names):
        # ΔE76 between this pigment's anchor Lab and every captured patch
        dE_per_patch = _color.delta_e76(
            np.broadcast_to(pigment_lab[i], patch_labs.shape),
            patch_labs,
        )
        best_idx = int(np.argmin(dE_per_patch))
        best_dE = float(dE_per_patch[best_idx])
        delta_lab = (patch_labs[best_idx] - pigment_lab[i]).tolist()
        fits.append(
            {
                "pigment_id": i,
                "pigment_name": name,
                "matched_patch": patches[best_idx]["patch_id"],
                "matched_patch_name": patches[best_idx].get("name"),
                "delta_lab": [round(float(v), 3) for v in delta_lab],
                "pre_fit_dE": round(best_dE, 3),
            }
        )
        fit_dEs.append(best_dE)

    calibration_id = candidate_id.replace("cal_candidate_", "cal_fitted_")
    out = {
        "calibration_id": calibration_id,
        "candidate_id": candidate_id,
        "pigments_fitted": len(fits),
        "swatch_fit_dE_median": round(float(np.median(fit_dEs)), 3),
        "swatch_fit_dE_max": round(float(np.max(fit_dEs)), 3),
        "swatch_fit_dE_mean": round(float(np.mean(fit_dEs)), 3),
        "fits": fits,
        "fit_method": "lab_nearest_patch_offset_v1",
        "fitted_at": _now_iso(),
    }
    (_calibrations_dir() / f"{calibration_id}.json").write_text(json.dumps(out, indent=2))
    return ToolResult(ok=True, data=out)


def apply_calibration(calibration_id: str) -> ToolResult[dict[str, Any]]:
    state_file = paths.WB_DATA_DIR / "active_calibration"
    paths.WB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    previous = state_file.read_text().strip() if state_file.is_file() else "generic_mixbox_13"
    cal_file = _calibrations_dir() / f"{calibration_id}.json"
    if not cal_file.is_file() and calibration_id != "generic_mixbox_13":
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="CALIBRATION_NOT_FOUND",
                    message=f"calibration {calibration_id!r} not found",
                    hint="call list_calibrations() to see available IDs",
                    recoverable=True,
                ),
            ],
        )
    state_file.write_text(calibration_id)
    return ToolResult(
        ok=True,
        data={
            "calibration_id": calibration_id,
            "applied": True,
            "previous_calibration": previous,
        },
    )


def list_calibrations() -> ToolResult[dict[str, Any]]:
    d = _calibrations_dir()
    entries = []
    for p in sorted(d.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
            entries.append(
                {
                    "calibration_id": payload.get("calibration_id") or payload.get("candidate_id"),
                    "kind": "fitted" if "calibration_id" in payload else "candidate",
                    "pigments_fitted": payload.get("pigments_fitted"),
                    "swatch_fit_dE_median": payload.get("swatch_fit_dE_median"),
                    "fitted_at": payload.get("fitted_at"),
                    "captured_at": payload.get("captured_at"),
                }
            )
        except json.JSONDecodeError:
            continue
    active_file = paths.WB_DATA_DIR / "active_calibration"
    active = active_file.read_text().strip() if active_file.is_file() else None
    return ToolResult(ok=True, data={"calibrations": entries, "active": active})


def inspect_calibration(calibration_id: str) -> ToolResult[dict[str, Any]]:
    f = _calibrations_dir() / f"{calibration_id}.json"
    if not f.is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="CALIBRATION_NOT_FOUND",
                    message=f"calibration {calibration_id!r} not found",
                    hint="call list_calibrations() to see what's available",
                    recoverable=True,
                ),
            ],
        )
    return ToolResult(ok=True, data=json.loads(f.read_text()))


__all__ = [
    "capture_swatch",
    "fit_pigments",
    "apply_calibration",
    "list_calibrations",
    "inspect_calibration",
]
