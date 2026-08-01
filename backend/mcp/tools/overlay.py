"""D9B — Tier 6 overlay tools (4 tools, addendum-v4 render tier dispatch)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.mcp import paths
from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23.core import color as _color
from backend.services.v23.core import forward_render_jax as _fr
from backend.services.v23.core import render_tier as _rt

_PIGMENT_NAMES = [
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


def _calibrations_dir() -> Path:
    d = paths.WB_DATA_DIR / "calibrations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lut_path() -> Path:
    return _calibrations_dir() / "empirical_lut.npz"


def _image_bytes_to_rgb(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _parse_pigment(value: str) -> int:
    raw = value.strip().lower()
    if raw.isdigit():
        idx = int(raw)
    else:
        idx = _PIGMENT_NAMES.index(raw)
    if idx < 0 or idx >= len(_PIGMENT_NAMES):
        raise ValueError(f"pigment index out of range: {value!r}")
    return idx


def _parse_rgb(row: dict[str, str]) -> list[float]:
    if {"r", "g", "b"} <= set(row):
        vals = [float(row["r"]), float(row["g"]), float(row["b"])]
    elif {"measured_r", "measured_g", "measured_b"} <= set(row):
        vals = [float(row["measured_r"]), float(row["measured_g"]), float(row["measured_b"])]
    else:
        raw = row.get("rgb", "").strip().strip("[]()")
        if raw.startswith("#") and len(raw) == 7:
            vals = [int(raw[i : i + 2], 16) for i in (1, 3, 5)]
        else:
            parts = raw.replace(";", " ").replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError("row must include rgb, r/g/b, or measured_r/g/b")
            vals = [float(v) for v in parts]
    if max(vals) > 1.0:
        vals = [v / 255.0 for v in vals]
    return [float(np.clip(v, 0.0, 1.0)) for v in vals]


def _load_empirical_lut() -> dict[str, Any] | None:
    p = _lut_path()
    if not p.is_file():
        return None
    data = np.load(p, allow_pickle=False)
    return {
        "path": str(p),
        "rows": int(data["base_idx"].shape[0]),
        "mean_bias": data["mean_bias"].astype(np.float32),
        "calibration_id": str(data["calibration_id"]),
    }


def _apply_t2_empirical(t1_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
    lut = _load_empirical_lut()
    if lut is None or lut["rows"] == 0:
        return t1_rgb, lut
    corrected = np.clip(t1_rgb + lut["mean_bias"][None, None, :], 0.0, 1.0)
    return corrected.astype(np.float32), lut


def simulate_overprint(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Run the chosen render tier for a persisted plan."""
    from backend.services.v23 import orchestrator as _orch
    from backend.services.v23.stages import s10_emit

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        tier = get_render_tier().data["tier"]
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "tier": tier, "composite_path": None, "dE_map_path": None},
            errors=[exc.error],
        )

    tier_info = _choose_tier_for_plan(len(plan.impressions))
    tier = tier_info["tier"]
    errors = list(tier_info["errors"])
    plan_dir = _orch._plan_dir(plan.session_id, plan.plan_id)
    t1_rgb = _image_bytes_to_rgb(s10_emit._render_composite(plan))
    out_rgb = t1_rgb
    lut_info = None
    if tier == "t2_empirical":
        out_rgb, lut_info = _apply_t2_empirical(t1_rgb)
    composite_path = plan_dir / f"composite_overprint_{tier}.png"
    _write_rgb(composite_path, out_rgb)
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "tier": tier,
            "composite_path": str(composite_path),
            "dE_map_path": None,
            "reconstruction_dE_mean": plan.reconstruction_dE_mean,
            "empirical_lut": lut_info,
            "render_tier_note": (
                "T1 renders as if pigments were pre-mixed in a well. T2 applies "
                "the active swatch-matrix empirical correction when available."
            ),
        },
        errors=errors,
    )


def _choose_tier_for_plan(stack_depth: int) -> dict[str, Any]:
    cal_file = paths.WB_DATA_DIR / "active_calibration"
    cal_id = cal_file.read_text().strip() if cal_file.is_file() else None
    ctx = _rt.RenderTierContext(
        calibration_id=cal_id,
        empirical_lut_available=_lut_path().is_file(),
        spectral_ks_available=(_calibrations_dir() / "spectral_ks.npz").is_file(),
        stack_depth=stack_depth,
    )
    tier = _rt.choose_render_tier(ctx)
    errors: list[WoodblockError] = []
    if tier == "t3_spectral":
        tier = "t2_empirical" if ctx.empirical_lut_available else "t1_mixbox"
        errors.append(
            WoodblockError(
                tier="degraded",
                code="SPECTRAL_RENDER_UNAVAILABLE",
                message="spectral K/S data is present, but local T3 rendering is not enabled",
                hint="use t2_empirical locally or run the spectral renderer on the GPU host",
                recoverable=True,
            )
        )
    return {"tier": tier, "errors": errors}


def get_render_tier() -> ToolResult[dict[str, Any]]:
    """Compute the active forward-render tier from current calibration state."""
    cal_file = paths.WB_DATA_DIR / "active_calibration"
    cal_id = cal_file.read_text().strip() if cal_file.is_file() else None
    lut_path = _lut_path()
    spectral_path = paths.WB_DATA_DIR / "calibrations" / "spectral_ks.npz"

    ctx = _rt.RenderTierContext(
        calibration_id=cal_id,
        empirical_lut_available=lut_path.is_file(),
        spectral_ks_available=spectral_path.is_file(),
        stack_depth=0,  # solver fills in real depth at simulate_overprint time
    )
    chosen = _rt.choose_render_tier(ctx)
    return ToolResult(
        ok=True,
        data={
            "tier": chosen,
            "calibration_id": cal_id,
            "empirical_lut_available": ctx.empirical_lut_available,
            "spectral_ks_available": ctx.spectral_ks_available,
        },
    )


def upload_swatch_overprint_matrix(csv_path: str) -> ToolResult[dict[str, Any]]:
    """Ingest a 2-layer overprint CSV (base × top × dilution × measured RGB) → T2 LUT."""
    p = Path(csv_path)
    if not p.is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="CSV_FILE_MISSING",
                    message=f"swatch CSV not found: {csv_path}",
                    recoverable=True,
                ),
            ],
        )
    try:
        rows = _parse_swatch_csv(p)
    except (ValueError, KeyError) as exc:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal", code="INVALID_SWATCH_CSV", message=str(exc), recoverable=True
                ),
            ],
        )

    base_idx = np.asarray([r["base_idx"] for r in rows], dtype=np.int32)
    top_idx = np.asarray([r["top_idx"] for r in rows], dtype=np.int32)
    dilution = np.asarray([r["dilution"] for r in rows], dtype=np.float32)
    measured_rgb = np.asarray([r["measured_rgb"] for r in rows], dtype=np.float32)
    predicted_rgb = np.asarray([r["predicted_rgb"] for r in rows], dtype=np.float32)
    mean_bias = (
        (measured_rgb - predicted_rgb).mean(axis=0) if rows else np.zeros(3, dtype=np.float32)
    ).astype(np.float32)

    calibration_id = f"empirical_matrix_{p.stem}"
    lut_path = _lut_path()
    np.savez(
        lut_path,
        base_idx=base_idx,
        top_idx=top_idx,
        dilution=dilution,
        measured_rgb=measured_rgb,
        predicted_rgb=predicted_rgb,
        mean_bias=mean_bias,
        calibration_id=np.asarray(calibration_id),
        source_csv=np.asarray(str(p)),
    )
    (paths.WB_DATA_DIR / "active_calibration").write_text(calibration_id)
    return ToolResult(
        ok=True,
        data={
            "csv_path": str(p),
            "rows_ingested": len(rows),
            "lut_path": str(lut_path),
            "calibration_id": calibration_id,
            "mean_bias_rgb": [round(float(v), 5) for v in mean_bias],
            "tier_after_ingest": get_render_tier().data["tier"],
        },
    )


def _parse_swatch_csv(p: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with p.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")
        for raw in reader:
            if not any((v or "").strip() for v in raw.values()):
                continue
            base_idx = _parse_pigment(raw["base"])
            top_idx = _parse_pigment(raw["top"])
            dilution = float(raw.get("dilution") or 1.0)
            measured = np.asarray(_parse_rgb(raw), dtype=np.float32)
            base_rgb = _fr.PIGMENT_TABLE[base_idx]
            top_rgb = _fr.PIGMENT_TABLE[top_idx]
            pred = (1.0 - dilution) * base_rgb + dilution * top_rgb
            rows.append(
                {
                    "base_idx": base_idx,
                    "top_idx": top_idx,
                    "dilution": float(np.clip(dilution, 0.0, 1.0)),
                    "measured_rgb": measured,
                    "predicted_rgb": pred.astype(np.float32),
                }
            )
    return rows


def compare_render_tiers(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Render same plan under T1/T2/T3 + show ΔE deltas between tiers."""
    from backend.services.v23 import orchestrator as _orch
    from backend.services.v23.stages import s10_emit

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "tier_renders": {"t1_mixbox": None},
                "dE_deltas": {},
            },
            errors=[exc.error],
        )

    plan_dir = _orch._plan_dir(plan.session_id, plan.plan_id)
    t1_rgb = _image_bytes_to_rgb(s10_emit._render_composite(plan))
    t1_path = plan_dir / "compare_t1_mixbox.png"
    _write_rgb(t1_path, t1_rgb)

    renders: dict[str, str | None] = {"t1_mixbox": str(t1_path)}
    deltas: dict[str, dict[str, float]] = {}
    errors: list[WoodblockError] = []
    lut = _load_empirical_lut()
    if lut is not None and lut["rows"] > 0:
        t2_rgb, _ = _apply_t2_empirical(t1_rgb)
        t2_path = plan_dir / "compare_t2_empirical.png"
        _write_rgb(t2_path, t2_rgb)
        renders["t2_empirical"] = str(t2_path)
        summary = _color.delta_e_summary(t2_rgb, t1_rgb)
        deltas["t2_empirical_vs_t1_mixbox"] = {
            "dE_mean": round(summary["dE_mean"], 3),
            "dE_p95": round(summary["dE_p95"], 3),
            "dE_max": round(summary["dE_max"], 3),
        }
    else:
        renders["t2_empirical"] = None
        errors.append(
            WoodblockError(
                tier="warn",
                code="T2_LUT_MISSING",
                message="no empirical swatch LUT is active; only t1_mixbox was rendered",
                hint="call upload_swatch_overprint_matrix(csv_path)",
                recoverable=True,
            )
        )
    renders["t3_spectral"] = None
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "tier_renders": renders,
            "dE_deltas": deltas,
        },
        errors=errors,
    )


__all__ = [
    "simulate_overprint",
    "get_render_tier",
    "upload_swatch_overprint_matrix",
    "compare_render_tiers",
]
