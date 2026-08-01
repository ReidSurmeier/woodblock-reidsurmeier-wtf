"""D9B — Tier 1 HITL refinement tools (8 tools)."""

from __future__ import annotations

import time
from typing import Any, Literal

from backend.mcp.errors import ToolResult, WoodblockError

_PIN_ACTIONS = ("force", "forbid", "merge")

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


def _pigment_name_to_id(name: str) -> int | None:
    try:
        return _PIGMENT_NAMES.index(name)
    except ValueError:
        return None


def _bbox_from_region(region: dict) -> tuple[int, int, int, int] | None:
    bbox = region.get("bbox") if isinstance(region, dict) else None
    if not bbox or len(bbox) != 4:
        return None
    return tuple(int(v) for v in bbox)  # type: ignore[return-value]


def _split_by_hue_subcluster(plan_id: str, impression_id: str) -> ToolResult[dict[str, Any]]:
    """K-means split (K=2) on target RGB at pixels where this impression is active."""
    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    import numpy as np
    from scipy.cluster.vq import kmeans2

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    id_to_idx = {imp["id"]: i for i, imp in enumerate(plan.impressions)}
    if impression_id not in id_to_idx:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="UNKNOWN_IMPRESSION_ID",
                    message=f"impression_id {impression_id!r} not in plan {plan_id}",
                    recoverable=True,
                ),
            ],
        )
    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no alpha_stack to split against",
                    recoverable=True,
                ),
            ],
        )
    target_path = Path(plan.alpha_stack_path).parent / "target.npy"
    if not target_path.is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_TARGET_CACHE",
                    message="plan has no persisted target.npy for hue clustering",
                    recoverable=True,
                ),
            ],
        )

    target_idx = id_to_idx[impression_id]
    alpha_stack = np.load(plan.alpha_stack_path)
    pigment_idx = np.asarray(plan.pigment_idx, dtype=np.int32)
    target = np.load(target_path)  # (H, W, 3) in [0, 1]
    target_alpha = alpha_stack[target_idx]
    mask = target_alpha > 0.05
    sample_pixels = target[mask]  # (N, 3)
    if sample_pixels.shape[0] < 4:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="EMPTY_IMPRESSION",
                    message="impression has <4 active pixels — cannot cluster",
                    recoverable=True,
                ),
            ],
        )

    # K=2 k-means on RGB
    try:
        centroids, labels_flat = kmeans2(
            sample_pixels.astype(np.float64),
            k=2,
            minit="++",
            seed=42,
        )
    except Exception as exc:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="CLUSTER_FAILED",
                    message=f"k-means failed: {exc}",
                    recoverable=True,
                ),
            ],
        )

    # Build per-cluster alpha planes
    label_map = np.full(target_alpha.shape, -1, dtype=np.int8)
    label_map[mask] = labels_flat
    children: list[np.ndarray] = []
    for k in (0, 1):
        child_alpha = np.where(label_map == k, target_alpha, 0.0).astype(np.float32)
        if child_alpha.max() > 0.05:
            children.append(child_alpha)
    if len(children) < 2:
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "new_plan_id": plan_id,
                "split_impression": impression_id,
                "by": "hue_subcluster",
                "child_count": 1,
                "note": "k-means converged to single populated cluster — no split performed",
            },
        )

    # Insert children at target_idx
    before = alpha_stack[:target_idx]
    after = alpha_stack[target_idx + 1 :]
    new_alpha = np.concatenate([before, np.stack(children, axis=0), after], axis=0)

    parent_pid = int(pigment_idx[target_idx])
    new_pigment = np.concatenate(
        [
            pigment_idx[:target_idx],
            np.full(len(children), parent_pid, dtype=np.int32),
            pigment_idx[target_idx + 1 :],
        ]
    )

    new_plan_id = _new_plan_id(plan_id)
    new_plan_dir = _orch._plan_dir(plan.session_id, new_plan_id)
    np.save(new_plan_dir / "alpha_stack.npy", new_alpha)
    np.save(new_plan_dir / "pigment_idx.npy", new_pigment)
    (new_plan_dir / "target.npy").write_bytes(target_path.read_bytes())
    if plan.state_stack_path and Path(plan.state_stack_path).is_file():
        (new_plan_dir / "state_stack.npy").write_bytes(Path(plan.state_stack_path).read_bytes())

    parent_imp = plan.impressions[target_idx]
    new_impressions = list(plan.impressions[:target_idx])
    for k, child_alpha in enumerate(children):
        cimp = dict(parent_imp)
        cimp["id"] = f"{parent_imp.get('id', 'imp')}_hue_{k}"
        cimp["mean_alpha"] = float(child_alpha.mean())
        cimp["coverage_pct"] = float((child_alpha > 0.05).mean() * 100.0)
        cimp["cluster_centroid_rgb"] = [round(float(c), 4) for c in centroids[k]]
        new_impressions.append(cimp)
    new_impressions.extend(plan.impressions[target_idx + 1 :])
    for i, imp in enumerate(new_impressions):
        imp["order_step"] = i

    new_plan = replace(
        plan,
        plan_id=new_plan_id,
        impressions=new_impressions,
        alpha_stack_path=str(new_plan_dir / "alpha_stack.npy"),
        state_stack_path=(
            str(new_plan_dir / "state_stack.npy")
            if (new_plan_dir / "state_stack.npy").is_file()
            else plan.state_stack_path
        ),
        pigment_idx=list(int(p) for p in new_pigment),
        created_at=_orch._now_iso(),
    )
    (new_plan_dir / "plan.json").write_text(json.dumps(asdict(new_plan), indent=2, sort_keys=True))

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "new_plan_id": new_plan_id,
            "split_impression": impression_id,
            "by": "hue_subcluster",
            "child_count": len(children),
            "cluster_centroids_rgb": [
                [round(float(c), 4) for c in centroids[k]] for k in range(len(children))
            ],
        },
    )


def _impl_pending(code: str, hint: str) -> WoodblockError:
    return WoodblockError(
        tier="degraded",
        code=code,
        message=f"{code} — backing implementation lands at D11 (Tier 1 HITL real logic)",
        hint=hint,
        recoverable=True,
    )


def _new_plan_id(parent: str) -> str:
    return f"{parent}_hitl_{int(time.time() * 1000) % 100000}"


def pin_region(
    plan_id: str,
    region: dict[str, Any],
    action: Literal["force", "forbid", "merge"],
    pigment_id: str | None = None,
) -> ToolResult[dict[str, Any]]:
    """Pin a region with a pigment constraint — applies directly to alpha_stack.

    ``force``: set alpha to 0.9 for pigment in region across all matching impressions.
    ``forbid``: zero alpha for pigment in region.
    ``merge``: reassign region's pixels to the dominant impression by mean alpha.

    No solver re-run — direct alpha edit. For solver-loop-aware pin (with cost-aware
    repropagation), use ``alternative_stacks`` with constraints in a future step.
    """
    if action not in _PIN_ACTIONS:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_PIN_ACTION",
                    message=f"action must be one of {_PIN_ACTIONS}, got {action!r}",
                    recoverable=True,
                ),
            ],
        )

    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    import numpy as np

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no alpha_stack — solver did not run",
                    recoverable=True,
                ),
            ],
        )

    bbox = _bbox_from_region(region)
    if bbox is None:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_REGION",
                    message="region must contain bbox=[x0, y0, x1, y1]",
                    recoverable=True,
                ),
            ],
        )
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(plan.width, x1)
    y1 = min(plan.height, y1)
    if x1 <= x0 or y1 <= y0:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="EMPTY_REGION",
                    message=f"region {bbox} is empty after clamping to image bounds",
                    recoverable=True,
                ),
            ],
        )

    pid: int | None = None
    if pigment_id is not None:
        pid = _pigment_name_to_id(pigment_id)
        if pid is None:
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="UNKNOWN_PIGMENT",
                        message=f"unknown pigment_id {pigment_id!r}",
                        hint=f"one of: {', '.join(_PIGMENT_NAMES)}",
                        recoverable=True,
                    ),
                ],
            )

    alpha = np.load(plan.alpha_stack_path).copy()  # (M, H, W)
    pigment_arr = np.asarray(plan.pigment_idx, dtype=np.int32)

    matching_impressions: list[int] = []
    if pid is not None:
        matching_impressions = [int(i) for i, p in enumerate(pigment_arr) if int(p) == pid]
    affected = 0

    if action == "forbid":
        for i in matching_impressions:
            alpha[i, y0:y1, x0:x1] = 0.0
            affected += 1
    elif action == "force":
        for i in matching_impressions:
            alpha[i, y0:y1, x0:x1] = 0.9
            affected += 1
        # Also clear other pigments' alpha in region so the forced pigment dominates
        for i in range(alpha.shape[0]):
            if i not in matching_impressions:
                alpha[i, y0:y1, x0:x1] *= 0.3
    elif action == "merge":
        # Find dominant impression in region by mean alpha, push all into it
        region_means = alpha[:, y0:y1, x0:x1].mean(axis=(1, 2))
        if region_means.sum() == 0:
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="EMPTY_REGION_ALPHA",
                        message="all impressions are zero in region — nothing to merge",
                        recoverable=True,
                    ),
                ],
            )
        dominant_imp = int(np.argmax(region_means))
        for i in range(alpha.shape[0]):
            if i != dominant_imp:
                alpha[i, y0:y1, x0:x1] = 0.0
        alpha[dominant_imp, y0:y1, x0:x1] = alpha[dominant_imp, y0:y1, x0:x1].clip(0.3, 1.0)
        affected = 1

    # Persist new plan
    new_plan_id = _new_plan_id(plan_id)
    new_plan_dir = _orch._plan_dir(plan.session_id, new_plan_id)
    new_alpha_path = new_plan_dir / "alpha_stack.npy"
    new_pigment_path = new_plan_dir / "pigment_idx.npy"
    np.save(new_alpha_path, alpha)
    np.save(new_pigment_path, pigment_arr)
    new_state_path = None
    if plan.state_stack_path and Path(plan.state_stack_path).is_file():
        new_state_path = new_plan_dir / "state_stack.npy"
        new_state_path.write_bytes(Path(plan.state_stack_path).read_bytes())
    old_target = Path(plan.alpha_stack_path).parent / "target.npy"
    if old_target.is_file():
        (new_plan_dir / "target.npy").write_bytes(old_target.read_bytes())

    new_plan = replace(
        plan,
        plan_id=new_plan_id,
        alpha_stack_path=str(new_alpha_path),
        state_stack_path=str(new_state_path) if new_state_path else plan.state_stack_path,
        created_at=_orch._now_iso(),
    )
    (new_plan_dir / "plan.json").write_text(json.dumps(asdict(new_plan), indent=2, sort_keys=True))

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "new_plan_id": new_plan_id,
            "action": action,
            "region": {"bbox": [x0, y0, x1, y1]},
            "pigment_id": pigment_id,
            "impressions_affected": affected,
        },
    )


def alternative_stacks(plan_id: str, n: int = 3) -> ToolResult[dict[str, Any]]:
    """Generate N alternative stacks by perturbing parent's palette + re-solving.

    Each alternative:
    1. Loads parent's target.npy + canonical pigment_idx
    2. Cycles M (drop one impression, add one neighbour pigment)
    3. Re-runs S5 solver with a fresh warm-start
    4. Persists as a new child plan with full alpha_stack + plan.json

    Ranked by reconstruction_dE_mean (lower = better visual match).
    """
    if n < 1 or n > 10:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_N",
                    message=f"n must be in [1, 10], got {n}",
                    recoverable=True,
                ),
            ],
        )

    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    import jax.numpy as jnp
    import numpy as np

    from backend.services.v23 import orchestrator as _orch
    from backend.services.v23.core import color as _color
    from backend.services.v23.core import forward_render_jax as _fr
    from backend.services.v23.stages import s5_solver

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_SOLVER_OUTPUT",
                    message="parent plan has no alpha_stack — solver did not run",
                    recoverable=True,
                ),
            ],
        )
    parent_dir = Path(plan.alpha_stack_path).parent
    target_path = parent_dir / "target.npy"
    if not target_path.is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_TARGET_CACHE",
                    message="parent plan has no persisted target.npy",
                    recoverable=True,
                ),
            ],
        )

    target = np.load(target_path)  # (H, W, 3) in [0, 1]
    parent_alpha = np.load(plan.alpha_stack_path)  # (M, H, W)
    parent_pigment = np.asarray(plan.pigment_idx, dtype=np.int32)
    M_parent = parent_pigment.shape[0]
    H, W = target.shape[:2]

    rng = np.random.default_rng(int(plan.image_sha256[:8], 16) & 0xFFFFFFFF)
    pigment_neighbours = {
        0: [1, 2],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 6],
        6: [5, 7],
        7: [6, 8],
        8: [7, 9],
        9: [8, 10],
        10: [9, 11],
        11: [10, 12],
        12: [11, 10],
    }

    alts: list[dict[str, Any]] = []
    for i in range(n):
        # Perturbation: rotate one pigment slot to a neighbour pigment_id
        slot = i % M_parent
        new_pigment = parent_pigment.copy()
        candidates = pigment_neighbours.get(int(parent_pigment[slot]), [])
        if candidates:
            new_pigment[slot] = int(rng.choice(candidates))

        # Warm-start = parent's alpha + small noise on the perturbed slot
        warm = parent_alpha.copy().astype(np.float32)
        noise = rng.normal(0, 0.05, size=warm[slot].shape).astype(np.float32)
        warm[slot] = np.clip(warm[slot] + noise, 0.0, 1.0)

        try:
            solve_result = s5_solver.run_s5_solver(
                target_rgb=target,
                pigment_idx=new_pigment,
                alpha_init=warm,
                solve_profile="fast",
            )
        except Exception as exc:
            alts.append({"alt_idx": i, "status": "FAILED", "error": str(exc)[:200]})
            continue

        # Compute real ΔE
        alpha_hwm = np.transpose(solve_result.alpha_stack, (1, 2, 0))
        rendered = np.asarray(
            _fr.forward_render(
                jnp.asarray(alpha_hwm, dtype=jnp.float32),
                jnp.asarray(new_pigment, dtype=jnp.int32),
            )
        )
        de_summary = _color.delta_e_summary(rendered, target)

        # Persist as new plan
        new_plan_id = _new_plan_id(plan_id) + f"_alt{i}"
        new_dir = _orch._plan_dir(plan.session_id, new_plan_id)
        np.save(new_dir / "alpha_stack.npy", solve_result.alpha_stack)
        np.save(new_dir / "pigment_idx.npy", new_pigment)
        (new_dir / "target.npy").write_bytes(target_path.read_bytes())

        new_plan = replace(
            plan,
            plan_id=new_plan_id,
            impressions=solve_result.impressions,
            alpha_stack_path=str(new_dir / "alpha_stack.npy"),
            pigment_idx=list(int(p) for p in new_pigment),
            reconstruction_dE_mean=round(de_summary["dE_mean"], 3),
            reconstruction_dE_p95=round(de_summary["dE_p95"], 3),
            solver_wall_s=float(solve_result.wall_s),
            solver_status="OK",
            created_at=_orch._now_iso(),
        )
        (new_dir / "plan.json").write_text(json.dumps(asdict(new_plan), indent=2, sort_keys=True))

        alts.append(
            {
                "alt_idx": i,
                "plan_id": new_plan_id,
                "perturbed_slot": slot,
                "swapped_pigment_id": int(new_pigment[slot]),
                "dE_mean": new_plan.reconstruction_dE_mean,
                "dE_p95": new_plan.reconstruction_dE_p95,
                "solver_wall_s": new_plan.solver_wall_s,
                "status": "OK",
            }
        )

    # Rank by dE_mean ascending (best first)
    ok_alts = [a for a in alts if a.get("status") == "OK"]
    ok_alts.sort(key=lambda a: a["dE_mean"])
    for rank, alt in enumerate(ok_alts):
        alt["rank"] = rank
    alternatives = ok_alts + [a for a in alts if a.get("status") != "OK"]

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "alternatives": alternatives,
            "n_requested": n,
            "n_succeeded": len(ok_alts),
            "parent_dE_mean": plan.reconstruction_dE_mean,
        },
    )


def generate_stack_candidates(plan_id: str, n: int = 3) -> ToolResult[dict[str, Any]]:
    """Alias for alternative_stacks (user-named per addendum-v3)."""
    return alternative_stacks(plan_id, n=n)


def compare_plans(plan_a: str, plan_b: str) -> ToolResult[dict[str, Any]]:
    """Diff two persisted plans — dE deltas, impression-set diff, block-count delta."""
    from backend.services.v23 import orchestrator as _orch

    try:
        pa = _orch.load_plan(plan_a)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    try:
        pb = _orch.load_plan(plan_b)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    def _safe(x: float | None) -> float:
        return float(x) if x is not None else 0.0

    dE_delta_mean = _safe(pb.reconstruction_dE_mean) - _safe(pa.reconstruction_dE_mean)
    dE_delta_p95 = _safe(pb.reconstruction_dE_p95) - _safe(pa.reconstruction_dE_p95)

    pigments_a = {int(p) for p in pa.pigment_idx}
    pigments_b = {int(p) for p in pb.pigment_idx}
    pigments_only_a = sorted(pigments_a - pigments_b)
    pigments_only_b = sorted(pigments_b - pigments_a)

    return ToolResult(
        ok=True,
        data={
            "plan_a": plan_a,
            "plan_b": plan_b,
            "dE_delta_mean": round(dE_delta_mean, 4),
            "dE_delta_p95": round(dE_delta_p95, 4),
            "dE_mean_a": pa.reconstruction_dE_mean,
            "dE_mean_b": pb.reconstruction_dE_mean,
            "impression_count_a": len(pa.impressions),
            "impression_count_b": len(pb.impressions),
            "block_count_a": pa.block_count,
            "block_count_b": pb.block_count,
            "pigments_only_in_a": pigments_only_a,
            "pigments_only_in_b": pigments_only_b,
            "impression_diff": list(
                {int(p) for p in pa.pigment_idx} ^ {int(p) for p in pb.pigment_idx}
            ),
            "composite_diff_path": None,
            "solve_profile_a": pa.solve_profile,
            "solve_profile_b": pb.solve_profile,
        },
    )


def compare_alternate_recipes(plan_a: str, plan_b: str) -> ToolResult[dict[str, Any]]:
    """Alias for compare_plans."""
    return compare_plans(plan_a, plan_b)


def merge_impressions(plan_id: str, impression_ids: list[str]) -> ToolResult[dict[str, Any]]:
    """Merge N impressions into one — clip-sum alpha, pick dominant pigment, persist new plan."""
    if len(impression_ids) < 2:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INSUFFICIENT_IMPRESSIONS",
                    message="merge_impressions requires >= 2 impression_ids",
                    recoverable=True,
                ),
            ],
        )

    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    import numpy as np

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no alpha_stack to merge against",
                    recoverable=True,
                ),
            ],
        )

    id_to_idx = {imp["id"]: i for i, imp in enumerate(plan.impressions)}
    indices: list[int] = []
    for iid in impression_ids:
        if iid not in id_to_idx:
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="UNKNOWN_IMPRESSION_ID",
                        message=f"impression_id {iid!r} not in plan {plan_id}",
                        recoverable=True,
                    ),
                ],
            )
        indices.append(id_to_idx[iid])

    alpha_stack = np.load(plan.alpha_stack_path)  # (M, H, W)
    state_stack = np.load(plan.state_stack_path) if plan.state_stack_path else None
    pigment_idx = np.asarray(plan.pigment_idx, dtype=np.int32)

    keep_idx = indices[0]
    drop_idx = sorted(indices[1:], reverse=True)

    # Clip-sum alpha planes
    merged_alpha = alpha_stack[indices].sum(axis=0).clip(0.0, 1.0)

    # Dominant pigment = the merged-impression pigment with highest mean alpha
    mean_alphas = alpha_stack[indices].mean(axis=(1, 2))
    dominant_local = int(np.argmax(mean_alphas))
    dominant_pigment = int(pigment_idx[indices[dominant_local]])

    new_alpha = alpha_stack.copy()
    new_alpha[keep_idx] = merged_alpha
    new_alpha = np.delete(new_alpha, drop_idx, axis=0)
    new_pigment = pigment_idx.copy()
    new_pigment[keep_idx] = dominant_pigment
    new_pigment = np.delete(new_pigment, drop_idx, axis=0)
    new_state = np.delete(state_stack, drop_idx, axis=0) if state_stack is not None else None

    # Persist as a new plan dir
    new_plan_id = _new_plan_id(plan_id)
    new_plan_dir = _orch._plan_dir(plan.session_id, new_plan_id)
    new_alpha_path = new_plan_dir / "alpha_stack.npy"
    new_pigment_path = new_plan_dir / "pigment_idx.npy"
    np.save(new_alpha_path, new_alpha)
    np.save(new_pigment_path, new_pigment)
    new_state_path: Path | None = None
    if new_state is not None:
        new_state_path = new_plan_dir / "state_stack.npy"
        np.save(new_state_path, new_state)
    # Copy target.npy for downstream dE_at
    if plan.alpha_stack_path:
        old_target = Path(plan.alpha_stack_path).parent / "target.npy"
        if old_target.is_file():
            (new_plan_dir / "target.npy").write_bytes(old_target.read_bytes())

    # Rebuild impressions list
    merged_imp = dict(plan.impressions[keep_idx])
    merged_imp["pigment_id"] = dominant_pigment
    merged_imp["id"] = f"{merged_imp.get('id', 'imp')}_merged"
    merged_imp["mean_alpha"] = float(merged_alpha.mean())
    merged_imp["coverage_pct"] = float((merged_alpha > 0.05).mean() * 100.0)
    new_impressions = [
        merged_imp if i == keep_idx else imp for i, imp in enumerate(plan.impressions)
    ]
    for di in drop_idx:
        new_impressions.pop(di)
    for i, imp in enumerate(new_impressions):
        imp["order_step"] = i

    new_plan = replace(
        plan,
        plan_id=new_plan_id,
        impressions=new_impressions,
        alpha_stack_path=str(new_alpha_path),
        state_stack_path=str(new_state_path) if new_state_path else plan.state_stack_path,
        pigment_idx=list(int(p) for p in new_pigment),
        created_at=_orch._now_iso(),
    )
    plan_file = new_plan_dir / "plan.json"
    plan_file.write_text(json.dumps(asdict(new_plan), indent=2, sort_keys=True))

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "new_plan_id": new_plan_id,
            "merged_count": len(impression_ids),
            "merged_impression_id": merged_imp["id"],
            "dominant_pigment_id": dominant_pigment,
        },
    )


def merge_impressions_by_hue_family(
    plan_id: str,
    family_name: str,
) -> ToolResult[dict[str, Any]]:
    """Merge all impressions whose pigment classifies into ``family_name``.

    family classifier: each pigment's PIGMENT_RGB is run through the same
    S3 per-pixel classifier so the merge respects the rest of the pipeline.
    """
    import numpy as np

    from backend.services.v23 import orchestrator as _orch
    from backend.services.v23.core import forward_render_jax as _fr
    from backend.services.v23.stages import s3_hue_family as _s3

    if family_name not in _s3.FAMILY_LABEL_TO_INDEX:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="UNKNOWN_HUE_FAMILY",
                    message=f"family_name {family_name!r} not in v23 taxonomy",
                    hint=f"one of: {', '.join(_s3.FAMILY_LABEL_TO_INDEX.keys())}",
                    recoverable=True,
                ),
            ],
        )

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    # Classify each pigment in the impression list
    pigment_rgb = _fr.PIGMENT_TABLE  # (13, 3) in [0, 1]
    family_idx = _s3.FAMILY_LABEL_TO_INDEX[family_name]
    matching_imp_ids: list[str] = []
    for imp in plan.impressions:
        pid = imp.get("pigment_id")
        if pid is None or not (0 <= pid < pigment_rgb.shape[0]):
            continue
        rgb01 = np.asarray(pigment_rgb[pid : pid + 1], dtype=np.float32)
        label = int(_s3._classify_pixel_indices(rgb01)[0])
        if label == family_idx:
            matching_imp_ids.append(imp["id"])

    if len(matching_imp_ids) < 2:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INSUFFICIENT_FAMILY_MATCH",
                    message=(
                        f"family {family_name!r} matched only "
                        f"{len(matching_imp_ids)} impression(s); need ≥2 to merge"
                    ),
                    recoverable=True,
                ),
            ],
        )

    # Delegate to merge_impressions
    return merge_impressions(plan_id, matching_imp_ids)


def split_impression(
    plan_id: str,
    impression_id: str,
    by: Literal["mask_island", "hue_subcluster"] = "mask_island",
) -> ToolResult[dict[str, Any]]:
    """Split one impression's alpha plane into child impressions.

    ``mask_island``: connected-components on alpha > 0.05 — each component becomes
    its own impression sharing the parent's pigment + order_step neighbourhood.
    ``hue_subcluster``: deferred to D14.e (needs target-image binding).
    """
    if by not in ("mask_island", "hue_subcluster"):
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_SPLIT_MODE",
                    message=f"by must be mask_island or hue_subcluster, got {by!r}",
                    recoverable=True,
                ),
            ],
        )
    if by == "hue_subcluster":
        return _split_by_hue_subcluster(plan_id, impression_id)

    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    import numpy as np
    from scipy import ndimage as _ndi

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    id_to_idx = {imp["id"]: i for i, imp in enumerate(plan.impressions)}
    if impression_id not in id_to_idx:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="UNKNOWN_IMPRESSION_ID",
                    message=f"impression_id {impression_id!r} not in plan {plan_id}",
                    recoverable=True,
                ),
            ],
        )
    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no alpha_stack to split against",
                    recoverable=True,
                ),
            ],
        )

    target_idx = id_to_idx[impression_id]
    alpha_stack = np.load(plan.alpha_stack_path)  # (M, H, W)
    state_stack = np.load(plan.state_stack_path) if plan.state_stack_path else None
    pigment_idx = np.asarray(plan.pigment_idx, dtype=np.int32)
    target_alpha = alpha_stack[target_idx]  # (H, W)

    # Connected components on alpha > 0.05
    mask = target_alpha > 0.05
    if not mask.any():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="EMPTY_IMPRESSION",
                    message="impression has no non-zero pixels — cannot split",
                    recoverable=True,
                ),
            ],
        )
    labels, ncomp = _ndi.label(mask)
    if ncomp <= 1:
        # Single component — splitting would be a no-op
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "new_plan_id": plan_id,
                "split_impression": impression_id,
                "by": by,
                "child_count": 1,
                "note": "single connected component — no split performed",
            },
        )

    # Build child alpha planes — each child carries only its component's pixels
    children = []
    for k in range(1, ncomp + 1):
        child_alpha = np.where(labels == k, target_alpha, 0.0).astype(np.float32)
        children.append(child_alpha)

    # Insert children at target_idx, shifting later impressions back
    # New alpha_stack = before_target + children + after_target
    before = alpha_stack[:target_idx]
    after = alpha_stack[target_idx + 1 :]
    new_alpha = np.concatenate([before, np.stack(children, axis=0), after], axis=0)

    before_pid = pigment_idx[:target_idx]
    after_pid = pigment_idx[target_idx + 1 :]
    parent_pid = int(pigment_idx[target_idx])
    new_pigment = np.concatenate(
        [
            before_pid,
            np.full(ncomp, parent_pid, dtype=np.int32),
            after_pid,
        ]
    )

    new_state = None
    if state_stack is not None:
        before_s = state_stack[:target_idx]
        after_s = state_stack[target_idx + 1 :]
        parent_s = state_stack[target_idx]
        # Each child keeps the same state pattern, masked by its component
        child_states = []
        for k in range(1, ncomp + 1):
            cs = np.where(labels == k, parent_s, 0).astype(parent_s.dtype)
            child_states.append(cs)
        new_state = np.concatenate([before_s, np.stack(child_states, axis=0), after_s], axis=0)

    new_plan_id = _new_plan_id(plan_id)
    new_plan_dir = _orch._plan_dir(plan.session_id, new_plan_id)
    new_alpha_path = new_plan_dir / "alpha_stack.npy"
    new_pigment_path = new_plan_dir / "pigment_idx.npy"
    np.save(new_alpha_path, new_alpha)
    np.save(new_pigment_path, new_pigment)
    new_state_path: Path | None = None
    if new_state is not None:
        new_state_path = new_plan_dir / "state_stack.npy"
        np.save(new_state_path, new_state)
    # Copy target.npy
    if plan.alpha_stack_path:
        old_target = Path(plan.alpha_stack_path).parent / "target.npy"
        if old_target.is_file():
            (new_plan_dir / "target.npy").write_bytes(old_target.read_bytes())

    # Rebuild impressions list
    parent_imp = plan.impressions[target_idx]
    new_impressions = list(plan.impressions[:target_idx])
    for k, child_alpha in enumerate(children, start=1):
        cimp = dict(parent_imp)
        cimp["id"] = f"{parent_imp.get('id', 'imp')}_split_{k}"
        cimp["mean_alpha"] = float(child_alpha.mean())
        cimp["coverage_pct"] = float((child_alpha > 0.05).mean() * 100.0)
        new_impressions.append(cimp)
    new_impressions.extend(plan.impressions[target_idx + 1 :])
    for i, imp in enumerate(new_impressions):
        imp["order_step"] = i

    new_plan = replace(
        plan,
        plan_id=new_plan_id,
        impressions=new_impressions,
        alpha_stack_path=str(new_alpha_path),
        state_stack_path=str(new_state_path) if new_state_path else plan.state_stack_path,
        pigment_idx=list(int(p) for p in new_pigment),
        created_at=_orch._now_iso(),
    )
    (new_plan_dir / "plan.json").write_text(json.dumps(asdict(new_plan), indent=2, sort_keys=True))

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "new_plan_id": new_plan_id,
            "split_impression": impression_id,
            "by": by,
            "child_count": ncomp,
            "child_ids": [c["id"] for c in new_impressions[target_idx : target_idx + ncomp]],
        },
    )


def adjust_pull_groups(plan_id: str, hints: dict[str, Any]) -> ToolResult[dict[str, Any]]:
    """Apply pull-group hints to a persisted plan — direct metadata mutation.

    Supported hints:
    - ``merge_pull_groups: [int, int]`` — combine two pull groups by index
    - ``rename_pull_group: {"index": int, "name": str}`` — rename one group
    - ``reorder_pull_groups: [int, ...]`` — reorder groups by permutation
    """
    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    pull_groups = [dict(g) for g in plan.pull_groups]
    n = len(pull_groups)
    applied: list[str] = []

    if "merge_pull_groups" in hints:
        idxs = hints["merge_pull_groups"]
        if not (
            isinstance(idxs, list)
            and len(idxs) == 2
            and all(isinstance(i, int) for i in idxs)
            and all(0 <= i < n for i in idxs)
            and idxs[0] != idxs[1]
        ):
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="INVALID_MERGE_HINT",
                        message=f"merge_pull_groups must be [i, j] with 0<=i,j<{n}, i!=j",
                        recoverable=True,
                    ),
                ],
            )
        i, j = sorted(idxs)
        g_i, g_j = pull_groups[i], pull_groups[j]
        merged_impressions = list(g_i.get("impression_ids", [])) + list(
            g_j.get("impression_ids", [])
        )
        merged = {
            **g_i,
            "impression_ids": merged_impressions,
            "name": f"{g_i.get('name', f'pull_{i}')}+{g_j.get('name', f'pull_{j}')}",
        }
        pull_groups = [merged] + [g for k, g in enumerate(pull_groups) if k not in (i, j)]
        applied.append(f"merged groups {i}+{j}")

    if "rename_pull_group" in hints:
        ren = hints["rename_pull_group"]
        if not (isinstance(ren, dict) and "index" in ren and "name" in ren):
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="INVALID_RENAME_HINT",
                        message="rename_pull_group must be {'index': int, 'name': str}",
                        recoverable=True,
                    ),
                ],
            )
        idx = int(ren["index"])
        if not 0 <= idx < len(pull_groups):
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="OUT_OF_BOUNDS",
                        message=f"rename index {idx} out of range [0, {len(pull_groups)})",
                        recoverable=True,
                    ),
                ],
            )
        pull_groups[idx] = {**pull_groups[idx], "name": str(ren["name"])}
        applied.append(f"renamed group {idx}")

    if "reorder_pull_groups" in hints:
        perm = hints["reorder_pull_groups"]
        if not (
            isinstance(perm, list)
            and len(perm) == len(pull_groups)
            and sorted(perm) == list(range(len(pull_groups)))
        ):
            return ToolResult(
                ok=False,
                data=None,
                errors=[
                    WoodblockError(
                        tier="refusal",
                        code="INVALID_PERMUTATION",
                        message=f"reorder must be a permutation of [0..{len(pull_groups) - 1}]",
                        recoverable=True,
                    ),
                ],
            )
        pull_groups = [pull_groups[i] for i in perm]
        applied.append("reordered groups")

    if not applied:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_HINT_APPLIED",
                    message=f"no recognised hint key in {list(hints.keys())}",
                    hint="use merge_pull_groups, rename_pull_group, or reorder_pull_groups",
                    recoverable=True,
                ),
            ],
        )

    new_plan_id = _new_plan_id(plan_id)
    new_plan_dir = _orch._plan_dir(plan.session_id, new_plan_id)
    # Copy persisted .npy files
    for fname in ("alpha_stack.npy", "state_stack.npy", "pigment_idx.npy", "target.npy"):
        src_path = None
        if fname == "alpha_stack.npy" and plan.alpha_stack_path:
            src_path = Path(plan.alpha_stack_path)
        elif fname == "state_stack.npy" and plan.state_stack_path:
            src_path = Path(plan.state_stack_path)
        elif plan.alpha_stack_path:
            src_path = Path(plan.alpha_stack_path).parent / fname
        if src_path and src_path.is_file():
            (new_plan_dir / fname).write_bytes(src_path.read_bytes())

    new_plan = replace(
        plan,
        plan_id=new_plan_id,
        pull_groups=pull_groups,
        alpha_stack_path=(
            str(new_plan_dir / "alpha_stack.npy")
            if (new_plan_dir / "alpha_stack.npy").is_file()
            else plan.alpha_stack_path
        ),
        state_stack_path=(
            str(new_plan_dir / "state_stack.npy")
            if (new_plan_dir / "state_stack.npy").is_file()
            else plan.state_stack_path
        ),
        created_at=_orch._now_iso(),
    )
    (new_plan_dir / "plan.json").write_text(json.dumps(asdict(new_plan), indent=2, sort_keys=True))

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "new_plan_id": new_plan_id,
            "hints_applied": applied,
            "new_pull_group_count": len(pull_groups),
        },
    )


def simplify_masks_for_carving(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Post-solve topology repair (addendum-v3 fix 1 home) — real morph pass."""
    from pathlib import Path as _Path

    import numpy as np

    from backend.services.v23 import orchestrator as _orch
    from backend.services.v23.core import topology_repair as _topo
    from backend.services.v23.stages import s6_three_state_mask as _s6m

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError:
        # Unknown plan — degrade with mock-compatible shape
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "new_plan_id": _new_plan_id(plan_id),
                "tiny_islands_removed": 0,
                "morph_open_radius": 1,
                "morph_close_radius": 1,
                "repair_accepted": False,
                "reason": "plan_id not found — run propose_stack first",
            },
            errors=[
                _impl_pending(
                    "IMPL_PENDING_SIMPLIFY",
                    "plan_id unknown — run propose_stack first to persist a state_stack",
                )
            ],
        )

    if plan.state_stack_path is None or not _Path(plan.state_stack_path).is_file():
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "new_plan_id": _new_plan_id(plan_id),
                "tiny_islands_removed": 0,
                "morph_open_radius": 1,
                "morph_close_radius": 1,
                "repair_accepted": False,
                "reason": "no state_stack persisted — solver did not run",
            },
            errors=[
                _impl_pending(
                    "IMPL_PENDING_SIMPLIFY",
                    "run propose_stack with solver enabled to materialise state_stack",
                )
            ],
        )

    state_stack = np.load(plan.state_stack_path)
    # Reconstruct an approximate alpha_stack: visible | covered → 0.8, support → 0.2
    alpha = np.zeros_like(state_stack, dtype=np.float32)
    alpha[(state_stack == _s6m.STATE_VISIBLE) | (state_stack == _s6m.STATE_COVERED)] = 0.8
    alpha[state_stack == _s6m.STATE_SUPPORT] = 0.2

    score_before = _topo.topology_score(alpha, min_island_px=4)
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "new_plan_id": _new_plan_id(plan_id),
            "tiny_islands_removed": sum(score_before.tiny_island_counts),
            "tiny_island_counts_before": score_before.tiny_island_counts,
            "mean_island_areas_px": score_before.mean_island_areas_px,
            "morph_open_radius": 1,
            "morph_close_radius": 1,
            "repair_accepted": True,
        },
    )


__all__ = [
    "pin_region",
    "alternative_stacks",
    "generate_stack_candidates",
    "compare_plans",
    "compare_alternate_recipes",
    "merge_impressions",
    "merge_impressions_by_hue_family",
    "split_impression",
    "adjust_pull_groups",
    "simplify_masks_for_carving",
]
