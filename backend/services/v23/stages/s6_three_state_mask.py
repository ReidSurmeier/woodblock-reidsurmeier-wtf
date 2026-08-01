"""D11.a — S6 three-state mask classifier (post-solve).

Per addendum-v2 §"per-mask three-state pixel ontology": every pixel of
every impression gets a label relative to the rest of the stack.

State encoding (uint8 for cheap on-disk persistence):
- 0 = none     — impression has α < τ_sup at this pixel (not printed)
- 1 = visible  — α ≥ τ_vis AND no later impression covers
- 2 = covered  — α ≥ τ_vis AND some later impression has α ≥ τ_cov
- 3 = support  — τ_sup ≤ α < τ_vis (faint underprint contribution)

Thresholds per interface contract B-5 — kept simple for day-1; D11.b
will let calibration tune them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

STATE_NONE: int = 0
STATE_VISIBLE: int = 1
STATE_COVERED: int = 2
STATE_SUPPORT: int = 3

# Default thresholds — keep aligned with interface contract B-5.
_TAU_VIS: float = 0.40
_TAU_COV: float = 0.50
_TAU_SUP: float = 0.10

_STATE_LABEL: dict[int, str] = {
    STATE_NONE: "none",
    STATE_VISIBLE: "visible",
    STATE_COVERED: "covered",
    STATE_SUPPORT: "support",
}


def classify_three_state(
    alpha_stack: NDArray[np.float32],
    *,
    tau_vis: float = _TAU_VIS,
    tau_cov: float = _TAU_COV,
    tau_sup: float = _TAU_SUP,
) -> NDArray[np.uint8]:
    """Classify every pixel of every impression into {none, visible, covered, support}.

    Args:
        alpha_stack: ``(M, H, W)`` float in [0, 1]. Impressions in print order
            (i=0 first, i=M-1 last on top).
        tau_vis: minimum α to count an impression as printed at all (default 0.40).
        tau_cov: minimum α of a LATER impression that "hides" this one (default 0.50).
        tau_sup: minimum α for faint-support classification (default 0.10).

    Returns:
        ``(M, H, W)`` uint8 with state encoding 0..3.
    """
    m, h, w = alpha_stack.shape
    state = np.full((m, h, w), STATE_NONE, dtype=np.uint8)

    # Pre-compute "any later impression covers this pixel" per impression i.
    # cum_cov[i] = True wherever there exists j > i with α_j ≥ tau_cov.
    cov_mask = alpha_stack >= tau_cov  # (M, H, W) bool
    later_covered = np.zeros((m, h, w), dtype=bool)
    if m > 1:
        # Iterate from second-to-last impression downward
        next_acc = np.zeros((h, w), dtype=bool)
        for i in range(m - 1, -1, -1):
            later_covered[i] = next_acc
            next_acc = next_acc | cov_mask[i]

    for i in range(m):
        a = alpha_stack[i]
        # Default none unless an explicit band fires
        vis_band = a >= tau_vis
        sup_band = (a >= tau_sup) & ~vis_band
        # visible & not later-covered
        state[i] = np.where(
            vis_band & ~later_covered[i],
            STATE_VISIBLE,
            np.where(
                vis_band & later_covered[i],
                STATE_COVERED,
                np.where(sup_band, STATE_SUPPORT, STATE_NONE),
            ),
        ).astype(np.uint8)

    return state


def summarise_states(
    state_stack: NDArray[np.uint8],
) -> list[dict[str, Any]]:
    """Per-impression state percentages. ``state_stack`` shape ``(M, H, W)``."""
    m, h, w = state_stack.shape
    total = float(h * w)
    summary: list[dict[str, Any]] = []
    for i in range(m):
        sl = state_stack[i]
        entry: dict[str, Any] = {"impression_index": i}
        for code, label in _STATE_LABEL.items():
            count = int((sl == code).sum())
            entry[f"{label}_pct"] = round(count / total * 100.0, 3)
        summary.append(entry)
    return summary


__all__ = [
    "STATE_NONE",
    "STATE_VISIBLE",
    "STATE_COVERED",
    "STATE_SUPPORT",
    "classify_three_state",
    "summarise_states",
]
