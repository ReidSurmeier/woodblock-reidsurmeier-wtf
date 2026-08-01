"""D11.a RED — S6 three-state mask classifier.

Per addendum-v2: each pixel of each impression is classified
{none, visible, covered, support} relative to the rest of the stack.

Bottom-up sweep over print order. Thresholds from interface contract B-5:
- τ_vis = 0.40  — minimum α for an impression to count as printed
- τ_cov = 0.50  — minimum α of a LATER impression that hides this one
- τ_sup = 0.10  — minimum α for an impression to count as faint support
"""

from __future__ import annotations

import numpy as np


def test_pure_paper_pixel_returns_none() -> None:
    from backend.services.v23.stages.s6_three_state_mask import classify_three_state

    alphas = np.zeros((3, 4, 4), dtype=np.float32)
    state = classify_three_state(alphas)
    assert state.shape == (3, 4, 4)
    # 0 = none in the canonical encoding
    assert (state == 0).all()


def test_only_one_impression_visible_returns_visible() -> None:
    from backend.services.v23.stages.s6_three_state_mask import (
        STATE_VISIBLE,
        classify_three_state,
    )

    alphas = np.zeros((2, 4, 4), dtype=np.float32)
    alphas[0, :, :] = 0.8  # impression 0 fully visible everywhere
    state = classify_three_state(alphas)
    assert (state[0] == STATE_VISIBLE).all()


def test_later_impression_covers_earlier() -> None:
    """Two stacked impressions: i=0 covered, i=1 visible."""
    from backend.services.v23.stages.s6_three_state_mask import (
        STATE_COVERED,
        STATE_VISIBLE,
        classify_three_state,
    )

    alphas = np.zeros((2, 4, 4), dtype=np.float32)
    alphas[0, :, :] = 0.8  # earlier — covered by later
    alphas[1, :, :] = 0.7  # later — visible
    state = classify_three_state(alphas)
    assert (state[0] == STATE_COVERED).all()
    assert (state[1] == STATE_VISIBLE).all()


def test_faint_underprint_classified_support() -> None:
    """Earlier impression with α between τ_sup and τ_vis = support."""
    from backend.services.v23.stages.s6_three_state_mask import (
        STATE_SUPPORT,
        STATE_VISIBLE,
        classify_three_state,
    )

    alphas = np.zeros((2, 4, 4), dtype=np.float32)
    alphas[0, :, :] = 0.20  # faint support (between 0.10 and 0.40)
    alphas[1, :, :] = 0.7
    state = classify_three_state(alphas)
    assert (state[0] == STATE_SUPPORT).all()
    assert (state[1] == STATE_VISIBLE).all()


def test_state_constants_distinct() -> None:
    from backend.services.v23.stages.s6_three_state_mask import (
        STATE_COVERED,
        STATE_NONE,
        STATE_SUPPORT,
        STATE_VISIBLE,
    )

    assert len({STATE_NONE, STATE_VISIBLE, STATE_COVERED, STATE_SUPPORT}) == 4


def test_classify_with_partial_overlap() -> None:
    """Different regions classified differently in the same impression mask."""
    from backend.services.v23.stages.s6_three_state_mask import (
        STATE_COVERED,
        STATE_NONE,
        STATE_VISIBLE,
        classify_three_state,
    )

    alphas = np.zeros((2, 4, 4), dtype=np.float32)
    # impression 0 prints on right half
    alphas[0, :, 2:] = 0.7
    # impression 1 prints only on bottom-right quadrant
    alphas[1, 2:, 2:] = 0.7
    state = classify_three_state(alphas)
    # impression 0 — top right is visible, bottom right is covered
    assert (state[0, :2, 2:] == STATE_VISIBLE).all()
    assert (state[0, 2:, 2:] == STATE_COVERED).all()
    assert (state[0, :, :2] == STATE_NONE).all()
    # impression 1 — bottom right visible only
    assert (state[1, 2:, 2:] == STATE_VISIBLE).all()


def test_summary_stats_per_impression() -> None:
    from backend.services.v23.stages.s6_three_state_mask import summarise_states

    state = np.array(
        [
            [[1, 1, 0, 0], [1, 1, 0, 0]],  # impression 0: 4 visible, 4 none
            [[2, 2, 0, 0], [2, 2, 0, 0]],  # impression 1: 4 covered, 4 none
        ],
        dtype=np.uint8,
    )
    summary = summarise_states(state)
    assert len(summary) == 2
    assert summary[0]["visible_pct"] == 50.0
    assert summary[1]["covered_pct"] == 50.0


def test_summarise_includes_all_state_labels() -> None:
    from backend.services.v23.stages.s6_three_state_mask import summarise_states

    state = np.full((1, 4, 4), 3, dtype=np.uint8)  # all support
    summary = summarise_states(state)
    assert summary[0]["support_pct"] == 100.0
    assert summary[0]["visible_pct"] == 0.0
