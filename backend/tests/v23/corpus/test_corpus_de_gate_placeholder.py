"""Ring 5 — full-flow ΔE regression gate.

Two tests per corpus fixture:
1. ``test_corpus_fixture_directory_exists`` — fast sanity, runs in default CI
2. ``test_corpus_baseline_run`` — gated behind ``WOODBLOCK_RUN_CORPUS=1`` env.
   Pre-resizes the corpus image (long-edge 192 px on CPU JAX so the solver
   fits in <60s/fixture) and records actual ΔE values to
   ``corpus/baseline-<utc>.json``. xfails on Tier-1 gate miss until GPU
   JAX + spectral tier ship the ΔE budget.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from PIL import Image

from backend.tests.v23.corpus.conftest import CorpusFixture

CORPUS_RESIZE_LONG_EDGE = int(os.environ.get("WOODBLOCK_CORPUS_LONG_EDGE", "192"))
CORPUS_RUN_FLAG = "WOODBLOCK_RUN_CORPUS"


def test_corpus_fixture_directory_exists(corpus_fixture: CorpusFixture) -> None:
    """Run today — every fixture in corpus_tiers.yaml is on disk."""
    assert corpus_fixture.path.is_dir(), f"missing corpus dir: {corpus_fixture.path}"
    originals = list(corpus_fixture.path.glob("original.*"))
    assert originals, f"no original.* in {corpus_fixture.path}"


def _resize_for_corpus_run(src_path: Path, dst_dir: Path, long_edge: int) -> Path:
    """Resize the corpus original so CPU JAX solver fits in a sane wall-time."""
    dst = dst_dir / f"corpus_input_{src_path.stem}.png"
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(dst)
    return dst


@pytest.mark.skipif(
    not os.environ.get(CORPUS_RUN_FLAG),
    reason=f"set {CORPUS_RUN_FLAG}=1 to run full corpus regression (slow on CPU JAX)",
)
@pytest.mark.xfail(reason="ΔE gates not yet hit on CPU JAX t1_mixbox; awaits GPU + spectral tier")
def test_corpus_baseline_run(corpus_fixture: CorpusFixture, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)
    from backend.services.v23 import session as _sess

    importlib.reload(_sess)
    from backend.services.v23 import orchestrator as _orch

    importlib.reload(_orch)
    from backend.mcp.tools import core as _core

    importlib.reload(_core)

    src_image = next(corpus_fixture.path.glob("original.*"))
    resized = _resize_for_corpus_run(src_image, tmp_path, CORPUS_RESIZE_LONG_EDGE)

    t0 = time.perf_counter()
    r = _core.propose_stack(str(resized), solve_profile="fast")
    elapsed = time.perf_counter() - t0
    assert r.ok is True, r.errors

    de_mean = r.data["reconstruction_dE_mean"]
    de_p95 = r.data["reconstruction_dE_p95"]

    # Append to a baseline log under repo corpus/ — survives test
    log_path = (
        Path(__file__).resolve().parents[4]
        / "corpus"
        / f"baseline-{time.strftime('%Y-%m-%d')}.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "fixture_id": corpus_fixture.id,
        "tier": corpus_fixture.tier,
        "resized_long_edge": CORPUS_RESIZE_LONG_EDGE,
        "solver_status": r.data["solver_status"],
        "solver_wall_s": r.data["solver_wall_s"],
        "test_wall_s": round(elapsed, 2),
        "dE_mean": de_mean,
        "dE_p95": de_p95,
        "gate_de_mean": corpus_fixture.de_mean_gate,
        "gate_de_p95": corpus_fixture.de_p95_gate,
        "impression_count": r.data["impression_count"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    existing = []
    if log_path.is_file():
        try:
            existing = json.loads(log_path.read_text())
        except json.JSONDecodeError:
            existing = []
    existing.append(record)
    log_path.write_text(json.dumps(existing, indent=2))

    assert de_mean is not None, "solver did not produce dE_mean"
    assert de_mean <= corpus_fixture.de_mean_gate, (
        f"{corpus_fixture.id}: ΔE mean {de_mean:.2f} > {corpus_fixture.de_mean_gate}"
    )
    assert de_p95 <= corpus_fixture.de_p95_gate, (
        f"{corpus_fixture.id}: ΔE p95 {de_p95:.2f} > {corpus_fixture.de_p95_gate}"
    )
