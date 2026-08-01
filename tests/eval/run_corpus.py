"""Eval corpus runner — discovers fixtures, runs pipeline, writes results JSON.

Usage:
    python -m tests.eval.run_corpus --tier=A --output=eval_results.json
    python -m tests.eval.run_corpus --fixture=hiroshige_edo_116 --engine=tan

V2 scaffold: `eval_fixture` returns a passing-by-default stub. MVP-A wires the
actual engine + scoring math.

Reference: validation-system-v1.md sections 2 + 10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .eval_result import EvalResult, SummaryStats


def discover_fixtures(corpus_root: Path, tier: str | None = None) -> list[dict[str, Any]]:
    """Walk `corpus_root` for `*/annotations.json` files.

    Each fixture's directory holds the original image + ground-truth annotations
    plus any cached intermediates. `tier` filters by `annotations.json["tier"]`
    so we can run "A" (gold, hand-annotated) separately from "B" (auto-labeled).
    """
    fixtures: list[dict[str, Any]] = []
    for ann_path in corpus_root.glob("*/annotations.json"):
        try:
            ann = json.loads(ann_path.read_text())
        except json.JSONDecodeError:
            # Skip malformed annotation files — log loudly so we notice in CI.
            print(f"[WARN] malformed annotations.json: {ann_path}")
            continue
        if tier is None or ann.get("tier") == tier:
            fixtures.append({"path": ann_path.parent, "ann": ann})
    return fixtures


def eval_fixture(fixture_dir: Path, ann: dict[str, Any], engine: str = "stub") -> EvalResult:
    """Run the pipeline against one fixture and score the result.

    V2 stub: returns a passing-by-default result. MVP-A:
        1. Load image from fixture_dir.
        2. Run engine to get (masks, palette, order).
        3. forward_render_km(masks, palette, order).
        4. delta_e2000_image(original, recon).
        5. summarize -> SummaryStats.
        6. is_pass via threshold.
    """
    return EvalResult(
        fixture_id=ann["image_id"],
        image_path=str(fixture_dir),
        recon_path="",
        dE_heatmap_path="",
        dE2000=SummaryStats(mean=0.0, p50=0.0, p95=0.0, p99=0.0, max=0.0),
        engine=engine,  # type: ignore[arg-type]
        passed=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_corpus",
        description="Run the validation eval corpus and emit results JSON.",
    )
    p.add_argument("--corpus", type=Path, default=Path("corpus"), help="Corpus root directory.")
    p.add_argument("--tier", default=None, help="Filter to a single tier (A, B, ...).")
    p.add_argument("--fixture", default=None, help="Run a single fixture by image_id.")
    p.add_argument(
        "--engine",
        default="stub",
        choices=["tan", "km_nnls", "qwen_layered", "stub"],
        help="Which decomposition engine to evaluate.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="Where to write the results JSON array.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    fixtures = discover_fixtures(args.corpus, args.tier)
    if args.fixture:
        fixtures = [f for f in fixtures if f["ann"]["image_id"] == args.fixture]

    results = [eval_fixture(f["path"], f["ann"], args.engine) for f in fixtures]

    # Roundtrip through to_json so we know on-disk format == in-memory format.
    payload = [json.loads(r.to_json()) for r in results]
    args.output.write_text(json.dumps(payload, indent=2))

    passed = sum(1 for r in results if r.passed)
    print(f"Ran {len(results)} fixtures · {passed} passed · output -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
