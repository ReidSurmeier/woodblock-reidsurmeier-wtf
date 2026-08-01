"""Unified evaluation result schema.

The single source of truth for what an eval run emits per fixture. Every engine
(stub, tan, km_nnls, qwen_layered) MUST produce an `EvalResult` so the corpus
runner can compare results across engines without per-engine adapter code.

Reference: validation-system-v1.md section 2 + 10.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

EngineName = Literal["tan", "km_nnls", "qwen_layered", "stub"]


@dataclass
class SummaryStats:
    """Distribution summary for a per-pixel error map (e.g. ΔE2000)."""

    mean: float
    p50: float
    p95: float
    p99: float
    max: float


@dataclass
class EvalResult:
    """One row of the eval corpus.

    Required: fixture_id, image/recon/heatmap paths, dE2000 stats.
    Optional metrics: pigment_iou, block_iou, chromatic_class_recovery
    (populated only when the fixture has ground-truth annotations for them).
    """

    fixture_id: str
    image_path: str
    recon_path: str
    dE_heatmap_path: str
    dE2000: SummaryStats
    pigment_iou: dict[str, float] | None = None
    block_iou: float | None = None
    chromatic_class_recovery: float | None = None
    block_count: int = 0
    pigment_count: int = 0
    print_order: list[str] = field(default_factory=list)
    duration_ms: int = 0
    git_sha: str = ""
    engine: EngineName = "stub"
    params: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_pass(self) -> bool:
        """Master gate: pass iff mean ΔE2000 < 1.5 AND p95 ΔE2000 < 3.0.

        Thresholds from validation-system-v1.md section 10. Strictly less-than:
        boundary values (1.5, 3.0) are FAIL — they sit at the perceptibility limit
        and we want headroom.
        """
        return self.dE2000.mean < 1.5 and self.dE2000.p95 < 3.0

    def to_json(self) -> str:
        """Serialize to indented JSON. SummaryStats is unrolled via asdict."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> EvalResult:
        """Deserialize from JSON produced by `to_json`.

        Rehydrates the nested SummaryStats dataclass — plain dict reconstruction
        would leave it as a dict and break dotted access (.mean, .p95).
        """
        d = json.loads(s)
        d["dE2000"] = SummaryStats(**d["dE2000"])
        return cls(**d)
