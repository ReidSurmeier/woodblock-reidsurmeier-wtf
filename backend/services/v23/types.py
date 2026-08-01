"""D2 Pydantic types for the v23-MCP woodblock_stack package.

Authority: ``/tmp/research-v23-mcp-interfaces.md`` §2.1 (Pigment) and
``/tmp/research-v23-data-model.md`` (Block, Impression, Mask, PullGroup, Plan).

All types are frozen + JSON-serialisable. Tensor fields live in the
``Plan.tensors`` dict and are NOT carried inline (see §3 of data-model).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Pigment
# ---------------------------------------------------------------------------

PigmentFamily = Literal[
    "cream",
    "cool",
    "flesh",
    "warm",
    "shadow",
    "detail",
    "accent",
]


class Pigment(BaseModel):
    """Hue + density + opacity recipe applied during one Impression.

    Wraps the legacy ``backend.algorithms.decomposition.palette_extract.Pigment``
    (4-field record: id/name/rgb/hex) with v23 fields:

    * ``family``: hue-family slot in a Strategy template
    * ``density``: paste-to-water ratio, normalized 0..1
    * ``opacity_curve``: dict of dilution → k_opacity
    * ``calibration_source``: ``generic_mixbox_13`` or ``fitted_<cal_id>``
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    rgb: tuple[int, int, int]
    hex: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    family: PigmentFamily
    density: float = Field(ge=0.0, le=1.0)
    opacity_curve: dict[str, float] = Field(default_factory=dict)
    calibration_source: str = Field(
        default="generic_mixbox_13",
        pattern=r"^(generic_mixbox_13|fitted_[a-z0-9_\-]{1,40})$",
    )

    def k_opacity(self, dilution: float = 1.0) -> float:
        if not self.opacity_curve:
            return 1.0
        key = min(
            self.opacity_curve,
            key=lambda d: abs(float(d.rstrip("x")) - dilution),
        )
        return self.opacity_curve[key]


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

BlockMaterial = Literal["maple_plywood", "shina_plywood", "cherry"]


class Block(BaseModel):
    """A physical woodblock. Carries many Impressions (Pace-Editions)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    face_ids: tuple[str, ...]
    material: BlockMaterial = "maple_plywood"
    sheet_w_mm: PositiveFloat
    sheet_h_mm: PositiveFloat
    impression_ids: tuple[str, ...] = ()
    dsatur_color: NonNegativeInt = 0


# ---------------------------------------------------------------------------
# Mask + Impression
# ---------------------------------------------------------------------------

ConfidenceLabel = Literal[
    "visible-in-final",
    "inferred-underprint",
    "ambiguous",
]


class Mask(BaseModel):
    """Three-state mask reference for one Impression.

    The state grid + optional soft α live on disk (PNG / .npy). The
    Pydantic record only carries paths + dims + confidence so the model
    stays JSON-serializable. Loader functions in ``v23.io`` materialise
    the arrays when needed.
    """

    model_config = ConfigDict(frozen=True)

    width: PositiveInt
    height: PositiveInt
    state_path: Path
    alpha_path: Path | None = None
    confidence: ConfidenceLabel
    confidence_dE: float = Field(ge=0.0)
    confidence_alt_ids: list[str] = Field(default_factory=list)

    @field_validator("state_path", "alpha_path")
    @classmethod
    def _must_be_png(cls, v: Path | None) -> Path | None:
        if v is None:
            return v
        v = Path(v)
        if v.suffix.lower() != ".png":
            raise ValueError(f"mask path must end in .png; got {v}")
        return v


class Impression(BaseModel):
    """One atomic Pigment+Mask application at one Order step on one Block."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    order_step: PositiveInt
    block_id: str = Field(min_length=1)
    block_face_id: str = Field(pattern=r"^.+::face_[a-z0-9]+$")
    pull_group: NonNegativeInt = 0
    pigment_id: str = Field(min_length=1)
    mask: Mask
    hidden_coverage_ref: str = Field(min_length=1)
    luminance_okL: float = Field(ge=0.0, le=1.0)
    coverage_pct: float = Field(ge=0.0, le=100.0)
    notes: str = ""


# ---------------------------------------------------------------------------
# PullGroup
# ---------------------------------------------------------------------------


class PullGroup(BaseModel):
    """Display-time aggregation. Derived from Impressions; not a solver entity."""

    model_config = ConfigDict(frozen=True)

    block_face_id: str
    order_step: PositiveInt
    pull_group: NonNegativeInt
    impression_ids: tuple[str, ...]
    label: str = ""

    @model_validator(mode="after")
    def _must_have_impressions(self) -> PullGroup:
        if not self.impression_ids:
            raise ValueError("pull group must reference >= 1 impression")
        return self


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

StrategyTemplate = Literal["portrait_emma", "landscape", "high_chroma_graphic"]
SolveProfile = Literal["fast", "default", "thorough"]


class Plan(BaseModel):
    """Stack + Masks + Block assignments + Order + confidence labels.

    Addendum-v3 fix 3: NO ``mode`` field. The fast/default/thorough knob
    is named ``solve_profile``. Plan rejects unknown fields (``extra=forbid``)
    so accidental writes of ``mode=`` raise at validation.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    schema_version: Literal["v23.0"] = "v23.0"
    target_image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: PositiveInt
    height: PositiveInt
    pigments: list[Pigment]
    blocks: list[Block]
    impressions: list[Impression]
    pull_groups: list[PullGroup] = Field(default_factory=list)
    strategy_template: StrategyTemplate | None = None
    solve_profile: SolveProfile = "default"
    reconstruction_dE_mean: float = Field(ge=0.0)
    reconstruction_dE_p95: float = Field(ge=0.0)
    solver_wall_s: float = Field(ge=0.0)
    tensors: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
