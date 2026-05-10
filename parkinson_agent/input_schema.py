"""Schema for `data.json` — the only input the agent reads.

The upstream model (under `models/`) ingests the OAK landmark CSV, runs
its own analysis (clinical features and optionally a TCN classifier),
and emits a single `data.json` document. This module validates that
document into a typed `KnowledgePayload` the agent can reason about.

If the upstream pipeline emits slightly different field names, update
this schema. The agent code keys off these names.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Clinical features (computed upstream from the raw landmarks).
# ---------------------------------------------------------------------------

class RegionStat(BaseModel):
    weight: float = Field(..., description="Clinical prior for this region.")
    range_of_motion: float = Field(..., description="Peak-to-peak excursion, normalized.")
    velocity_p95: float | None = None
    velocity_median: float | None = None
    available_landmarks: int | None = Field(
        default=None,
        description="How many landmarks of this region were present in the capture.",
    )


class RegionalMotion(BaseModel):
    valid: bool
    per_region: dict[str, RegionStat] | None = None
    composite_expressivity_score: float | None = Field(
        default=None,
        description="Weighted average of per-region range-of-motion. Lower = more hypomimic.",
    )
    duration_s: float | None = None
    fps: float | None = None
    n_frames: int | None = None
    reason: str | None = Field(default=None, description="Failure reason when valid=false.")


class JawTremor(BaseModel):
    valid: bool
    dominant_frequency_hz: float | None = None
    in_parkinsonian_range_4_6hz: bool | None = None
    in_band_fraction_of_total: float | None = None
    spectral_peakedness: float | None = None
    anchor_used: str | None = None
    fps: float | None = None
    reason: str | None = None


class MouthAsymmetry(BaseModel):
    valid: bool
    rom_left: float | None = None
    rom_right: float | None = None
    asymmetry_ratio: float | None = None
    less_mobile_side: str | None = None
    weight_mouth_corners: float | None = None
    reason: str | None = None


class ClinicalFeatures(BaseModel):
    """All upstream-computed clinical features. Sections may be omitted
    if the upstream model didn't compute them."""

    regional_motion: RegionalMotion | None = None
    jaw_tremor: JawTremor | None = None
    mouth_asymmetry: MouthAsymmetry | None = None


# ---------------------------------------------------------------------------
# ML model inference (optional — only present if the TCN classifier ran).
# ---------------------------------------------------------------------------

class ModelInference(BaseModel):
    model_name: str
    version: str | None = None
    pd_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate probability of Parkinson's, [0, 1].",
    )
    n_windows_analyzed: int | None = None
    window_probabilities_summary: dict[str, float] | None = Field(
        default=None,
        description="Optional aggregate stats over per-window probabilities (mean, p50, p95, min, max).",
    )
    notes: str | None = None


# ---------------------------------------------------------------------------
# Patient metadata (most fields optional; many sessions have only patient_id).
# ---------------------------------------------------------------------------

class Metadata(BaseModel):
    age: float | None = None
    sex: str | None = None
    ground_truth_label: int | None = Field(
        default=None,
        description="0 healthy, 1 PD. Usually None at inference time.",
    )


class Quality(BaseModel):
    face_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of frames with a detected face. <0.5 = data quality issue.",
    )
    n_landmarks_available: int | None = None
    missing_modalities: list[str] = Field(
        default_factory=list,
        description="Signals the upstream model could not compute (e.g. blink rate when no eyelid landmarks).",
    )


# ---------------------------------------------------------------------------
# Top-level document.
# ---------------------------------------------------------------------------

class KnowledgePayload(BaseModel):
    """Top-level `data.json` produced by the upstream model.

    The agent reads exactly this. Each section is surfaced through a
    dedicated tool so the LLM can inspect them on demand.
    """

    patient_id: str
    session_id: str
    captured_at: str | None = None      # ISO-8601
    duration_s: float
    n_frames: int | None = None
    fps: float | None = None
    metadata: Metadata = Field(default_factory=Metadata)
    regional_weights: dict[str, float] = Field(
        ...,
        description=(
            "Clinical priors per facial region. Echoed in the payload so the "
            "agent can reason about which signals are primary vs supportive."
        ),
    )
    clinical_features: ClinicalFeatures = Field(default_factory=ClinicalFeatures)
    model_inference: ModelInference | None = None
    quality: Quality | None = None

    # Convenience constructors --------------------------------------------

    @classmethod
    def from_json_file(cls, path: str | Path) -> "KnowledgePayload":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    @classmethod
    def from_json_str(cls, raw: str) -> "KnowledgePayload":
        return cls.model_validate(json.loads(raw))
