"""Schema for the JSON payload that OAK delivers to this agent.

OAK runs the heavy lifting (FaceMesh inference, signal processing, regional
metrics) and emits a single JSON document. This module validates that
document into a typed `PatientMetricsPayload` the agent can reason about.

If your OAK side emits slightly different field names, update this schema
and the sample under `samples/`. Everything downstream is keyed on these
names.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class TaskWindow(BaseModel):
    name: str
    start_s: float
    end_s: float


class RegionStat(BaseModel):
    weight: float = Field(..., description="Clinical prior for this region.")
    range_of_motion: float = Field(..., description="Peak-to-peak excursion, normalized by inter-ocular distance.")
    velocity_p95: float | None = None
    velocity_median: float | None = None


class RegionalMotion(BaseModel):
    valid: bool
    task: str | None = None
    per_region: dict[str, RegionStat] | None = None
    composite_expressivity_score: float | None = None
    reason: str | None = None


class JawTremor(BaseModel):
    valid: bool
    task: str | None = None
    dominant_frequency_hz: float | None = None
    in_parkinsonian_range_4_6hz: bool | None = None
    in_band_fraction_of_total: float | None = None
    spectral_peakedness: float | None = None
    reason: str | None = None


class BlinkRate(BaseModel):
    valid: bool
    duration_s: float | None = None
    n_blinks: int | None = None
    blink_rate_per_min: float | None = None
    weight_eyelids: float | None = None
    reason: str | None = None


class MouthAsymmetry(BaseModel):
    valid: bool
    task: str | None = None
    rom_left: float | None = None
    rom_right: float | None = None
    asymmetry_ratio: float | None = None
    less_mobile_side: str | None = None
    weight_mouth_corners: float | None = None
    reason: str | None = None


class FaceMetrics(BaseModel):
    """Container for the per-area metrics OAK computed on-device."""

    regional_motion: RegionalMotion | None = None
    jaw_tremor: JawTremor | None = None
    blink_rate: BlinkRate | None = None
    mouth_asymmetry: MouthAsymmetry | None = None


class PatientMetricsPayload(BaseModel):
    """Top-level JSON document handed off from OAK to this agent."""

    patient_id: str
    session_id: str
    captured_at: str | None = None     # ISO-8601 timestamp
    duration_s: float
    capture_fps: float | None = None
    device: str | None = None
    tasks: list[TaskWindow] = Field(default_factory=list)
    face_coverage: float | None = Field(
        default=None,
        description="Fraction of frames where a face was detected. <0.5 = data-quality issue.",
    )
    regional_weights: dict[str, float] = Field(
        ...,
        description=(
            "Clinical priors per facial region. Echoed in the payload so the "
            "agent can reason about which signals are primary vs supportive."
        ),
    )
    metrics: FaceMetrics

    # ------------------------------------------------------------------
    # Convenience: load from a file path or a raw dict.
    # ------------------------------------------------------------------

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PatientMetricsPayload":
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))

    @classmethod
    def from_json_str(cls, raw: str) -> "PatientMetricsPayload":
        return cls.model_validate(json.loads(raw))
