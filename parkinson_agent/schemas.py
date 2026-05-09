"""Pydantic schemas for the face-only screening report.

The agent's terminal `submit_report` tool is bound to `ScreeningReport`. By
forcing structured output through a tool call we guarantee the LLM produces a
schema-valid object — no JSON-in-text parsing, no regex.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    BORDERLINE = "borderline"
    ELEVATED = "elevated"


class Confidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class SignName(str, Enum):
    """Face-only motor signs the agent can flag.

    Mapped loosely to MDS-UPDRS Part III items. We deliberately keep the
    surface area small: this is screening, not diagnosis.
    """

    HYPOMIMIA = "hypomimia"                            # 3.2 facial masking
    JAW_TREMOR = "jaw_tremor"                          # 3.17 rest tremor (jaw)
    REDUCED_BLINK_RATE = "reduced_blink_rate"          # supportive of 3.2
    MOUTH_CORNER_ASYMMETRY = "mouth_corner_asymmetry"  # supportive
    EYELID_REDUCED_MOTION = "eyelid_reduced_motion"    # supportive


class MotorSign(BaseModel):
    name: SignName
    detected: bool
    side: str | None = Field(
        default=None,
        description="'left', 'right', 'bilateral', or null if not lateralized.",
    )
    severity: int | None = Field(
        default=None,
        ge=0,
        le=4,
        description="MDS-UPDRS-style 0–4 severity. Null if not applicable.",
    )
    confidence: Confidence
    key_metrics: dict = Field(
        default_factory=dict,
        description="Subset of the numeric tool outputs that justify this sign.",
    )
    evidence_tool_calls: list[str] = Field(
        default_factory=list,
        description="Names of the tools whose results back this sign.",
    )
    rationale: str = Field(
        ...,
        description="One or two sentences linking the metrics to the call.",
    )


class ScreeningReport(BaseModel):
    """Final structured output of a screening session."""

    patient_id: str
    session_id: str
    overall_risk_level: RiskLevel
    asymmetry_detected: bool = False
    motor_signs: list[MotorSign] = Field(default_factory=list)
    quality_issues: list[str] = Field(default_factory=list)
    flagged_findings: list[str] = Field(
        default_factory=list,
        description="One-line bullets for the clinician's at-a-glance view.",
    )
    recommended_followup: list[str] = Field(default_factory=list)
    clinician_notes: str = ""
    disclaimer: str = (
        "Screening tool. Not a diagnostic device. "
        "Findings require in-person evaluation by a qualified clinician."
    )
