"""Validate that the OAK JSON payload schema parses correctly."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from parkinson_agent.input_schema import PatientMetricsPayload


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


class TestSamplePayloads:
    def test_demo_session_loads(self):
        payload = PatientMetricsPayload.from_json_file(SAMPLES_DIR / "demo_session.json")
        assert payload.patient_id == "DEMO-P-0001"
        assert payload.metrics.regional_motion is not None
        assert payload.metrics.jaw_tremor is not None
        assert payload.metrics.regional_motion.composite_expressivity_score is not None

    def test_healthy_session_loads(self):
        payload = PatientMetricsPayload.from_json_file(SAMPLES_DIR / "healthy_session.json")
        assert payload.patient_id == "DEMO-P-0002"
        assert payload.metrics.jaw_tremor.in_parkinsonian_range_4_6hz is False

    def test_weights_match_user_priors(self):
        payload = PatientMetricsPayload.from_json_file(SAMPLES_DIR / "demo_session.json")
        # The weights the project owner specified.
        assert payload.regional_weights["chin_jaw"] == 1.0
        assert payload.regional_weights["lower_lip"] == 1.0
        assert payload.regional_weights["upper_lip"] == 0.6
        assert payload.regional_weights["mouth_corners"] == 0.6
        assert payload.regional_weights["cheeks"] == 0.4
        assert payload.regional_weights["eyelids"] == 0.2
        assert payload.regional_weights["neck"] == 0.2


class TestSchemaValidation:
    def test_missing_required_field_raises(self):
        bad = {
            # patient_id missing
            "session_id": "x",
            "duration_s": 10.0,
            "regional_weights": {"chin_jaw": 1.0},
            "metrics": {},
        }
        with pytest.raises(ValidationError):
            PatientMetricsPayload.model_validate(bad)

    def test_partial_metrics_section_is_ok(self):
        # OAK can omit individual metric sections (e.g. jaw_tremor unavailable).
        partial = {
            "patient_id": "P",
            "session_id": "S",
            "duration_s": 10.0,
            "regional_weights": {"chin_jaw": 1.0},
            "metrics": {
                "blink_rate": {
                    "valid": False,
                    "reason": "insufficient_data",
                },
            },
        }
        payload = PatientMetricsPayload.model_validate(partial)
        assert payload.metrics.blink_rate is not None
        assert payload.metrics.blink_rate.valid is False
        assert payload.metrics.regional_motion is None
