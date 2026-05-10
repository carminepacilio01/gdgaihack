"""Validate the data.json schema and the sample payload."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from parkinson_agent.input_schema import KnowledgePayload


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = REPO_ROOT / "data" / "data.json"


@pytest.mark.skipif(not SAMPLE.exists(), reason="data/data.json missing")
class TestSamplePayload:
    def test_loads(self):
        p = KnowledgePayload.from_json_file(SAMPLE)
        assert p.patient_id
        assert p.session_id
        assert p.duration_s > 0

    def test_clinical_features_present(self):
        p = KnowledgePayload.from_json_file(SAMPLE)
        cf = p.clinical_features
        assert cf.regional_motion is not None
        assert cf.jaw_tremor is not None
        assert cf.mouth_asymmetry is not None

    def test_user_weights_round_trip(self):
        p = KnowledgePayload.from_json_file(SAMPLE)
        assert p.regional_weights["chin_jaw"] == 1.0
        assert p.regional_weights["lower_lip"] == 1.0
        assert p.regional_weights["mouth_corners"] == 0.6


class TestSchemaValidation:
    def test_minimum_required_fields(self):
        minimal = {
            "patient_id": "p", "session_id": "s",
            "duration_s": 1.0,
            "regional_weights": {"chin_jaw": 1.0},
        }
        p = KnowledgePayload.model_validate(minimal)
        assert p.clinical_features.regional_motion is None
        assert p.model_inference is None

    def test_missing_required_field_raises(self):
        bad = {"session_id": "s", "duration_s": 1.0, "regional_weights": {}}
        with pytest.raises(ValidationError):
            KnowledgePayload.model_validate(bad)

    def test_pd_probability_bounds(self):
        with pytest.raises(ValidationError):
            KnowledgePayload.model_validate({
                "patient_id": "p", "session_id": "s", "duration_s": 1.0,
                "regional_weights": {"chin_jaw": 1.0},
                "model_inference": {"model_name": "x", "pd_probability": 1.5},
            })

    def test_partial_clinical_features(self):
        # Upstream model may compute only some features.
        partial = {
            "patient_id": "p", "session_id": "s", "duration_s": 1.0,
            "regional_weights": {"chin_jaw": 1.0},
            "clinical_features": {
                "jaw_tremor": {"valid": False, "reason": "no_data"},
            },
        }
        p = KnowledgePayload.model_validate(partial)
        assert p.clinical_features.jaw_tremor is not None
        assert p.clinical_features.regional_motion is None
