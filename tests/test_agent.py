"""Agent loop tests with a mocked Ollama client.

Validates:
- Tool calls are dispatched correctly and results fed back.
- Schema-validation failure on submit_report triggers a retry.
- Iteration cap is enforced.
- Unknown tools are reported back to the model rather than crashing.
- Plain-text responses get nudged back into tool-use mode.
- The new get_model_inference tool surfaces the upstream model output.

Zero LLM cost — no Ollama server is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from parkinson_agent.agent import run_screening_agent


def _valid_report_for(payload) -> dict:
    return {
        "patient_id": payload.patient_id,
        "session_id": payload.session_id,
        "overall_risk_level": "borderline",
        "asymmetry_detected": False,
        "motor_signs": [
            {
                "name": "hypomimia",
                "detected": True,
                "side": "bilateral",
                "severity": 1,
                "confidence": "moderate",
                "key_metrics": {"composite_expressivity_score": 0.04},
                "evidence_tool_calls": ["get_regional_motion"],
                "rationale": "Composite expressivity is reduced.",
            },
        ],
        "quality_issues": [],
        "flagged_findings": ["Reduced expressivity."],
        "recommended_followup": ["In-person MDS-UPDRS evaluation."],
        "clinician_notes": "Borderline screening result.",
    }


def _msg(tool_calls: list[dict] | None = None, content: str = "") -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or [],
        }
    }


def _tool_use(name: str, args: dict | None = None) -> dict:
    return {"function": {"name": name, "arguments": args or {}}}


class TestAgentHappyPath:
    def test_agent_completes_with_valid_report(self, payload):
        report = _valid_report_for(payload)
        responses = [
            _msg([_tool_use("get_session_info")]),
            _msg([_tool_use("get_regional_motion")]),
            _msg([_tool_use("submit_report", report)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)

        assert result.report.patient_id == payload.patient_id
        assert result.report.session_id == payload.session_id
        assert result.iterations == 3
        assert {c["name"] for c in result.tool_calls} == {
            "get_session_info",
            "get_regional_motion",
            "submit_report",
        }

    def test_get_regional_motion_returns_payload_data(self, payload):
        report = _valid_report_for(payload)
        responses = [
            _msg([_tool_use("get_regional_motion")]),
            _msg([_tool_use("submit_report", report)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)
        rm = next(c for c in result.tool_calls if c["name"] == "get_regional_motion")
        assert rm["result"]["valid"] is True
        # Tool surfaced the payload's pre-computed score unchanged.
        if payload.clinical_features.regional_motion is not None:
            assert (
                rm["result"]["composite_expressivity_score"]
                == payload.clinical_features.regional_motion.composite_expressivity_score
            )

    def test_get_model_inference_handles_missing_section(self, payload):
        # Sample data.json has no model_inference; tool returns the missing marker.
        report = _valid_report_for(payload)
        responses = [
            _msg([_tool_use("get_model_inference")]),
            _msg([_tool_use("submit_report", report)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)
        mi = next(c for c in result.tool_calls if c["name"] == "get_model_inference")
        if payload.model_inference is None:
            assert mi["result"]["valid"] is False
            assert "section_missing" in mi["result"]["reason"]


class TestAgentValidation:
    def test_invalid_report_triggers_retry(self, payload):
        valid = _valid_report_for(payload)
        invalid = {**valid, "overall_risk_level": "very_high"}
        responses = [
            _msg([_tool_use("submit_report", invalid)]),
            _msg([_tool_use("submit_report", valid)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)
        assert result.iterations == 2
        assert result.report.overall_risk_level.value == "borderline"


class TestAgentLimits:
    def test_iteration_cap_enforced(self, payload):
        forever = _msg([_tool_use("get_session_info")])
        client = MagicMock()
        client.chat.side_effect = [forever] * 10

        with pytest.raises(RuntimeError, match="did not produce"):
            run_screening_agent(payload, client=client, max_iterations=3)

    def test_unknown_tool_returns_error_to_model(self, payload):
        report = _valid_report_for(payload)
        responses = [
            _msg([_tool_use("nonexistent_tool")]),
            _msg([_tool_use("submit_report", report)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)
        assert result.tool_calls[0]["name"] == "nonexistent_tool"
        assert "unknown_tool" in str(result.tool_calls[0]["result"])

    def test_text_only_response_nudges_model(self, payload):
        report = _valid_report_for(payload)
        responses = [
            _msg(content="I think the patient has hypomimia."),
            _msg([_tool_use("submit_report", report)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)
        assert result.iterations == 2
