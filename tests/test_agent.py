"""Agent loop tests with a mocked Ollama client.

Validates:
- Tool calls are dispatched correctly and results fed back.
- Schema-validation failure on submit_report triggers a retry.
- Iteration cap is enforced.
- Unknown tools are reported back to the model rather than crashing.
- Plain-text responses get nudged back into tool-use mode.

Zero LLM cost — no Ollama server is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from parkinson_agent.agent import run_screening_agent


VALID_REPORT = {
    "patient_id": "DEMO-P-0001",
    "session_id": "demo-session-001",
    "overall_risk_level": "elevated",
    "asymmetry_detected": True,
    "motor_signs": [
        {
            "name": "hypomimia",
            "detected": True,
            "side": "bilateral",
            "severity": 2,
            "confidence": "moderate",
            "key_metrics": {"composite_expressivity_score": 0.026},
            "evidence_tool_calls": ["get_regional_motion"],
            "rationale": "Composite expressivity is reduced; chin/lower-lip RoM very low.",
        },
        {
            "name": "mouth_corner_asymmetry",
            "detected": True,
            "side": "left",
            "severity": 1,
            "confidence": "moderate",
            "key_metrics": {"asymmetry_ratio": 0.7},
            "evidence_tool_calls": ["get_mouth_asymmetry"],
            "rationale": "Left mouth corner mobility is ~70% lower than right.",
        },
    ],
    "quality_issues": [],
    "flagged_findings": ["Reduced overall expressivity, asymmetric on the left."],
    "recommended_followup": ["In-person MDS-UPDRS Part III evaluation"],
    "clinician_notes": "Findings consistent with early left-onset facial parkinsonism.",
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
        responses = [
            _msg([_tool_use("get_session_info")]),
            _msg([_tool_use("get_regional_motion")]),
            _msg([_tool_use("submit_report", VALID_REPORT)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)

        assert result.report.patient_id == "DEMO-P-0001"
        assert result.report.overall_risk_level.value == "elevated"
        assert result.report.asymmetry_detected is True
        assert len(result.report.motor_signs) == 2
        assert result.iterations == 3
        assert {c["name"] for c in result.tool_calls} == {
            "get_session_info",
            "get_regional_motion",
            "submit_report",
        }

    def test_get_regional_motion_returns_payload_data(self, payload):
        # Capture the tool result the agent saw on the second turn.
        responses = [
            _msg([_tool_use("get_regional_motion")]),
            _msg([_tool_use("submit_report", VALID_REPORT)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)

        motion_call = next(
            c for c in result.tool_calls if c["name"] == "get_regional_motion"
        )
        # Tool surfaced the payload's pre-computed composite score unchanged.
        assert motion_call["result"]["composite_expressivity_score"] == 0.026


class TestAgentValidation:
    def test_invalid_report_triggers_retry(self, payload):
        invalid_report = {**VALID_REPORT, "overall_risk_level": "very_high"}
        responses = [
            _msg([_tool_use("submit_report", invalid_report)]),
            _msg([_tool_use("submit_report", VALID_REPORT)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)

        assert result.iterations == 2
        assert result.report.overall_risk_level.value == "elevated"


class TestAgentLimits:
    def test_iteration_cap_enforced(self, payload):
        forever = _msg([_tool_use("get_session_info")])
        client = MagicMock()
        client.chat.side_effect = [forever] * 10

        with pytest.raises(RuntimeError, match="did not produce"):
            run_screening_agent(payload, client=client, max_iterations=3)

    def test_unknown_tool_returns_error_to_model(self, payload):
        responses = [
            _msg([_tool_use("nonexistent_tool")]),
            _msg([_tool_use("submit_report", VALID_REPORT)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)
        assert result.tool_calls[0]["name"] == "nonexistent_tool"
        assert "unknown_tool" in str(result.tool_calls[0]["result"])

    def test_text_only_response_nudges_model(self, payload):
        responses = [
            _msg(content="I think the patient has hypomimia."),
            _msg([_tool_use("submit_report", VALID_REPORT)]),
        ]
        client = MagicMock()
        client.chat.side_effect = responses

        result = run_screening_agent(payload, client=client, max_iterations=5)
        assert result.iterations == 2
