"""Live integration test against a local Ollama server.

Skipped automatically when Ollama is not reachable, so this is safe in CI
without a model server. To run locally:

    ollama serve &
    ollama pull llama3.2:3b
    pytest tests/test_integration.py -v

Override the model with `OLLAMA_MODEL=qwen2.5:3b pytest ...`.
"""
from __future__ import annotations

import os

import pytest

from parkinson_agent.agent import run_screening_agent


def _ollama_reachable() -> bool:
    try:
        import urllib.request

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable(),
    reason="Ollama server not reachable; skipping live integration test.",
)


def test_agent_produces_report(payload):
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    result = run_screening_agent(payload, model=model, max_iterations=12)

    report = result.report
    assert report.patient_id == payload.patient_id
    assert report.session_id == payload.session_id
    assert report.overall_risk_level.value in {"low", "borderline", "elevated"}

    called = {c["name"] for c in result.tool_calls}
    assert called & {
        "get_regional_motion",
        "get_jaw_tremor",
        "get_mouth_asymmetry",
    }, f"Agent did not call any feature tool. Called: {called}"
