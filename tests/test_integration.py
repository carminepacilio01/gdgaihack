"""Live integration test against a local Ollama server.

Skipped automatically when Ollama is not reachable, so this is safe in CI
without a model server. To run locally:

    ollama serve &
    ollama pull llama3.1
    pytest tests/test_integration.py -v

Override the model with `OLLAMA_MODEL=qwen2.5 pytest ...`.
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


def test_face_only_synthetic_session(session):
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    result = run_screening_agent(session, model=model, max_iterations=12)

    # Soft assertions: any clinically reasonable agent should at least
    # flag SOMETHING on this synthetic patient (hypomimia + jaw tremor +
    # left mouth asymmetry are all built into the data).
    report = result.report
    assert report.patient_id == "DEMO-P-0001"
    assert report.overall_risk_level.value in {"borderline", "elevated"}, (
        f"Expected borderline/elevated for the impaired-face session, "
        f"got {report.overall_risk_level.value}"
    )

    # The agent should have actually queried at least one face metric.
    called = {c["name"] for c in result.tool_calls}
    assert called & {
        "get_regional_motion",
        "get_jaw_tremor",
        "get_mouth_asymmetry",
    }, f"Agent did not call any face metric tool. Called: {called}"
