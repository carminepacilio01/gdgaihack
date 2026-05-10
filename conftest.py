"""Pytest fixtures shared across the test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from parkinson_agent.input_schema import PatientMetricsPayload


SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


@pytest.fixture
def payload() -> PatientMetricsPayload:
    """Demo payload with hypomimia + jaw tremor + L-side mouth asymmetry."""
    return PatientMetricsPayload.from_json_file(SAMPLES_DIR / "demo_session.json")


@pytest.fixture
def healthy_payload() -> PatientMetricsPayload:
    """Demo payload with normal expressivity, no tremor, symmetric mouth."""
    return PatientMetricsPayload.from_json_file(SAMPLES_DIR / "healthy_session.json")
