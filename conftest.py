"""Pytest fixtures shared across the test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from parkinson_agent.input_schema import KnowledgePayload


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = REPO_ROOT / "data" / "data.json"


@pytest.fixture
def payload() -> KnowledgePayload:
    """The agent's input — `data.json` produced by the upstream pipeline."""
    if not DEFAULT_DATA.exists():
        pytest.skip(
            f"{DEFAULT_DATA.relative_to(REPO_ROOT)} missing — "
            f"run `python -m models.generate_knowledge` first."
        )
    return KnowledgePayload.from_json_file(DEFAULT_DATA)
