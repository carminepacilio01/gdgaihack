"""Pytest fixtures shared across the test suite."""
from __future__ import annotations

import pytest

from parkinson_agent.run_demo import synthetic_session


@pytest.fixture
def session():
    """Synthetic face-only session with hypomimia + jaw tremor + L-side asymmetry."""
    return synthetic_session()
