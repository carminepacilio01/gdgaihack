"""Smoke-test entrypoint: load a sample JSON and run the agent end-to-end.

Usage:
    python -m parkinson_agent.run_demo                    # uses samples/demo_session.json
    python -m parkinson_agent.run_demo path/to/file.json  # custom payload

Requires a running Ollama server with the chosen model pulled. See README.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .agent import run_screening_agent
from .input_schema import PatientMetricsPayload


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = REPO_ROOT / "samples" / "demo_session.json"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE
    if not path.exists():
        print(f"[run_demo] Payload not found: {path}", file=sys.stderr)
        return 1

    payload = PatientMetricsPayload.from_json_file(path)
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    print(f"[run_demo] Payload: {path}")
    print(f"[run_demo] Patient: {payload.patient_id} / Session: {payload.session_id}")
    print(f"[run_demo] Running screening agent with model={model}...")

    result = run_screening_agent(payload, model=model)
    print(result.report.model_dump_json(indent=2))
    print(
        f"[run_demo] iterations={result.iterations}, "
        f"tool_calls={len(result.tool_calls)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
