"""CLI entrypoint: load `data.json`, run agent, print the report.

Usage:
    python -m parkinson_agent.run_demo                    # data/data.json
    python -m parkinson_agent.run_demo path/to/other.json # custom payload

Requires a running Ollama server with the chosen model pulled (see README).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .agent import run_screening_agent
from .input_schema import KnowledgePayload


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAYLOAD = REPO_ROOT / "data" / "data.json"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PAYLOAD
    if not path.exists():
        print(f"[run_demo] Payload not found: {path}", file=sys.stderr)
        print("[run_demo] Generate one with: python -m models.generate_knowledge", file=sys.stderr)
        return 1

    payload = KnowledgePayload.from_json_file(path)
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    print(f"[run_demo] payload : {path}")
    print(f"[run_demo] session : {payload.patient_id} / {payload.session_id}")
    has_model = payload.model_inference is not None
    print(f"[run_demo] sections: clinical_features={'✓' if payload.clinical_features else '·'} "
          f"model_inference={'✓' if has_model else '·'}")
    print(f"[run_demo] LLM     : {model} (Ollama)")
    print(f"[run_demo] Running screening agent...")

    result = run_screening_agent(payload, model=model)
    print(result.report.model_dump_json(indent=2))
    print(
        f"[run_demo] iterations={result.iterations}, "
        f"tool_calls={len(result.tool_calls)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
