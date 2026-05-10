"""CLI entrypoint: load `data.json`, run agent, print the report.

Usage:
    python -m parkinson_agent.run_demo                       # tool-use mode (default)
    python -m parkinson_agent.run_demo --simple              # single-shot, no tool-use
    python -m parkinson_agent.run_demo path/to/other.json
    python -m parkinson_agent.run_demo --simple path/to/other.json

Modes:
    tool-use (default): agent calls tools iteratively, structured output
        forced via JSON Schema. Best quality, needs a model that handles
        function calling well (3B+ recommended).
    --simple:           one LLM call, full payload inlined in the prompt,
        JSON-in-text response. Use this on small local models (≤1.5B) or
        when tool-use hangs.

Requires a running Ollama server with the chosen model pulled (see README).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import run_screening_agent, run_screening_agent_simple
from .input_schema import KnowledgePayload


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAYLOAD = REPO_ROOT / "data" / "data.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("payload", nargs="?", default=str(DEFAULT_PAYLOAD),
                        help="data.json path (default: data/data.json)")
    parser.add_argument("--simple", action="store_true",
                        help="Single-shot, no tool-use. For small models / hanging issues.")
    args = parser.parse_args()

    path = Path(args.payload)
    if not path.exists():
        print(f"[run_demo] Payload not found: {path}", file=sys.stderr)
        print("[run_demo] Generate one with: python -m models.generate_knowledge", file=sys.stderr)
        return 1

    payload = KnowledgePayload.from_json_file(path)
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    mode = "simple (no tool-use)" if args.simple else "tool-use"

    print(f"[run_demo] payload : {path}")
    print(f"[run_demo] session : {payload.patient_id} / {payload.session_id}")
    has_model = payload.model_inference is not None
    print(f"[run_demo] sections: clinical_features={'✓' if payload.clinical_features else '·'} "
          f"model_inference={'✓' if has_model else '·'}")
    print(f"[run_demo] LLM     : {model} (Ollama, {mode})")
    print(f"[run_demo] Running screening agent...")

    if args.simple:
        result = run_screening_agent_simple(payload, model=model)
    else:
        result = run_screening_agent(payload, model=model)

    print(result.report.model_dump_json(indent=2))
    print(
        f"[run_demo] iterations={result.iterations}, "
        f"tool_calls={len(result.tool_calls)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
