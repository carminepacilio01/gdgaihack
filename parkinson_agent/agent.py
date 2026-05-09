"""Agent loop: Ollama chat + tool-use, terminal `submit_report` for structured output.

Decoupled from the wire protocol where it matters:
- The Ollama Python client is imported lazily so tests can pass a mock.
- The loop dispatches tool calls, validates the terminal report against the
  Pydantic schema, and retries on validation failure (bounded by max_iterations).

Why Ollama: zero API cost for the hackathon, runs locally on the same laptop
as the OAK device. Use `llama3.1` or `qwen2.5` — both support function
calling. We pin temperature=0 by default for repeatability.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .oak_adapter import CaptureSession
from .schemas import ScreeningReport
from .tools import all_tool_schemas, make_tools


SYSTEM_PROMPT = """\
You are a clinical screening assistant for early Parkinson's disease.

You receive a capture session from a Luxonis OAK camera that recorded the
patient's FACE only — there is no hand or gait data. You analyze the
session by calling tools that return numeric features. You never see raw
landmarks; you only see the tool outputs.

CLINICAL TARGETS (MDS-UPDRS Part III, face-only subset):
- 3.2  Hypomimia (facial masking) — primary screening target.
- 3.17 Rest tremor (jaw) — assess at rest.
- Supportive: reduced blink rate, mouth-corner asymmetry, eyelid hypokinesia.

REGIONAL WEIGHTS (clinical priors built into the metrics):
  HIGH:    chin/jaw, lower lip
  MID:     upper lip, mouth corners
  LOW-MID: cheeks
  LOW:     eyelids, neck
The composite expressivity score in `get_regional_motion` already applies
these weights. Treat low-weight regions as supportive, not primary.

PROCEDURE:
1. Call `get_session_info` first.
2. Call `get_regional_motion` (during 'facial_expression' if available).
3. Call `get_jaw_tremor` (during 'rest_seated' if available).
4. Call `get_blink_rate` and `get_mouth_asymmetry` for supportive evidence.
5. Reason about the numbers, then call `submit_report` exactly once.

REPORT GUIDANCE:
- `overall_risk_level`:
    'low'        — no convincing motor signs, normal expressivity.
    'borderline' — one weak/ambiguous sign or quality issues.
    'elevated'   — at least one strong sign (e.g. composite score clearly
                   reduced AND in-band jaw tremor) or convergent signs.
- Add a `MotorSign` entry only if you have a numeric tool output that
  supports it. Cite the tool name(s) in `evidence_tool_calls`.
- If a tool returns `valid: false`, log a `quality_issues` entry and avoid
  asserting that sign with high confidence.
- Be conservative. This is screening for clinician review, not diagnosis.
"""


@dataclass
class AgentResult:
    report: ScreeningReport
    transcript: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 0


def _default_client():
    """Lazy-import the Ollama client. Avoids forcing the dep on test runs."""
    import ollama  # noqa: PLC0415

    host = os.environ.get("OLLAMA_HOST")
    if host:
        return ollama.Client(host=host)
    return ollama.Client()


def _extract_message(response: Any) -> dict:
    """Normalize an Ollama chat response to a plain message dict.

    The Python client returns either a dict or a pydantic-style object
    depending on version. Tests pass plain dicts — both must work.
    """
    if isinstance(response, dict):
        msg = response.get("message", {})
    else:
        msg = getattr(response, "message", {})
    if not isinstance(msg, dict):
        # pydantic-style: convert to dict
        msg = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
    return msg


def _tool_call_args(tc: Any) -> tuple[str, dict]:
    """Extract (name, arguments) from a tool_call entry."""
    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", {})
    if isinstance(fn, dict):
        name = fn.get("name", "")
        args = fn.get("arguments", {}) or {}
    else:
        name = getattr(fn, "name", "")
        args = getattr(fn, "arguments", {}) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return name, args


def run_screening_agent(
    session: CaptureSession,
    *,
    client: Any = None,
    model: str | None = None,
    max_iterations: int = 10,
    temperature: float = 0.0,
) -> AgentResult:
    """Run the tool-use loop until `submit_report` produces a valid report.

    Parameters
    ----------
    session : CaptureSession
        The capture to analyze.
    client : Any, optional
        Anything with `.chat(model=..., messages=..., tools=..., options=...)`
        returning an Ollama-shaped response. Defaults to a real Ollama client.
    model : str, optional
        Ollama model tag. Defaults to env `OLLAMA_MODEL` or `llama3.1`.
    max_iterations : int
        Hard cap on round-trips with the LLM.
    temperature : float
        Sampling temperature. 0.0 for repeatability.
    """
    if client is None:
        client = _default_client()
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")

    tool_registry = make_tools(session)
    tool_schemas = all_tool_schemas()

    user_prompt = (
        f"Screen patient {session.patient_id} (session {session.session_id}). "
        f"Capture duration: {session.duration_s:.1f}s. "
        f"Use the tools to inspect the data, then submit a structured report."
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    transcript: list[dict] = []
    tool_calls_log: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        response = client.chat(
            model=model,
            messages=messages,
            tools=tool_schemas,
            options={"temperature": temperature},
        )
        msg = _extract_message(response)
        transcript.append({"iteration": iteration, "message": msg})

        # Persist the assistant turn so the next request has full history.
        messages.append({
            "role": "assistant",
            "content": msg.get("content", "") or "",
            "tool_calls": msg.get("tool_calls") or [],
        })

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # The model didn't call a tool. Nudge it back on rails.
            messages.append({
                "role": "user",
                "content": (
                    "You must use the available tools and finish by calling "
                    "`submit_report`. Do not answer in free text."
                ),
            })
            continue

        for tc in tool_calls:
            name, args = _tool_call_args(tc)

            if name == "submit_report":
                try:
                    report = ScreeningReport.model_validate(args)
                except ValidationError as e:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({
                            "validation_error": e.errors(include_url=False),
                            "hint": "Re-call submit_report with a corrected payload.",
                        }),
                    })
                    tool_calls_log.append({
                        "name": name,
                        "input": args,
                        "result": {"validation_error": True},
                    })
                    continue
                tool_calls_log.append({
                    "name": name,
                    "input": args,
                    "result": {"submitted": True},
                })
                return AgentResult(
                    report=report,
                    transcript=transcript,
                    tool_calls=tool_calls_log,
                    iterations=iteration,
                )

            fn = tool_registry.get(name)
            if fn is None:
                result: Any = {"error": "unknown_tool", "name": name}
            else:
                try:
                    result = fn(args)
                except Exception as exc:  # surfaces to the model as a tool error
                    result = {"error": "tool_exception", "detail": str(exc)}

            tool_calls_log.append({"name": name, "input": args, "result": result})
            messages.append({
                "role": "tool",
                "content": json.dumps(result, default=str),
            })

    raise RuntimeError(
        f"Agent did not produce a valid ScreeningReport within "
        f"{max_iterations} iterations."
    )
