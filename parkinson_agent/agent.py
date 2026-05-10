"""Agent loop: Ollama chat + tool-use, terminal `submit_report` for structured output.

Decoupled from the wire protocol where it matters:
- The Ollama Python client is imported lazily so tests can pass a mock.
- The loop dispatches tool calls, validates the terminal report against the
  Pydantic schema, and retries on validation failure (bounded by max_iterations).

Why Ollama: zero API cost for the hackathon, runs locally on the same laptop
as the OAK device. Default `llama3.2:3b` — small enough to be fast on CPU
(Intel Macs) and supports function calling. We pin temperature=0 by default
for repeatability.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .input_schema import KnowledgePayload
from .schemas import ScreeningReport
from .tools import all_tool_schemas, make_tools


SYSTEM_PROMPT = """\
You are a clinical screening assistant for early Parkinson's disease.

Your only input is a `data.json` payload produced by an upstream model
that already analyzed the patient's face capture. You do NOT see raw
video, landmarks, or do any signal processing yourself. You call tools
that read sections of that JSON and return numeric results. Your job is
to interpret those numbers in MDS-UPDRS terms and produce a structured
report that helps a clinician decide whether deeper assessment is warranted.

WHAT THE UPSTREAM MODEL PROVIDES (each as a separate tool):
- `get_regional_motion` — per-region range-of-motion + composite score.
- `get_jaw_tremor` — 3–7 Hz spectral analysis of the chin.
- `get_mouth_asymmetry` — left vs right mouth-corner mobility.
- `get_model_inference` — output of a TCN classifier (PD probability),
  if the upstream model ran it. May be missing.

CLINICAL TARGETS (MDS-UPDRS Part III, face-only subset):
- 3.2  Hypomimia (facial masking) — primary screening target.
- 3.17 Rest tremor (jaw) — assess from chin spectral analysis.
- Supportive: mouth-corner asymmetry.

REGIONAL WEIGHTS (clinical priors already applied in composite scores):
  HIGH:    chin/jaw, lower lip
  MID:     upper lip, mouth corners
  LOW-MID: cheeks
Eyelids and neck are NOT measured (the upstream sparse landmark set
lacks them) — do not assert eyelid hypokinesia or reduced blink rate.

HOW TO COMBINE SIGNALS:
- Clinical features are interpretable and primary.
- The model probability (`get_model_inference`) is a strong but
  black-box signal. Use it to corroborate or temper your conclusions —
  not to override clinical reasoning.
- Convergence between model probability and clinical features = high
  confidence. Disagreement = lower confidence + flag in
  `clinician_notes`.

PROCEDURE:
1. Call `get_session_info` first to see what's available.
2. Call the relevant clinical-feature tools.
3. Call `get_model_inference` if it's available.
4. Reason about the numbers, then call `submit_report` exactly once.

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
- Write `clinician_notes` so a busy doctor can read it in 15 seconds.
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
        msg = msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
    return msg


def _strip_code_fence(text: str) -> str:
    """Strip markdown fences from a model's text response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    # Crop to the outermost JSON object so trailing commentary doesn't break parsing.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def run_screening_agent_simple(
    payload: KnowledgePayload,
    *,
    client: Any = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> AgentResult:
    """One-shot agent: no tool-use, single LLM call, JSON-in-text response.

    Designed for small local models (≤1.5B) where forcing complex JSON
    Schema enforcement via tool-use causes hangs. The full payload is
    inlined in the prompt and the model is asked to return a single JSON
    object matching `ScreeningReport`. Validation is done client-side.
    """
    if client is None:
        client = _default_client()
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    knowledge = json.dumps(payload.model_dump(exclude_none=True), indent=2)
    prompt = f"""You are a clinical screening assistant for early Parkinson's disease.

The upstream model produced this knowledge JSON for the patient:

{knowledge}

REGIONAL WEIGHTS (clinical priors): chin/jaw=HIGH(1.0), lower_lip=HIGH(1.0), \
upper_lip=MID(0.6), mouth_corners=MID(0.6), cheeks=LOW-MID(0.4). Eyelids and \
neck are NOT measured — never assert blink rate or eyelid hypokinesia.

INTERPRET the metrics with MDS-UPDRS Part III:
- 3.2 Hypomimia → composite_expressivity_score (lower = more hypomimic) and \
chin/lower-lip RoM (HIGH-weight regions).
- 3.17 Jaw rest tremor → jaw_tremor.dominant_frequency_hz, especially when \
in_parkinsonian_range_4_6hz is true and spectral_peakedness is high.
- Mouth-corner asymmetry → mouth_asymmetry.asymmetry_ratio > 0.3 with a \
clearly less_mobile_side suggests unilateral PD onset.
- model_inference.pd_probability (if present) is supportive — combine, \
don't override clinical reasoning.

OVERALL RISK:
- low: no convincing motor signs.
- borderline: one weak/ambiguous sign or quality issues.
- elevated: at least one strong sign (e.g. composite clearly reduced AND \
in-band jaw tremor) or convergent signs.

OUTPUT exactly one JSON object, no markdown, no commentary, with this shape:

{{
  "patient_id": "{payload.patient_id}",
  "session_id": "{payload.session_id}",
  "overall_risk_level": "low" | "borderline" | "elevated",
  "asymmetry_detected": true | false,
  "motor_signs": [
    {{
      "name": "hypomimia" | "jaw_tremor" | "mouth_corner_asymmetry" | \
"reduced_blink_rate" | "eyelid_reduced_motion",
      "detected": true | false,
      "side": "left" | "right" | "bilateral" | null,
      "severity": 0 | 1 | 2 | 3 | 4 | null,
      "confidence": "low" | "moderate" | "high",
      "key_metrics": {{}},
      "evidence_tool_calls": [],
      "rationale": "one or two sentences"
    }}
  ],
  "quality_issues": [],
  "flagged_findings": ["short bullets for the clinician"],
  "recommended_followup": ["short bullets"],
  "clinician_notes": "concise 15-second read"
}}

Be conservative. Output ONLY the JSON object."""

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    msg = _extract_message(response)
    raw = msg.get("content", "") or ""
    text = _strip_code_fence(raw)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Model output is not valid JSON: {exc}\n--- raw output ---\n{raw}"
        )

    report = ScreeningReport.model_validate(data)
    return AgentResult(
        report=report,
        transcript=[{"iteration": 1, "message": msg}],
        tool_calls=[],
        iterations=1,
    )


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
    payload: KnowledgePayload,
    *,
    client: Any = None,
    model: str | None = None,
    max_iterations: int = 10,
    temperature: float = 0.0,
) -> AgentResult:
    """Run the tool-use loop until `submit_report` produces a valid report.

    Parameters
    ----------
    payload : KnowledgePayload
        The `data.json` document from the upstream model, already validated.
    client : Any, optional
        Anything with `.chat(model=..., messages=..., tools=..., options=...)`
        returning an Ollama-shaped response. Defaults to a real Ollama client.
    model : str, optional
        Ollama model tag. Defaults to env `OLLAMA_MODEL` or `llama3.2:3b`.
    max_iterations : int
        Hard cap on round-trips with the LLM.
    temperature : float
        Sampling temperature. 0.0 for repeatability.
    """
    if client is None:
        client = _default_client()
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    tool_registry = make_tools(payload)
    tool_schemas = all_tool_schemas()

    user_prompt = (
        f"Screen patient {payload.patient_id} (session {payload.session_id}). "
        f"Capture duration: {payload.duration_s:.1f}s. "
        f"Inspect the upstream model's knowledge with the tools, then submit "
        f"a structured report. In the report set "
        f"`patient_id={payload.patient_id!r}` and "
        f"`session_id={payload.session_id!r}`."
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

        messages.append({
            "role": "assistant",
            "content": msg.get("content", "") or "",
            "tool_calls": msg.get("tool_calls") or [],
        })

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
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
                except Exception as exc:
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
