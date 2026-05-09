"""Tool registry exposed to the LLM agent.

Two layers:

1. `make_tools(session)` returns a `(name -> callable)` map. Each callable
   takes a JSON-decoded `arguments` dict and returns a JSON-serializable
   result. Closures over `session` so the agent never sees raw landmarks.

2. `ANALYSIS_TOOL_SCHEMAS` is the OpenAI/Ollama-compatible tool schema list:
   `[{"type": "function", "function": {name, description, parameters}}]`.

The terminal tool is `submit_report` whose `parameters` schema is the
`ScreeningReport` JSON Schema. When the agent calls `submit_report` we
validate against the Pydantic model and break the loop on success — that
call is the structured output.
"""
from __future__ import annotations

from typing import Any, Callable

from .oak_adapter import CaptureSession
from .schemas import ScreeningReport
from .signal_processing import (
    REGION_WEIGHTS,
    blink_rate,
    face_for_task,
    jaw_tremor,
    mouth_corner_asymmetry,
    regional_motion,
    session_summary,
)


ToolFn = Callable[[dict[str, Any]], Any]


def make_tools(session: CaptureSession) -> dict[str, ToolFn]:
    """Build the per-session tool registry. Closes over session."""

    def _get_session_info(_: dict) -> dict:
        return session_summary(session)

    def _get_regional_motion(args: dict) -> dict:
        face = face_for_task(session, args.get("task"))
        return regional_motion(face)

    def _get_jaw_tremor(args: dict) -> dict:
        # Tremor is best assessed at rest. Default to the rest task if available.
        task = args.get("task") or "rest_seated"
        face = face_for_task(session, task)
        if face is None:
            face = session.face
        return jaw_tremor(face)

    def _get_blink_rate(args: dict) -> dict:
        face = face_for_task(session, args.get("task"))
        return blink_rate(face)

    def _get_mouth_asymmetry(args: dict) -> dict:
        # Asymmetry is best read during expression task.
        task = args.get("task") or "facial_expression"
        face = face_for_task(session, task)
        if face is None:
            face = session.face
        return mouth_corner_asymmetry(face)

    return {
        "get_session_info": _get_session_info,
        "get_regional_motion": _get_regional_motion,
        "get_jaw_tremor": _get_jaw_tremor,
        "get_blink_rate": _get_blink_rate,
        "get_mouth_asymmetry": _get_mouth_asymmetry,
    }


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / Ollama function-calling format).
# ---------------------------------------------------------------------------

_TASK_PARAM = {
    "type": "string",
    "description": (
        "Optional capture-task name to restrict the analysis window "
        "(e.g. 'rest_seated', 'facial_expression', 'speech'). "
        "Omit to analyze the full session."
    ),
}


ANALYSIS_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_session_info",
            "description": (
                "Return high-level metadata about the capture: patient_id, "
                "session_id, duration, FPS, available tasks, face coverage, "
                "and the regional clinical weights used downstream. Call this "
                "first to orient yourself before requesting metrics."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regional_motion",
            "description": (
                "Range-of-motion and velocity per facial region "
                "(chin/jaw, lower lip, upper lip, mouth corners, cheeks, "
                "eyelids, neck), each tagged with its clinical weight. "
                "Returns a `composite_expressivity_score` (weighted average "
                "of normalized region RoM). Lower is more suggestive of "
                "MDS-UPDRS 3.2 hypomimia. Call during the 'facial_expression' "
                "task for the most informative signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {"task": _TASK_PARAM},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_jaw_tremor",
            "description": (
                "Spectral analysis of chin motion (anchored to the inter-ocular "
                "midline to remove head sway) in the 3–7 Hz band. Reports the "
                "dominant frequency, whether it falls in the parkinsonian "
                "4–6 Hz range, the in-band power fraction, and spectral "
                "peakedness. Best evaluated during 'rest_seated'. Maps to "
                "MDS-UPDRS 3.17 (rest tremor, jaw)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"task": _TASK_PARAM},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_blink_rate",
            "description": (
                "Blinks per minute over the requested window. Adult resting "
                "norms are ~15–20/min; sustained values <8/min are supportive "
                "of MDS-UPDRS 3.2 hypomimia. Eyelids are LOW-weight in our "
                "regional priors — treat reduced blink as supportive evidence, "
                "not a primary signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {"task": _TASK_PARAM},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mouth_asymmetry",
            "description": (
                "Compare left vs right mouth-corner range of motion. Returns "
                "the side with reduced mobility and a normalized asymmetry "
                "ratio in [0, 1]. Asymmetric facial bradykinesia (>0.3) is "
                "consistent with unilateral PD onset. Best evaluated during "
                "'facial_expression'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"task": _TASK_PARAM},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Terminal tool — structured output.
# ---------------------------------------------------------------------------

def submit_report_schema() -> dict:
    """Build the submit_report tool schema from the Pydantic model.

    We pass `ScreeningReport.model_json_schema()` directly as the `parameters`
    block. The model is forced to populate every required field before it
    can call the tool, which is exactly the structured-output contract.
    """
    schema = ScreeningReport.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": (
                "FINAL action. Submit the structured screening report. Call "
                "exactly once when you have enough evidence to characterize "
                "the patient. After this call the session ends. Populate "
                "`motor_signs` only with signs you have direct numeric "
                "evidence for; mark `detected: false` when you ruled a sign "
                "out. Cite the tool names in `evidence_tool_calls`."
            ),
            "parameters": schema,
        },
    }


def all_tool_schemas() -> list[dict]:
    """Schemas + terminal tool, ready to send to Ollama."""
    return [*ANALYSIS_TOOL_SCHEMAS, submit_report_schema()]


# Re-export so callers can introspect.
__all__ = [
    "ANALYSIS_TOOL_SCHEMAS",
    "REGION_WEIGHTS",
    "all_tool_schemas",
    "make_tools",
    "submit_report_schema",
]
