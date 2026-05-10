"""Tool registry exposed to the LLM agent.

The agent's data source is a `PatientMetricsPayload` (parsed from the JSON
that OAK delivers). Tools are thin readers that surface one section of
that payload at a time — the agent decides what to look at.

Two layers:

1. `make_tools(payload)` returns a `(name -> callable)` map. Each callable
   takes a JSON-decoded `arguments` dict and returns a JSON-serializable
   result. Closures over `payload`.

2. `ANALYSIS_TOOL_SCHEMAS` is the OpenAI/Ollama-compatible tool schema list:
   `[{"type": "function", "function": {name, description, parameters}}]`.

The terminal tool is `submit_report` whose `parameters` schema is the
`ScreeningReport` JSON Schema. When the agent calls `submit_report` we
validate against the Pydantic model and break the loop on success — that
call is the structured output.
"""
from __future__ import annotations

from typing import Any, Callable

from .input_schema import PatientMetricsPayload
from .schemas import ScreeningReport


ToolFn = Callable[[dict[str, Any]], Any]


def _missing(field: str) -> dict:
    return {
        "valid": False,
        "reason": f"metric_missing_in_payload:{field}",
    }


def make_tools(payload: PatientMetricsPayload) -> dict[str, ToolFn]:
    """Build the per-payload tool registry. Closes over `payload`."""

    def _get_session_info(_: dict) -> dict:
        return {
            "patient_id": payload.patient_id,
            "session_id": payload.session_id,
            "captured_at": payload.captured_at,
            "duration_s": payload.duration_s,
            "capture_fps": payload.capture_fps,
            "device": payload.device,
            "face_coverage": payload.face_coverage,
            "tasks": [t.model_dump() for t in payload.tasks],
            "regional_weights": payload.regional_weights,
            "available_metrics": [
                k for k, v in payload.metrics.model_dump(exclude_none=True).items()
                if v is not None
            ],
        }

    def _get_regional_motion(_: dict) -> dict:
        m = payload.metrics.regional_motion
        if m is None:
            return _missing("regional_motion")
        return m.model_dump(exclude_none=True)

    def _get_jaw_tremor(_: dict) -> dict:
        m = payload.metrics.jaw_tremor
        if m is None:
            return _missing("jaw_tremor")
        return m.model_dump(exclude_none=True)

    def _get_blink_rate(_: dict) -> dict:
        m = payload.metrics.blink_rate
        if m is None:
            return _missing("blink_rate")
        return m.model_dump(exclude_none=True)

    def _get_mouth_asymmetry(_: dict) -> dict:
        m = payload.metrics.mouth_asymmetry
        if m is None:
            return _missing("mouth_asymmetry")
        return m.model_dump(exclude_none=True)

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

ANALYSIS_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_session_info",
            "description": (
                "Return high-level metadata for the capture: patient_id, "
                "session_id, duration, FPS, available tasks, face coverage, "
                "the regional clinical weights, and which metric sections "
                "the payload contains. Call this first to orient yourself."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regional_motion",
            "description": (
                "Per-region range-of-motion and velocity (chin/jaw, lower lip, "
                "upper lip, mouth corners, cheeks, eyelids, neck), each tagged "
                "with its clinical weight, plus a `composite_expressivity_score` "
                "(weighted average of normalized region RoM). Lower composite "
                "is more suggestive of MDS-UPDRS 3.2 hypomimia. Reduced motion "
                "in HIGH-weight regions (chin/jaw, lower lip) is the strongest "
                "face-only signal."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_jaw_tremor",
            "description": (
                "Spectral analysis of chin motion in the 3–7 Hz band. Fields: "
                "dominant frequency, whether it falls in the parkinsonian "
                "4–6 Hz range, the in-band power fraction, and spectral "
                "peakedness. Best evaluated during 'rest_seated'. Maps to "
                "MDS-UPDRS 3.17 (rest tremor, jaw)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_blink_rate",
            "description": (
                "Blinks per minute over the session. Adult resting norms are "
                "~15–20/min; sustained values <8/min are supportive of "
                "MDS-UPDRS 3.2 hypomimia. Eyelids carry a LOW weight in our "
                "regional priors — treat reduced blink as supportive evidence, "
                "not a primary signal."
            ),
            "parameters": {"type": "object", "properties": {}},
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
                "consistent with unilateral PD onset."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Terminal tool — structured output.
# ---------------------------------------------------------------------------

def submit_report_schema() -> dict:
    """Build the submit_report tool schema from the Pydantic model."""
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
                "out. Cite the tool names in `evidence_tool_calls`. The "
                "report is shown to the clinician — be clear and conservative."
            ),
            "parameters": schema,
        },
    }


def all_tool_schemas() -> list[dict]:
    """Schemas + terminal tool, ready to send to Ollama."""
    return [*ANALYSIS_TOOL_SCHEMAS, submit_report_schema()]


__all__ = [
    "ANALYSIS_TOOL_SCHEMAS",
    "all_tool_schemas",
    "make_tools",
    "submit_report_schema",
]
