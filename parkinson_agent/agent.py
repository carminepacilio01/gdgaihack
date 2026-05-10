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
import sys
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .input_schema import (
    JawTremor,
    KnowledgePayload,
    MouthAsymmetry,
    RegionalMotion,
)
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


# ---------------------------------------------------------------------------
# Deterministic clinical decisions.
#
# Small local models (≤1.5B) cannot be trusted to apply MDS-UPDRS thresholds
# correctly — they pattern-match on prompt examples rather than reason. So we
# compute every decision in Python from the upstream numbers, and ask the LLM
# only to write the natural-language explanations. This keeps the pipeline
# auditable (the rules are visible code) and works regardless of model size.
# ---------------------------------------------------------------------------

# Hypomimia thresholds on the composite expressivity score.
# Lower score = more hypomimic. Normal expressivity in our data ≈ 0.4–0.5.
HYPOMIMIA_SEVERE_BELOW = 0.15
HYPOMIMIA_MODERATE_BELOW = 0.25
HYPOMIMIA_MILD_BELOW = 0.35

# Jaw tremor: parkinsonian band + spectral evidence.
TREMOR_PEAKEDNESS_STRONG = 0.5
TREMOR_PEAKEDNESS_WEAK = 0.3
TREMOR_INBAND_FRAC_STRONG = 0.30
TREMOR_INBAND_FRAC_WEAK = 0.15

# Mouth-corner asymmetry: ratio thresholds.
ASYM_STRONG = 0.5
ASYM_MILD = 0.3


def _decide_hypomimia(rm: RegionalMotion | None) -> dict:
    if rm is None or not rm.valid or rm.composite_expressivity_score is None:
        return {
            "name": "hypomimia",
            "detected": False,
            "side": None,
            "severity": None,
            "confidence": "low",
            "key_metrics": {},
            "summary": "regional_motion data not available",
        }
    score = rm.composite_expressivity_score
    chin = (rm.per_region or {}).get("chin_jaw")
    lip = (rm.per_region or {}).get("lower_lip")
    metrics: dict[str, Any] = {"composite_expressivity_score": round(score, 4)}
    if chin:
        metrics["chin_jaw_rom"] = round(chin.range_of_motion, 4)
    if lip:
        metrics["lower_lip_rom"] = round(lip.range_of_motion, 4)

    if score < HYPOMIMIA_SEVERE_BELOW:
        return {**_sign("hypomimia", True, 3, "high", metrics, "bilateral"),
                "summary": f"composite {score:.3f} severely reduced"}
    if score < HYPOMIMIA_MODERATE_BELOW:
        return {**_sign("hypomimia", True, 2, "moderate", metrics, "bilateral"),
                "summary": f"composite {score:.3f} moderately reduced"}
    if score < HYPOMIMIA_MILD_BELOW:
        return {**_sign("hypomimia", True, 1, "low", metrics, "bilateral"),
                "summary": f"composite {score:.3f} mildly reduced"}
    return {**_sign("hypomimia", False, 0, "high", metrics, None),
            "summary": f"composite {score:.3f} within normal range"}


def _decide_jaw_tremor(jt: JawTremor | None) -> dict:
    if jt is None or not jt.valid:
        return {
            "name": "jaw_tremor",
            "detected": False,
            "side": None,
            "severity": None,
            "confidence": "low",
            "key_metrics": {},
            "summary": "jaw_tremor data not available",
        }
    f = jt.dominant_frequency_hz
    peak = jt.spectral_peakedness or 0.0
    in_band = jt.in_band_fraction_of_total or 0.0
    in_pd_band = bool(jt.in_parkinsonian_range_4_6hz)
    metrics = {
        "dominant_frequency_hz": round(f or 0.0, 3),
        "in_parkinsonian_range_4_6hz": in_pd_band,
        "spectral_peakedness": round(peak, 3),
        "in_band_fraction_of_total": round(in_band, 3),
    }
    if in_pd_band and peak > TREMOR_PEAKEDNESS_STRONG and in_band > TREMOR_INBAND_FRAC_STRONG:
        return {**_sign("jaw_tremor", True, 2, "high", metrics, None),
                "summary": f"strong rhythmic peak at {f:.2f} Hz in parkinsonian band"}
    if in_pd_band and (peak > TREMOR_PEAKEDNESS_WEAK or in_band > TREMOR_INBAND_FRAC_WEAK):
        return {**_sign("jaw_tremor", True, 1, "moderate", metrics, None),
                "summary": f"weak rhythmic peak at {f:.2f} Hz in parkinsonian band"}
    return {**_sign("jaw_tremor", False, 0, "high", metrics, None),
            "summary": f"dominant frequency {f:.2f} Hz outside parkinsonian band 4–6 Hz"}


def _decide_mouth_asymmetry(ma: MouthAsymmetry | None) -> dict:
    if ma is None or not ma.valid or ma.asymmetry_ratio is None:
        return {
            "name": "mouth_corner_asymmetry",
            "detected": False,
            "side": None,
            "severity": None,
            "confidence": "low",
            "key_metrics": {},
            "summary": "mouth_asymmetry data not available",
        }
    asym = ma.asymmetry_ratio
    side = ma.less_mobile_side
    metrics = {
        "asymmetry_ratio": round(asym, 3),
        "less_mobile_side": side,
        "rom_left": round(ma.rom_left or 0.0, 4),
        "rom_right": round(ma.rom_right or 0.0, 4),
    }
    if asym > ASYM_STRONG:
        return {**_sign("mouth_corner_asymmetry", True, 2, "high", metrics, side),
                "summary": f"strong asymmetry ratio {asym:.1%} on {side} side"}
    if asym > ASYM_MILD:
        return {**_sign("mouth_corner_asymmetry", True, 1, "moderate", metrics, side),
                "summary": f"mild asymmetry ratio {asym:.1%} on {side} side"}
    return {**_sign("mouth_corner_asymmetry", False, 0, "high", metrics, None),
            "summary": f"asymmetry ratio {asym:.1%} within normal range"}


def _sign(name: str, detected: bool, severity: int | None, confidence: str,
          key_metrics: dict, side: str | None) -> dict:
    return {
        "name": name,
        "detected": detected,
        "side": side,
        "severity": severity,
        "confidence": confidence,
        "key_metrics": key_metrics,
    }


def _decide_overall_risk(signs: list[dict]) -> str:
    strong = sum(
        1 for s in signs
        if s.get("detected") and (s.get("severity") or 0) >= 2
    )
    weak = sum(
        1 for s in signs
        if s.get("detected") and (s.get("severity") or 0) == 1
    )
    if strong >= 1:
        return "elevated"
    if weak >= 2:
        return "elevated"
    if weak == 1:
        return "borderline"
    return "low"


def _decide_clinical_state(payload: KnowledgePayload) -> dict:
    """Apply MDS-UPDRS thresholds deterministically. Returns the structured
    skeleton of the report; only the prose fields remain to be written."""
    cf = payload.clinical_features
    signs = [
        _decide_hypomimia(cf.regional_motion),
        _decide_jaw_tremor(cf.jaw_tremor),
        _decide_mouth_asymmetry(cf.mouth_asymmetry),
    ]
    asymmetry_detected = any(
        s["name"] == "mouth_corner_asymmetry" and s.get("detected")
        for s in signs
    )
    overall = _decide_overall_risk(signs)
    return {
        "overall_risk_level": overall,
        "asymmetry_detected": asymmetry_detected,
        "signs": signs,
    }


_RISK_LEVELS = {"low", "borderline", "elevated"}
_CONFIDENCE_LEVELS = {"low", "moderate", "high"}
_SIGN_NAMES = {
    "hypomimia",
    "jaw_tremor",
    "reduced_blink_rate",
    "mouth_corner_asymmetry",
    "eyelid_reduced_motion",
}
_SIGN_ALIASES = {
    "hypomimia": "hypomimia",
    "facial_masking": "hypomimia",
    "facial masking": "hypomimia",
    "masked_face": "hypomimia",
    "masked face": "hypomimia",
    "jaw_tremor": "jaw_tremor",
    "jaw tremor": "jaw_tremor",
    "rest_tremor_jaw": "jaw_tremor",
    "rest tremor jaw": "jaw_tremor",
    "rest_tremor": "jaw_tremor",
    "mouth_corner_asymmetry": "mouth_corner_asymmetry",
    "mouth corner asymmetry": "mouth_corner_asymmetry",
    "mouth_asymmetry": "mouth_corner_asymmetry",
    "mouth asymmetry": "mouth_corner_asymmetry",
    "lip_asymmetry": "mouth_corner_asymmetry",
    "lip asymmetry": "mouth_corner_asymmetry",
    "asymmetry": "mouth_corner_asymmetry",
    "reduced_blink_rate": "reduced_blink_rate",
    "reduced blink rate": "reduced_blink_rate",
    "blink_rate": "reduced_blink_rate",
    "eyelid_reduced_motion": "eyelid_reduced_motion",
    "eyelid reduced motion": "eyelid_reduced_motion",
}


def _canonicalize_report(data: dict) -> dict:
    """Repair common small-LLM mistakes before Pydantic validation.

    Small models often:
      - Echo the prompt placeholder ("low|borderline|elevated") instead of
        picking one value.
      - Use spaces instead of underscores in enum names.
      - Drop required fields entirely.
    We map known variants to canonical values and fall back to safe defaults.
    """
    if not isinstance(data, dict):
        return data

    # overall_risk_level: pick a sensible default if the model echoed the placeholder
    risk = str(data.get("overall_risk_level", "")).strip().lower()
    if risk not in _RISK_LEVELS:
        # Pipe-syntax echo, free text, etc. — default to borderline (conservative).
        data["overall_risk_level"] = "borderline"
    else:
        data["overall_risk_level"] = risk

    # asymmetry_detected: coerce stringy bools
    if isinstance(data.get("asymmetry_detected"), str):
        data["asymmetry_detected"] = (
            data["asymmetry_detected"].strip().lower() in {"true", "yes", "1"}
        )

    # motor_signs: normalize each entry
    signs = data.get("motor_signs") or []
    fixed_signs: list[dict] = []
    for s in signs:
        if not isinstance(s, dict):
            continue
        name_raw = str(s.get("name", "")).strip().lower()
        canon = _SIGN_ALIASES.get(name_raw)
        if canon is None and name_raw in _SIGN_NAMES:
            canon = name_raw
        if canon is None:
            # Skip unknown sign names rather than failing the whole report.
            continue
        s["name"] = canon

        conf = str(s.get("confidence", "")).strip().lower()
        s["confidence"] = conf if conf in _CONFIDENCE_LEVELS else "low"

        side = s.get("side")
        if isinstance(side, str):
            side_norm = side.strip().lower()
            s["side"] = side_norm if side_norm in {"left", "right", "bilateral"} else None

        if isinstance(s.get("detected"), str):
            s["detected"] = s["detected"].strip().lower() in {"true", "yes", "1"}

        s.setdefault("evidence_tool_calls", [])
        s.setdefault("key_metrics", {})
        s.setdefault("rationale", "")
        fixed_signs.append(s)
    data["motor_signs"] = fixed_signs

    # If the model called asymmetry detected but forgot the top-level flag,
    # set it for consistency.
    if any(
        s["name"] == "mouth_corner_asymmetry" and s.get("detected")
        for s in fixed_signs
    ):
        data["asymmetry_detected"] = True

    # Defaults for optional list/string fields
    data.setdefault("quality_issues", [])
    data.setdefault("flagged_findings", [])
    data.setdefault("recommended_followup", [])
    if not data.get("clinician_notes"):
        data["clinician_notes"] = (
            "Automated screening completed. Numbers reported above; clinician "
            "review recommended."
        )

    return data


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
    """Hybrid agent: deterministic Python rules + LLM-generated narrative.

    Clinical decisions (which signs are detected, severity, overall risk)
    are computed in Python from the upstream numbers using documented
    MDS-UPDRS thresholds. The LLM is asked ONLY to write the natural-
    language explanations: each sign's `rationale`, `flagged_findings`,
    `recommended_followup`, and `clinician_notes`. This works reliably
    even on very small models (≤0.5B) and keeps clinical decisions
    auditable.
    """
    if client is None:
        client = _default_client()
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    # ── Step 1: deterministic clinical decisions (no LLM) ─────────────
    state = _decide_clinical_state(payload)

    # ── Step 2: ask the LLM to write the narrative for those decisions ─
    decisions_summary = "\n".join(
        f"- {s['name']} | detected={s['detected']} | "
        f"severity={s['severity']} | side={s['side']} | "
        f"key_metrics={s['key_metrics']} | analysis: {s['summary']}"
        for s in state["signs"]
    )
    risk = state["overall_risk_level"]

    narrative_prompt = f"""You are a clinical screening assistant writing the narrative
section of a face-based Parkinson's screening report. Clinical decisions
have ALREADY been made by deterministic rules — do not revise them. Your
ONLY job is to write the English text that explains them to a clinician.

OVERALL RISK (already decided): {risk}
ASYMMETRY DETECTED (already decided): {state['asymmetry_detected']}

MOTOR SIGNS (decisions and supporting numbers):
{decisions_summary}

CLINICAL THRESHOLDS USED (so your prose is consistent):
- Hypomimia: composite_expressivity_score < 0.35 = mild, < 0.25 = moderate, < 0.15 = severe.
- Jaw tremor: dominant frequency in 4–6 Hz with peakedness > 0.5 and in-band fraction > 0.3 = strong; relaxed thresholds = weak.
- Mouth-corner asymmetry: ratio > 0.3 = mild, > 0.5 = strong; less_mobile_side identifies the candidate side.

Write a JSON object with EXACTLY these four fields, no others, no markdown:

{{
  "rationales": {{
    "hypomimia": "1–2 sentences explaining the hypomimia decision using the actual numbers",
    "jaw_tremor": "1–2 sentences explaining the jaw tremor decision using the actual numbers",
    "mouth_corner_asymmetry": "1–2 sentences explaining the asymmetry decision using the actual numbers"
  }},
  "flagged_findings": ["short English bullets — one per real finding worth highlighting"],
  "recommended_followup": ["short English bullets — concrete next steps for the clinician"],
  "clinician_notes": "2–3 English sentences: what was found overall, why it matters, and the recommendation."
}}

Cite the numbers. Be conservative. If overall risk is `low`, recommended_followup
should reflect that (e.g. routine monitoring), not unnecessary referrals."""

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": narrative_prompt}],
        options={"temperature": temperature, "num_predict": 600},
    )
    msg = _extract_message(response)
    raw = msg.get("content", "") or ""
    text = _strip_code_fence(raw)

    narrative = _parse_narrative_json(text, raw)

    # ── Step 3: assemble the final ScreeningReport ────────────────────
    motor_signs = []
    for s in state["signs"]:
        rationale = (narrative.get("rationales") or {}).get(s["name"], "") or s["summary"]
        motor_signs.append({
            "name": s["name"],
            "detected": s["detected"],
            "side": s["side"],
            "severity": s["severity"],
            "confidence": s["confidence"],
            "key_metrics": s["key_metrics"],
            "evidence_tool_calls": [],
            "rationale": rationale.strip(),
        })

    report_dict = {
        "patient_id": payload.patient_id,
        "session_id": payload.session_id,
        "overall_risk_level": state["overall_risk_level"],
        "asymmetry_detected": state["asymmetry_detected"],
        "motor_signs": motor_signs,
        "quality_issues": _quality_issues_from_payload(payload),
        "flagged_findings": _coerce_str_list(narrative.get("flagged_findings"))
            or _default_findings(state),
        "recommended_followup": _coerce_str_list(narrative.get("recommended_followup"))
            or _default_followup(state),
        "clinician_notes": (narrative.get("clinician_notes") or "").strip()
            or _default_notes(state),
    }
    report = ScreeningReport.model_validate(report_dict)
    return AgentResult(
        report=report,
        transcript=[{"iteration": 1, "message": msg}],
        tool_calls=[],
        iterations=1,
    )


# ---------------------------------------------------------------------------
# Helpers for hybrid mode
# ---------------------------------------------------------------------------

def _parse_narrative_json(text: str, raw: str) -> dict:
    """Best-effort parse of the narrative JSON. Falls back to empty dict
    so the deterministic skeleton still produces a valid report."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json  # noqa: PLC0415

            return json.loads(repair_json(text, return_objects=False))
        except Exception:
            print(
                "[run_screening_agent_simple] warning: narrative not parseable, "
                "using deterministic defaults.",
                file=sys.stderr,
            )
            print(f"--- raw narrative ---\n{raw}", file=sys.stderr)
            return {}


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _quality_issues_from_payload(payload: KnowledgePayload) -> list[str]:
    issues: list[str] = []
    cf = payload.clinical_features
    if cf.regional_motion is not None and not cf.regional_motion.valid:
        issues.append(f"regional_motion: {cf.regional_motion.reason or 'invalid'}")
    if cf.jaw_tremor is not None and not cf.jaw_tremor.valid:
        issues.append(f"jaw_tremor: {cf.jaw_tremor.reason or 'invalid'}")
    if cf.mouth_asymmetry is not None and not cf.mouth_asymmetry.valid:
        issues.append(f"mouth_asymmetry: {cf.mouth_asymmetry.reason or 'invalid'}")
    if payload.quality and payload.quality.face_coverage is not None:
        if payload.quality.face_coverage < 0.5:
            issues.append(
                f"low face coverage ({payload.quality.face_coverage:.0%}) — "
                f"results may be unreliable"
            )
    return issues


def _default_findings(state: dict) -> list[str]:
    detected = [s for s in state["signs"] if s.get("detected")]
    if not detected:
        return ["No motor signs convincingly detected from face-only metrics."]
    return [f"{s['name'].replace('_', ' ').capitalize()}: {s['summary']}"
            for s in detected]


def _default_followup(state: dict) -> list[str]:
    risk = state["overall_risk_level"]
    if risk == "elevated":
        return ["Refer for in-person MDS-UPDRS Part III evaluation by a neurologist."]
    if risk == "borderline":
        return ["Monitor and consider re-screening; refer if symptoms progress."]
    return ["No follow-up triggered by this screening run."]


def _default_notes(state: dict) -> str:
    risk = state["overall_risk_level"].upper()
    detected = [s["name"].replace("_", " ") for s in state["signs"] if s.get("detected")]
    if not detected:
        return (
            f"Overall risk: {risk}. No motor signs convincingly detected on "
            f"face-only screening; metrics within expected range."
        )
    listed = ", ".join(detected)
    return (
        f"Overall risk: {risk}. Detected motor signs on face-only screening: "
        f"{listed}. Clinician review recommended."
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
