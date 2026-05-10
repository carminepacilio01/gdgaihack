"""Streamlit UI for the clinician demo.

Two modes:
- "Sample payload" — pick one of the JSON files under `samples/`.
- "Upload JSON" — load an OAK-emitted payload from disk.

Once a payload is loaded, click "Run screening" and we display:
- The structured ScreeningReport, rendered as the clinician would see it.
- The full audit trail of tool calls.

Runtime: a local Ollama server. Make sure `ollama serve` is running and the
chosen model is pulled (e.g. `ollama pull llama3.1`).

Run with:
    streamlit run clinician_ui.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from parkinson_agent.agent import run_screening_agent
from parkinson_agent.input_schema import PatientMetricsPayload


REPO_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = REPO_ROOT / "samples"


st.set_page_config(page_title="PD Face Screening", layout="wide")
st.title("Parkinson's Face Screening — Clinician View")
st.caption("Screening tool. Not a diagnostic device. Findings require clinical evaluation.")


with st.sidebar:
    st.header("Runtime")
    model = st.selectbox(
        "Ollama model",
        ["llama3.1", "qwen2.5", "llama3.1:70b", "qwen2.5:14b"],
        index=0,
        help="Must be pulled locally (`ollama pull <name>`) and support function-calling.",
    )
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    st.caption(f"Ollama host: `{ollama_host}`")

    st.header("Source")
    samples = sorted(SAMPLES_DIR.glob("*.json")) if SAMPLES_DIR.exists() else []
    sample_names = [p.name for p in samples]
    source = st.radio("Payload source", ["Sample", "Upload JSON"])

    payload: PatientMetricsPayload | None = None
    raw_dict: dict | None = None

    if source == "Sample":
        if not sample_names:
            st.warning("No sample JSON files found under `samples/`.")
        else:
            chosen = st.selectbox("Sample payload", sample_names)
            with open(SAMPLES_DIR / chosen, "r", encoding="utf-8") as f:
                raw_dict = json.load(f)
    else:
        uploaded = st.file_uploader("OAK payload", type=["json"])
        if uploaded is not None:
            raw_dict = json.loads(uploaded.read().decode("utf-8"))


if raw_dict is None:
    st.info("Pick a sample payload or upload an OAK JSON file from the sidebar.")
    st.stop()

try:
    payload = PatientMetricsPayload.model_validate(raw_dict)
except Exception as exc:
    st.error(f"Payload does not match the expected schema:\n\n```\n{exc}\n```")
    st.stop()


# Session summary panel
st.subheader("Session")
col1, col2, col3 = st.columns(3)
col1.metric("Patient", payload.patient_id)
col2.metric("Duration", f"{payload.duration_s:.1f}s")
col3.metric(
    "Face coverage",
    f"{payload.face_coverage*100:.0f}%" if payload.face_coverage is not None else "N/A",
)

with st.expander("Tasks"):
    st.write([t.model_dump() for t in payload.tasks])

with st.expander("Regional weights (clinical priors)"):
    st.json(payload.regional_weights)

with st.expander("Raw OAK metrics"):
    st.json(payload.metrics.model_dump(exclude_none=True))


# Run agent
run = st.button("Run screening", type="primary")

if run:
    with st.spinner(f"Agent is analyzing the payload with `{model}`..."):
        try:
            result = run_screening_agent(payload, model=model)
        except Exception as exc:
            st.error(f"Agent failed: {exc}")
            st.stop()

    report = result.report

    risk_color = {"low": "🟢", "borderline": "🟡", "elevated": "🔴"}[
        report.overall_risk_level.value
    ]
    st.markdown(
        f"### {risk_color} Overall risk: **{report.overall_risk_level.value.upper()}**"
    )
    if report.asymmetry_detected:
        st.warning("Significant left-right asymmetry detected.")

    st.subheader("Motor signs")
    for sign in report.motor_signs:
        title = (
            f"{'✅' if sign.detected else '➖'} {sign.name.value} "
            f"({sign.confidence.value} confidence"
            + (f", side: {sign.side}" if sign.side else "")
            + (f", severity: {sign.severity}" if sign.severity is not None else "")
            + ")"
        )
        with st.expander(title):
            st.write(sign.rationale)
            st.write("**Evidence (tool calls):**", ", ".join(sign.evidence_tool_calls))
            st.json(sign.key_metrics)

    st.subheader("Flagged findings")
    for f in report.flagged_findings:
        st.markdown(f"- {f}")

    st.subheader("Recommended follow-up")
    for r in report.recommended_followup:
        st.markdown(f"- {r}")

    st.subheader("Clinician notes")
    st.write(report.clinician_notes)

    if report.quality_issues:
        st.subheader("Data quality caveats")
        for q in report.quality_issues:
            st.markdown(f"- ⚠️ {q}")

    st.caption(report.disclaimer)

    with st.expander(
        f"Audit trail — {len(result.tool_calls)} tool calls in {result.iterations} turns"
    ):
        for call in result.tool_calls:
            st.markdown(f"**`{call['name']}`**")
            st.json({"input": call["input"], "result": call["result"]})

    st.download_button(
        "Download report JSON",
        data=report.model_dump_json(indent=2),
        file_name=f"{report.session_id}_report.json",
        mime="application/json",
    )
