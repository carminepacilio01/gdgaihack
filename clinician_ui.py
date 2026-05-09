"""Streamlit UI for the clinician demo.

Two modes:
- "Synthetic demo" — for the offline demo (no OAK plugged in).
- "Upload .npz" — for a recorded session from `capture_oak.save_session`.

Once a session is loaded, click "Run screening" and we display:
- The structured ScreeningReport, rendered as the clinician would see it.
- The full audit trail of tool calls.

Runtime: a local Ollama server. Make sure `ollama serve` is running and the
chosen model is pulled (e.g. `ollama pull llama3.1`).

Run with:
    streamlit run clinician_ui.py
"""
from __future__ import annotations

import os

import streamlit as st

from capture_oak import load_session
from parkinson_agent.agent import run_screening_agent
from parkinson_agent.run_demo import synthetic_session
from parkinson_agent.signal_processing import REGION_WEIGHTS


st.set_page_config(page_title="PD Face Screening", layout="wide")
st.title("Parkinson's Face Screening — Clinician View")
st.caption("Screening tool. Not a diagnostic device. Findings require clinical evaluation.")


with st.sidebar:
    st.header("Runtime")
    model = st.selectbox(
        "Ollama model",
        ["llama3.1", "qwen2.5", "llama3.1:70b", "qwen2.5:14b"],
        index=0,
        help="The model must be pulled locally (`ollama pull <name>`) and "
             "must support function-calling.",
    )
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    st.caption(f"Ollama host: `{ollama_host}`")

    st.header("Source")
    source = st.radio("Session source", ["Synthetic demo", "Upload .npz"])
    uploaded_session = None
    if source == "Upload .npz":
        uploaded = st.file_uploader("Recorded session", type=["npz"])
        if uploaded is not None:
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            uploaded_session = load_session(tmp_path)

    st.header("Regional weights")
    st.caption("Clinical priors used by the metrics layer.")
    st.json(REGION_WEIGHTS)


# Build session
if source == "Synthetic demo":
    session = synthetic_session()
else:
    session = uploaded_session

if session is None:
    st.info("Upload a recorded .npz session or switch to the synthetic demo.")
    st.stop()

# Session summary panel
st.subheader("Session")
col1, col2, col3 = st.columns(3)
col1.metric("Patient", session.patient_id)
col2.metric("Duration", f"{session.duration_s:.1f}s")
col3.metric(
    "Face coverage",
    f"{session.face.coverage()*100:.0f}%" if session.face is not None else "N/A",
)
with st.expander("Available tasks"):
    st.write([
        {"name": t.name, "start_s": t.start, "end_s": t.end}
        for t in session.tasks
    ])


# Run agent
run = st.button("Run screening", type="primary")

if run:
    with st.spinner(f"Agent is analyzing the session with `{model}`..."):
        try:
            result = run_screening_agent(session, model=model)
        except Exception as exc:
            st.error(f"Agent failed: {exc}")
            st.stop()

    report = result.report

    # Headline strip
    risk_color = {"low": "🟢", "borderline": "🟡", "elevated": "🔴"}[
        report.overall_risk_level.value
    ]
    st.markdown(
        f"### {risk_color} Overall risk: **{report.overall_risk_level.value.upper()}**"
    )
    if report.asymmetry_detected:
        st.warning("Significant left-right asymmetry detected.")

    # Motor signs
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

    # Findings & recommendations
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

    # Audit panel
    with st.expander(
        f"Audit trail — {len(result.tool_calls)} tool calls in {result.iterations} turns"
    ):
        for call in result.tool_calls:
            st.markdown(f"**`{call['name']}`**")
            st.json({"input": call["input"], "result": call["result"]})

    # Raw export
    st.download_button(
        "Download report JSON",
        data=report.model_dump_json(indent=2),
        file_name=f"{report.session_id}_report.json",
        mime="application/json",
    )
