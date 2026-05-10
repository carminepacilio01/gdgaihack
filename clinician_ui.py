"""Streamlit UI for the clinician demo.

Loads `data.json` (produced by the upstream model under `models/`),
runs the screening agent, and renders the structured report with the
audit trail of every tool call.

Runtime: a local Ollama server. Make sure `ollama serve` is running
and the chosen model is pulled (e.g. `ollama pull llama3.2:3b`).

Run with:
    streamlit run clinician_ui.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from parkinson_agent.agent import run_screening_agent
from parkinson_agent.input_schema import KnowledgePayload


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = REPO_ROOT / "data" / "data.json"


st.set_page_config(page_title="PD Face Screening", layout="wide")
st.title("Parkinson's Face Screening — Clinician View")
st.caption("Screening tool. Not a diagnostic device. Findings require clinical evaluation.")


with st.sidebar:
    st.header("Runtime")
    model = st.selectbox(
        "Ollama model",
        ["llama3.2:3b", "qwen2.5:3b", "llama3.1", "qwen2.5", "mistral"],
        index=0,
        help="Must be pulled locally (`ollama pull <name>`) and support function-calling. "
             "`llama3.2:3b` is the recommended default for CPU-only laptops.",
    )
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    st.caption(f"Ollama host: `{ollama_host}`")

    st.header("Input (data.json)")
    path_str = st.text_input("Payload path", value=str(DEFAULT_DATA))
    uploaded = st.file_uploader("…or upload a JSON", type=["json"])

    raw_dict: dict | None = None
    if uploaded is not None:
        raw_dict = json.loads(uploaded.read().decode("utf-8"))
    else:
        path = Path(path_str)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw_dict = json.load(f)
        else:
            st.warning(f"Payload not found: `{path}`. "
                       f"Run `python -m models.generate_knowledge` to create it.")


if raw_dict is None:
    st.info("Provide a `data.json` payload via path or upload.")
    st.stop()

try:
    payload = KnowledgePayload.model_validate(raw_dict)
except Exception as exc:
    st.error(f"Payload does not match the expected schema:\n\n```\n{exc}\n```")
    st.stop()


# Session summary panel
st.subheader("Session")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Patient", payload.patient_id)
col2.metric("Frames", payload.n_frames or "?")
col3.metric("Duration", f"{payload.duration_s:.1f}s")
col4.metric("FPS", f"{payload.fps:.1f}" if payload.fps else "?")

bits = []
if payload.metadata.age is not None: bits.append(f"age={payload.metadata.age:g}")
if payload.metadata.sex is not None: bits.append(f"sex={payload.metadata.sex}")
if payload.metadata.ground_truth_label is not None:
    bits.append(f"ground-truth label={payload.metadata.ground_truth_label}")
if bits:
    st.caption("  ·  ".join(bits))

with st.expander("Regional weights (clinical priors)"):
    st.json(payload.regional_weights)

with st.expander("Upstream knowledge (data.json sections)"):
    st.json(payload.model_dump(exclude_none=True))


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
