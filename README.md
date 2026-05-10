# Parkinson Face Screening Agent — JSON in, report out

Hackathon project: a local agentic backend that takes the JSON metrics
emitted by a Luxonis OAK device after a face capture, and produces a
structured screening report for a clinician to review.

The clinician makes the final call; this tool just **explains the
numbers in clinical terms** so they can decide whether deeper assessment
is warranted.

The LLM runs **locally via Ollama** — no API keys, no per-call costs.

## Architecture

```
   ┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
   │  OAK device  │ →  │  JSON payload    │ →  │   Tools     │
   │  + signal    │    │  (PatientMetrics │    │  (read JSON │
   │  processing  │    │   Payload)       │    │   sections) │
   └──────────────┘    └──────────────────┘    └──────┬──────┘
                                                      │
                                                      ▼
                                          ┌──────────────────┐
                                          │  Ollama agent    │
                                          │  (tool-use loop) │
                                          └────────┬─────────┘
                                                   │ submit_report
                                                   ▼
                                          ┌──────────────────┐
                                          │ ScreeningReport  │
                                          │ (Pydantic JSON)  │
                                          └──────────────────┘
```

OAK runs the heavy lifting (FaceMesh inference, regional motion, tremor
spectrum, blink detection, asymmetry) and emits one JSON document per
session. This project validates that JSON, gives the agent tools to
inspect each section, and forces a structured report through the
terminal `submit_report` tool call.

## What lives where

```
gdgaihack/
├── parkinson_agent/
│   ├── input_schema.py     # PatientMetricsPayload (the OAK JSON contract)
│   ├── schemas.py          # ScreeningReport (the agent's output)
│   ├── tools.py            # Tools that read sections of the payload
│   ├── agent.py            # Ollama tool-use loop
│   └── run_demo.py         # CLI entrypoint
├── samples/
│   ├── demo_session.json     # Impaired patient (hypomimia + jaw tremor + L asymmetry)
│   └── healthy_session.json  # Control payload
├── clinician_ui.py         # Streamlit doctor-facing UI
├── tests/                  # Pytest suite (offline + gated integration)
└── requirements.txt
```

## Clinical scope

Face only — no hand or gait. Targets:

| MDS-UPDRS item | Field in the JSON payload |
|---|---|
| 3.2  Hypomimia          | `metrics.regional_motion` (composite score + per-region RoM) |
| 3.17 Rest tremor (jaw)  | `metrics.jaw_tremor` (3–7 Hz spectral analysis) |
| Supportive: blink rate  | `metrics.blink_rate` |
| Supportive: asymmetry   | `metrics.mouth_asymmetry` |

### Regional weights (clinical priors)

These are emitted in the JSON under `regional_weights` and surfaced to the
agent through `get_session_info`:

| Region | Weight |
|---|---|
| chin / jaw | **HIGH (1.0)** |
| lower lip | **HIGH (1.0)** |
| upper lip | MID (0.6) |
| mouth corners | MID (0.6) |
| cheeks | LOW-MID (0.4) |
| eyelids | LOW (0.2) |
| neck (proxy) | LOW (0.2) |

The composite expressivity score in `regional_motion` already applies
these weights on the OAK side; the agent reads it directly.

---

## Setup

### 1. Install Python deps

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Install and run Ollama

Download from https://ollama.com and start the server:

```bash
ollama serve &
ollama pull llama3.1     # or qwen2.5 — both support function calling
```

Override the model with `OLLAMA_MODEL=qwen2.5`. For a remote Ollama box,
set `OLLAMA_HOST=http://that-box:11434`.

---

## How to run

### Option A — Streamlit UI (recommended for the hackathon demo)

```bash
.venv/bin/streamlit run clinician_ui.py
```

In the sidebar, pick a sample payload (`demo_session.json` for impaired,
`healthy_session.json` for control), or upload your own JSON. Click
**Run screening** and the doctor-facing report renders, including the
audit trail of every tool call.

### Option B — CLI smoke test

```bash
.venv/bin/python -m parkinson_agent.run_demo
# or with a custom payload:
.venv/bin/python -m parkinson_agent.run_demo path/to/your_payload.json
```

Prints the structured report as JSON.

### Option C — Programmatic

```python
from parkinson_agent import PatientMetricsPayload, run_screening_agent

payload = PatientMetricsPayload.from_json_file("samples/demo_session.json")
result = run_screening_agent(payload, model="llama3.1")
print(result.report.model_dump_json(indent=2))
```

---

## Testing

### Offline unit + agent tests (zero LLM calls)

```bash
.venv/bin/pytest tests/ -v
```

Validates: payload schema, tool dispatch, schema-validation retry, iteration
cap, unknown-tool handling, text-only nudge.

`tests/test_integration.py` is auto-skipped if Ollama is not reachable.

### Live integration

```bash
ollama serve &
ollama pull llama3.1
.venv/bin/pytest tests/test_integration.py -v
```

Asserts that the agent flags the impaired sample as `borderline`/`elevated`
and the healthy sample as `low`/`borderline`.

---

## OAK JSON payload contract

The agent expects this shape (see `parkinson_agent/input_schema.py` for
the Pydantic source of truth, and `samples/demo_session.json` for a
complete example):

```jsonc
{
  "patient_id": "string",
  "session_id": "string",
  "captured_at": "2026-05-09T10:32:11Z",   // optional ISO-8601
  "duration_s": 28.0,
  "capture_fps": 30.0,                     // optional
  "device": "Luxonis OAK-D",               // optional
  "tasks": [
    { "name": "rest_seated",       "start_s": 0.0,  "end_s": 10.0 },
    { "name": "facial_expression", "start_s": 10.0, "end_s": 20.0 },
    { "name": "speech",            "start_s": 20.0, "end_s": 28.0 }
  ],
  "face_coverage": 0.96,                   // optional, 0..1
  "regional_weights": {
    "chin_jaw": 1.0, "lower_lip": 1.0, "upper_lip": 0.6,
    "mouth_corners": 0.6, "cheeks": 0.4, "eyelids": 0.2, "neck": 0.2
  },
  "metrics": {
    "regional_motion":  { /* per-region RoM + composite_expressivity_score */ },
    "jaw_tremor":       { /* dominant_frequency_hz, in_band_fraction, ... */ },
    "blink_rate":       { /* blink_rate_per_min, n_blinks, ... */ },
    "mouth_asymmetry":  { /* rom_left, rom_right, asymmetry_ratio, ... */ }
  }
}
```

Each section under `metrics` may set `valid: false` with a `reason` if the
OAK side could not compute it (e.g. face out of frame). The agent will
record those as quality issues in the report.

If your OAK side emits slightly different field names, edit
`input_schema.py` and the sample files — everything downstream keys off
those names.

---

## Demo script suggestion

For a 5-minute hackathon demo:

1. **30s — context.** "PD screening today is in-person, late, qualitative.
   Hypomimia and jaw tremor are detectable from the face alone."
2. **1m — show the device.** OAK on a tripod, patient seated. Run the
   short capture protocol; OAK emits the JSON.
3. **1m — show the agent.** Streamlit page renders the report: risk level,
   asymmetric findings, recommended follow-up. Open the audit trail to
   show every claim is backed by a numeric tool call.
4. **1m — pre-recorded contrast.** Switch between `demo_session.json` and
   `healthy_session.json`. Same UI, different reports.
5. **1m — closing + Q&A.** Mention what you did NOT do (validation against
   PPMI / labelled data, regulatory path) so judges know you understand
   the gap between hackathon and product.

---

## Design notes

- **Why JSON in, not raw landmarks:** the OAK side already runs FaceMesh
  + signal processing on-device. Decoupling at the JSON boundary means
  this agent has a stable, narrow contract and never has to know about
  depthai, FaceMesh, or filtering.
- **Why a terminal `submit_report` tool, not JSON-in-text:** structured
  output via tool-use is the most reliable way to force schema-valid
  output. The model literally cannot "submit" without producing an
  object that matches the `ScreeningReport` JSON Schema.
- **Why the regional weights are in the payload AND the prompt:**
  redundancy is intentional. The composite score is already biased
  toward HIGH-weight regions; the prompt ensures the LLM also reasons
  about regional priorities qualitatively.
- **Why temperature=0 by default:** for medical decision-support
  repeatability matters; same input, same report.
- **Why local Ollama:** no API keys, no per-call cost, runs on the same
  laptop as the OAK device. Patient data never leaves the box.

## Not in scope (yet)

- Streaming output to the clinician UI while the agent works.
- Multi-session longitudinal comparison (PD progression over visits).
- Confidence calibration against a labeled dataset — you need PPMI /
  mPower-style data for that, then run an eval harness over your
  prompts.
- The OAK-side capture pipeline. Wiring depthai + FaceMesh and emitting
  this JSON is a separate workstream owned outside this repo.
