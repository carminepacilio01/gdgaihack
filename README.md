# Parkinson Face Screening Agent — OAK + Ollama

Agentic backend for a hackathon project that uses the Luxonis OAK depth
camera to capture **facial motor data** from a patient and produces a
structured screening report for a clinician.

The LLM runs **locally via Ollama** — no API keys, no per-call costs.

## Architecture

```
   ┌──────────────┐    ┌────────────────────┐    ┌─────────────┐
   │  OAK device  │ →  │ CaptureSession     │ →  │   Tools     │
   │  (depthai)   │    │ (face landmarks +  │    │ (face       │
   │  FaceMesh    │    │  task windows)     │    │  features)  │
   └──────────────┘    └────────────────────┘    └──────┬──────┘
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

The agent never sees raw landmarks. It calls tools that return numeric
clinical features (regional motion, jaw tremor, blink rate, mouth-corner
asymmetry), reasons against MDS-UPDRS Part III items relevant to the face,
and finalizes by calling `submit_report` whose schema is the
`ScreeningReport` Pydantic model. That tool call is our forced structured
output.

## Clinical scope

This track screens **face only** — no hand or gait data. Targets:

| MDS-UPDRS item | Face-only mapping |
|---|---|
| 3.2  Hypomimia | regional motion (chin/jaw, lips), composite expressivity score |
| 3.17 Rest tremor (jaw) | spectral analysis of chin motion at 3–7 Hz |
| Supportive | blink rate, mouth-corner asymmetry, eyelid hypokinesia |

### Regional weights (clinical priors)

The metrics layer applies these per-region weights when computing the
composite expressivity score:

| Region | Weight |
|---|---|
| chin / jaw | **HIGH (1.0)** |
| lower lip | **HIGH (1.0)** |
| upper lip | MID (0.6) |
| mouth corners | MID (0.6) |
| cheeks | LOW-MID (0.4) |
| eyelids | LOW (0.2) |
| neck (proxy) | LOW (0.2) |

These weights are exposed to the agent (`get_session_info` returns them,
and each metric tool tags its outputs) so the LLM can reason about
which signals are primary vs supportive.

## Layout

```
gdgaihack/
├── parkinson_agent/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic: ScreeningReport, MotorSign, SignName
│   ├── oak_adapter.py          # CaptureSession, FaceTimeSeries, TaskWindow
│   ├── signal_processing.py    # Regional motion, jaw tremor, blink rate, asymmetry
│   ├── tools.py                # Tool registry + Ollama tool schemas
│   ├── agent.py                # Main loop: client.chat + tool dispatch
│   └── run_demo.py             # Synthetic face session + entrypoint
├── capture_oak.py              # depthai capture skeleton (face-only)
├── clinician_ui.py             # Streamlit UI for the doctor-facing demo
├── tests/                      # Pytest suite (offline + gated integration)
└── requirements.txt
```

---

## Setup

### 1. Install Python deps

```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama

Download from https://ollama.com and start the server:

```bash
ollama serve &
ollama pull llama3.1     # or qwen2.5 — both support function calling
```

The agent uses `llama3.1` by default. Override with `OLLAMA_MODEL=qwen2.5`.
If you run Ollama on a remote box, set `OLLAMA_HOST=http://that-box:11434`.

---

## Testing

Three layers, in order of cost:

### 1. Offline unit + agent tests (zero LLM calls)

The agent loop is tested with a mocked Ollama client that scripts tool-use
responses. Validates: dispatch, schema-validation retry, iteration cap,
unknown-tool error handling, text-only nudge.

```bash
pytest tests/ -v
```

`tests/test_integration.py` is automatically skipped if Ollama is not
reachable, so this is safe in CI.

Use this loop while iterating on prompts and tool descriptions — fast,
deterministic, free.

### 2. Live integration test (zero $ cost, local CPU/GPU)

With Ollama running and a model pulled:

```bash
ollama pull llama3.1
pytest tests/test_integration.py -v
```

The integration test uses the synthetic hypomimia + jaw-tremor session and
asserts that any clinically reasonable agent flags at least one face sign
and chooses a `borderline` or `elevated` overall risk. Run this every time
you change `SYSTEM_PROMPT` or tool descriptions.

For the final pre-demo sanity check, try a larger model:

```bash
OLLAMA_MODEL=llama3.1:70b pytest tests/test_integration.py -v
```

### 3. Hardware-in-the-loop test

Once `capture_oak.record_session()` is wired to your depthai pipeline:

```bash
python -c "
from capture_oak import record_session, save_session
session = record_session(patient_id='TEST-001', pipeline_factory=...)
save_session(session, '/tmp/test_session.npz')
"
```

Then drop the `.npz` into the Streamlit UI and run the agent. Save a few
`.npz` recordings of different patients/scenarios as regression fixtures.

---

## Deployment

### Topology A — Single-laptop (recommended for hackathon)

Everything in one Python process on the laptop attached to the OAK:

```
[OAK device] ── USB ──> [laptop running Ollama + Streamlit]
                          │
                          ├── capture_oak.record_session()  → CaptureSession
                          ├── parkinson_agent.run_screening_agent()
                          └── clinician_ui.py  (browser tab on same laptop)
```

Run:

```bash
ollama serve &
streamlit run clinician_ui.py
```

The clinician UI lets you pick the synthetic session or load an `.npz`
recording, run the agent, and see the structured report with the tool-call
audit trail. Use synthetic for the offline part of the demo, switch to
recorded captures during the live segment.

For the live capture step you can either:
- Add a button to `clinician_ui.py` that calls `capture_oak.record_session()`
  inline (block on the protocol, then run the agent), or
- Run a separate capture CLI that produces an `.npz` for the UI to load.

The CLI variant is more robust for a demo — captures happen in a
controlled window and the UI is just the "doctor-facing" surface.

### Configuration knobs

| Env var | Default | Notes |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_MODEL` | `llama3.1` | Override per environment. Must support function calling (llama3.1, qwen2.5, mistral). |

Inside `agent.py`, `temperature=0.0` by default — repeatability matters
for medical decision support.

---

## Demo script suggestion

For a 5-minute hackathon demo:

1. **30s — context.** "PD screening today is in-person, late, qualitative.
   Hypomimia and jaw tremor are diagnosable from the face alone with the
   right instrument."
2. **1m — show the device.** OAK on a tripod, patient seated. Run the
   short capture protocol (rest, expression, speech).
3. **1m — show the agent.** Streamlit page renders the report: risk level,
   asymmetric findings, recommended follow-up. Open the audit trail to show
   that every claim is backed by a numerical tool call.
4. **1m — pre-recorded contrast.** Load an `.npz` of an "elevated risk"
   recording vs a "low risk" one. Same UI, different reports.
5. **1m — closing + Q&A.** Mention what you did NOT do (validation against
   PPMI / labelled data, regulatory path) so judges know you understand
   the gap between hackathon and product.

---

## Extending

### Wiring real OAK data

Implement `record_session()` in `capture_oak.py`. Starting points:

- Luxonis examples: https://github.com/luxonis/depthai-experiments
- FaceMesh on OAK: https://github.com/geaxgx/depthai_blazepose

You only need to fill in:
1. The `pipeline_factory()` returning a `dai.Pipeline` with a FaceMesh node.
2. A small `parse_face` that converts each output queue message into
   `(timestamp, (478, 3) landmarks, blink_event_or_None)`.

Everything else (task scripting, buffer accumulation, packaging into a
`CaptureSession`) is already there in `record_session()`.

### Adding a new face metric

1. Add the extractor in `signal_processing.py` (use `REGION_INDICES` /
   `REGION_WEIGHTS` for consistency with the clinical priors).
2. Bind it as a tool in `tools.py` (`make_tools` + `ANALYSIS_TOOL_SCHEMAS`).
   Include the MDS-UPDRS item number and threshold guidance in the
   description — the LLM reasons from those.
3. Add the `SignName` enum value in `schemas.py` if it's a new clinical
   sign worth surfacing in the report.

No prompt changes needed; the agent picks it up from the tool schemas.

### Auditability

`AgentResult.transcript` and `AgentResult.tool_calls` give you the full
audit trail: which tools were called, with what arguments, what numbers
came back, and what the model concluded. Persist this alongside the report
— relevant for a future medical-device dossier.

---

## Design notes

- **Why a terminal `submit_report` tool, not JSON-in-text:** structured
  output via tool-use is the most reliable way to force schema-valid
  output. The model literally cannot "submit" without producing an object
  that matches the `ScreeningReport` JSON Schema.
- **Why thresholds live in tool descriptions, not in code:** clinical
  reasoning is what the LLM is for. The signal processing layer returns
  numbers; the model decides how to weight them. Tuning becomes prompt
  engineering, not code edits.
- **Why the regional weights are in metrics AND prompt:** redundancy is
  intentional. Even if the LLM ignores the prompt-level priors, the
  composite score is already biased toward the high-weight regions.
- **Why temperature=0 by default:** for medical decision-support
  repeatability matters; same input, same report.
- **Why local Ollama:** no API keys, no per-call cost, runs on the same
  laptop as the OAK device. Patient face data never leaves the box.

## Not in scope (yet)

- Streaming output to the clinician UI while the agent works.
- Multi-session longitudinal comparison (PD progression over visits).
- Confidence calibration against a labeled dataset — you need PPMI /
  mPower-style data for that, then run an eval harness over your prompts.
- Hand and gait modalities — explicitly removed from this track to keep
  scope tight. They live in the original Anthropic-based scaffold history.
