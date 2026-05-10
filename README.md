# Parkinson Face Screening Agent — `data.json` → clinician report

Hackathon project. Two clean stages:

1. **Upstream model** (`models/`) — reads OAK landmark CSVs, computes
   clinical features (regional motion, jaw tremor, mouth-corner asymmetry),
   optionally runs a TCN classifier, writes `data.json`.
2. **Agent** (`parkinson_agent/`) — reads only `data.json`, asks a local
   Ollama LLM to interpret it in MDS-UPDRS terms, and produces a
   structured screening report for the clinician.

The clinician makes the final call; this tool just **explains the
upstream model's knowledge** so they can decide whether deeper
assessment is warranted.

The LLM runs **locally via Ollama** — no API keys, no per-call costs,
patient data never leaves the laptop.

## Architecture

```
   ┌──────────────┐    ┌────────────────────┐    ┌─────────────────┐
   │  OAK device  │ →  │  Landmark CSV      │ →  │  Upstream model │
   │  (FaceMesh)  │    │  (per-frame x/y/z) │    │  (models/)      │
   └──────────────┘    └────────────────────┘    └────────┬────────┘
                                                          │ data.json
                                                          ▼
                                          ┌──────────────────────────┐
                                          │  Tools (read JSON)       │
                                          │  - get_session_info      │
                                          │  - get_regional_motion   │
                                          │  - get_jaw_tremor        │
                                          │  - get_mouth_asymmetry   │
                                          │  - get_model_inference   │
                                          └────────┬─────────────────┘
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

The boundary between the two stages is `data.json`. Its schema is the
single source of truth for the agent's input contract and lives in
`parkinson_agent/input_schema.py` (`KnowledgePayload`).

## Directory layout

```
gdgaihack/
├── parkinson_agent/            # AGENT — reads data.json only
│   ├── __init__.py
│   ├── input_schema.py         # KnowledgePayload (data.json contract)
│   ├── schemas.py              # ScreeningReport (output)
│   ├── tools.py                # JSON section readers
│   ├── agent.py                # Ollama tool-use loop
│   └── run_demo.py             # CLI entry
├── models/                     # UPSTREAM PIPELINE — CSV → data.json
│   ├── generate_knowledge.py   # ← entry point (CSV → data.json)
│   ├── dataset_preprocessing.py
│   ├── model.py                # SmallPDTCN classifier
│   ├── train.py                # Training script (produces .pth weights)
│   └── preprocessed/
├── data/                       # ALL DATA
│   ├── data.json               # ← agent input (sample, regenerable)
│   ├── raw/                    # raw OAK CSVs
│   │   └── embed_2026-05-10_013137.csv
│   └── master_dataset.csv      # training data (gitignored, 37MB)
├── tests/
├── clinician_ui.py             # Streamlit doctor-facing UI
├── conftest.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Clinical scope

Face only — no hand or gait. The agent reasons about:

| MDS-UPDRS item | Source in `data.json` | Tool |
|---|---|---|
| 3.2  Hypomimia          | `clinical_features.regional_motion` | `get_regional_motion` |
| 3.17 Rest tremor (jaw)  | `clinical_features.jaw_tremor`      | `get_jaw_tremor` |
| Supportive: asymmetry   | `clinical_features.mouth_asymmetry` | `get_mouth_asymmetry` |
| ML signal               | `model_inference` (optional)        | `get_model_inference` |

**Not measured** because the OAK sparse landmark set lacks the points:
- blink rate / eyelid hypokinesia (no eyelid landmarks)
- neck (not in MediaPipe FaceMesh)

The system prompt explicitly tells the agent these signals are
unavailable so it doesn't fabricate them.

### Regional weights (clinical priors)

These are baked into the upstream pipeline (`generate_knowledge.REGION_WEIGHTS`)
and surfaced to the agent through the `data.json`:

| Region | Weight |
|---|---|
| chin / jaw | **HIGH (1.0)** |
| lower lip | **HIGH (1.0)** |
| upper lip | MID (0.6) |
| mouth corners | MID (0.6) |
| cheeks | LOW-MID (0.4) |

The composite expressivity score in `regional_motion` is a weighted
average of normalized per-region range-of-motion using these weights.

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
ollama pull llama3.2:3b      # recommended default — fast on CPU-only laptops
```

Override the model with `OLLAMA_MODEL=qwen2.5:3b`. For a remote Ollama
box, set `OLLAMA_HOST=http://that-box:11434`.

### 3. Generate `data.json` (one-time, regenerable)

```bash
.venv/bin/python -m models.generate_knowledge
# → writes data/data.json from data/raw/embed_2026-05-10_013137.csv
```

You can pick a different patient/visit:

```bash
.venv/bin/python -m models.generate_knowledge \
    --csv data/raw/your_capture.csv \
    --patient demo_patient1 --visit 2026-05-10_013137 \
    --out data/data.json
```

If a trained TCN checkpoint exists at `models/pd_tcn_weights.pth`,
`generate_knowledge.py` will populate the `model_inference` section
(currently a stub pending wire-up — see TODO in that file).

---

## How to run the agent

### Option A — Streamlit UI (recommended for the demo)

```bash
.venv/bin/streamlit run clinician_ui.py
```

Sidebar lets you point at any `data.json` (default: `data/data.json`)
or upload one. Click **Run screening**.

### Option B — CLI

```bash
.venv/bin/python -m parkinson_agent.run_demo                 # data/data.json
.venv/bin/python -m parkinson_agent.run_demo path/to/x.json  # custom payload
```

### Option C — Programmatic

```python
from parkinson_agent import KnowledgePayload, run_screening_agent

payload = KnowledgePayload.from_json_file("data/data.json")
result = run_screening_agent(payload, model="llama3.2:3b")
print(result.report.model_dump_json(indent=2))
```

---

## Testing

### Offline tests (zero LLM calls)

```bash
.venv/bin/pytest tests/ -v
```

Covers:
- `KnowledgePayload` schema validation (sample + edge cases).
- Agent tool dispatch, schema-validation retry, iteration cap,
  unknown-tool handling, text-only nudge.
- The new `get_model_inference` tool returning a graceful
  `section_missing` marker when the upstream model didn't run.

`tests/test_integration.py` is auto-skipped if Ollama is not reachable.

### Live integration

```bash
ollama serve &
ollama pull llama3.2:3b
.venv/bin/pytest tests/test_integration.py -v
```

---

## `data.json` contract

Top level (see `parkinson_agent/input_schema.py` for the Pydantic source):

```jsonc
{
  "patient_id": "string",
  "session_id": "string",
  "captured_at": "2026-05-10T01:31:37Z",   // optional ISO-8601
  "duration_s": 60.3,
  "n_frames": 1054,
  "fps": 17.4,
  "metadata":         { "age": null, "sex": null, "ground_truth_label": null },
  "regional_weights": { "chin_jaw": 1.0, "lower_lip": 1.0, "upper_lip": 0.6,
                        "mouth_corners": 0.6, "cheeks": 0.4 },
  "clinical_features": {
    "regional_motion":  { /* per-region RoM + composite_expressivity_score */ },
    "jaw_tremor":       { /* dominant_frequency_hz, in_band_fraction, ... */ },
    "mouth_asymmetry":  { /* rom_left, rom_right, asymmetry_ratio, ...    */ }
  },
  "model_inference":  { "model_name": "SmallPDTCN", "pd_probability": 0.71, ... } | null,
  "quality":          { "face_coverage": 0.96, "missing_modalities": [...] }
}
```

Each section under `clinical_features` and `model_inference` may set
`valid: false` with a `reason` if the upstream stage couldn't compute
it; the agent records those as quality issues in the final report.

If the upstream pipeline emits slightly different field names, edit
`input_schema.py` and `models/generate_knowledge.py` together — they
must agree.

---

## Demo script suggestion

For a 5-minute hackathon demo:

1. **30s — context.** "PD screening today is in-person, late, qualitative.
   Hypomimia and jaw tremor are detectable from the face alone."
2. **1m — show the device.** OAK on a tripod. `models/generate_knowledge.py`
   turns the capture CSV into `data.json`.
3. **1m — show the agent.** Streamlit UI on `data.json`. The report
   renders: risk level, motor signs, recommended follow-up. Audit
   trail: every claim is backed by a numeric tool call.
4. **1m — contrast.** Regenerate `data.json` from a different patient
   or modify a value to see how the report changes.
5. **1m — closing + Q&A.** What's NOT done: PPMI validation, regulatory
   path. Wiring the trained TCN as the `model_inference` source is the
   next step.

---

## Design notes

- **Why a hard JSON boundary:** the agent never has to know about
  CSVs, FaceMesh, or signal processing. The upstream model can change
  underneath as long as it keeps emitting the `KnowledgePayload`
  contract.
- **Why a terminal `submit_report` tool, not JSON-in-text:** structured
  output via tool-use is the most reliable way to force schema-valid
  output. The LLM literally cannot "submit" without producing an
  object that matches the `ScreeningReport` JSON Schema.
- **Why temperature=0 by default:** for medical decision-support
  repeatability matters; same input, same report.
- **Why local Ollama on `llama3.2:3b`:** no API keys, no per-call cost,
  fast enough on CPU-only laptops, supports function calling. Patient
  data never leaves the box.

## Not in scope (yet)

- Streaming output to the clinician UI while the agent works.
- Multi-session longitudinal comparison.
- Calibrated confidence vs a labeled dataset (PPMI / mPower).
- Wiring the trained TCN inside `generate_knowledge._maybe_run_tcn`
  (currently a stub returning a placeholder when weights exist).
- The OAK-side capture pipeline. Wiring depthai + FaceMesh and
  emitting the CSV is a separate workstream.
