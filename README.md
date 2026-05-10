# Parkinson Face Screening Agent

Local agentic pipeline that turns a patient's face capture into a
structured screening report for the clinician — no cloud, no API keys,
no patient data leaving the laptop.

---

## The problem

Parkinson's Disease (PD) is currently screened in person, late, and
qualitatively. Two motor signs from MDS-UPDRS Part III are detectable
on the face alone:

- **Hypomimia (3.2)** — facial masking, reduced expressivity especially
  in the lower face.
- **Jaw rest tremor (3.17)** — rhythmic 4–6 Hz oscillation of the
  jaw at rest.

The premise of this project: a 60-second face capture is enough to flag
patients who deserve in-person evaluation. The bottleneck is interpreting
the numbers — that's the agent's job.

---

## Pipeline

```
   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │  OAK device  │ →  │  Landmark CSV    │ →  │  Upstream model │
   │  (FaceMesh)  │    │  (per-frame x/y/z)│   │  (models/)      │
   └──────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │ data.json
                                                        ▼
                                          ┌──────────────────────────┐
                                          │  LLM agent (Ollama)      │
                                          │  reads data.json,        │
                                          │  reasons MDS-UPDRS,       │
                                          │  emits ScreeningReport   │
                                          └────────┬─────────────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │  Clinician view  │
                                          │  (formatted text │
                                          │   + JSON audit)  │
                                          └──────────────────┘
```

The boundary between hardware/signal stage and the reasoning stage is
`data.json`. Its schema is documented in
`parkinson_agent/input_schema.py` (`KnowledgePayload`).

### Stage 1 — Capture (OAK + sparse landmarks)

A Luxonis OAK camera runs MediaPipe FaceMesh on-device and emits a CSV
with one row per frame. The dataset is **sparse** — ~108 of MediaPipe's
478 landmarks, focused on the **lower face**: lips, jaw, mouth corners,
nose bridge. No eyelid landmarks, no neck. Each row has `patient_id`,
`visit_id`, timestamp `t`, optional `age`/`sex`/`label`, and the
`x_{i}, y_{i}, z_{i}` coordinates.

### Stage 2 — Upstream model (CSV → data.json)

`models/generate_knowledge.py` ingests the CSV, anchors all landmarks to
the nose bridge (removing rigid head motion), and computes three
clinical features:

- **`regional_motion`** — for each anatomical region (chin/jaw,
  lower_lip, upper_lip, mouth_corners, cheeks): peak-to-peak range of
  motion, normalized by face size; plus a **composite expressivity
  score**, a weighted average across regions.
- **`jaw_tremor`** — Welch FFT on the chin centroid, 3–7 Hz band.
  Reports dominant frequency, in-band power fraction, and spectral
  peakedness.
- **`mouth_asymmetry`** — left vs right mouth-corner range of motion
  (landmarks 61 vs 291) → asymmetry ratio + less-mobile side.

A future extension wires the trained TCN classifier (`models/model.py`,
`train.py`) as `model_inference.pd_probability`. Not blocking for the
agent.

### Stage 3 — The agent (data.json → report)

The agent reads only `data.json`. It runs against a **local Ollama
model** (default `qwen2.5:1.5b`), interprets the upstream numbers
against MDS-UPDRS criteria, and produces a `ScreeningReport` that
contains: overall risk level, motor signs with side / severity /
confidence / rationale, flagged findings, recommended follow-up, and
clinician notes.

---

## Clinical priors (regional weights)

Not all face regions are equally informative for PD screening. The
upstream model applies these weights to the composite expressivity
score:

| Region | Weight | Rationale |
|---|---|---|
| chin / jaw | **HIGH (1.0)** | jaw tremor + lower-face bradykinesia are primary PD signals |
| lower lip | **HIGH (1.0)** | lower-face hypomimia is the most reliable face-only sign |
| upper lip | MID (0.6) | involved but later in the disease |
| mouth corners | MID (0.6) | asymmetry diagnostically useful |
| cheeks | LOW-MID (0.4) | non-specific |

**Not measured:**
- *Eyelids / blink rate* — the OAK sparse landmark set lacks eye
  contour points (159, 145, 386, 374). Reduced blink would be a useful
  supportive sign but we cannot compute it reliably.
- *Neck* — not in MediaPipe FaceMesh.

The agent's system prompt explicitly tells the LLM these signals are
unavailable so it does not fabricate them.

## Decision criteria the agent applies

- **Hypomimia detected** ↔ low `composite_expressivity_score` together
  with reduced range of motion in HIGH-weight regions (chin/jaw and
  lower lip).
- **Jaw tremor detected** ↔ `dominant_frequency_hz ∈ [4.0, 6.0]` AND
  `spectral_peakedness > 0.5` AND `in_band_fraction_of_total > 0.3`.
- **Mouth-corner asymmetry detected** ↔ `asymmetry_ratio > 0.3`, with
  `less_mobile_side` flagging the candidate side of unilateral onset.
- **Overall risk**: `low` (no convincing signs) → `borderline` (one
  weak sign or data-quality issues) → `elevated` (one strong sign or
  multiple convergent signs).

## Why a local LLM (Ollama)

- **Privacy by design:** patient data never leaves the laptop, no
  third-party API.
- **Zero per-call cost:** acceptable for a clinical setting that may
  process many sessions.
- **Reproducibility:** `temperature=0` + a small open model gives the
  same input → the same report.
- **Simple-mode loop:** for small CPU-only models (≤1.5B), the agent
  runs as a single LLM call with the upstream JSON inlined in the
  prompt and the report parsed back from the response. No tool-calling
  grammar overhead. A `json-repair` fallback handles minor JSON glitches
  small models occasionally produce.

---

## Demo

```bash
OLLAMA_MODEL=qwen2.5:1.5b .venv/bin/python -m parkinson_agent.run_demo --simple
```

The terminal prints a clinician-facing report:

- **Header** — patient ID, session ID, capture stats, overall risk
  badge (🟢 LOW / 🟡 BORDERLINE / 🔴 ELEVATED), asymmetry flag.
- **Motor signs assessed** — for each sign: detection, side, severity
  (0–4), confidence, supporting metrics with the actual numbers, and a
  short clinical rationale.
- **Flagged findings** and **Recommended follow-up** — bullet points
  the doctor can act on immediately.
- **Clinician notes** — 2–3 sentence summary readable in 15 seconds.
- **Analyzed data** — the upstream raw numbers (composite score,
  per-region RoM, tremor spectrum, asymmetry ratio) so the doctor can
  audit any claim in the rationale.
- **Data limitations** — explicit note about modalities not measured.

The structured JSON of the same report is saved alongside `data.json`
as `data/report_<session_id>.json` for downstream auditing.
