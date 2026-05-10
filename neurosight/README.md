# NeuroVista

> Facial biomarker longitudinal screening for neurodegenerative disease — runs entirely on a Luxonis OAK 4 D camera.

**Submission for GDG AI Hack 2026 — Track B "See Beyond"**

---

## ⚠ Disclaimer (read first)

NeuroVista is a **research/screening prototype**. It is **NOT a medical device** and **does NOT provide a diagnosis**. All clinical interpretation requires a licensed neurologist. Reference values come from peer-reviewed literature but the implementation has not undergone clinical validation. Do not use it to inform medical decisions for any individual.

---

## Why this project

Hypomimia (reduced facial expressivity), reduced spontaneous blink rate, gaze instability and subtle facial asymmetry are documented **early motor signs of Parkinson's disease** — sometimes detectable years before clinical diagnosis. Today these signs are assessed subjectively during clinical visits using the MDS-UPDRS scale, leading to inter-rater variability and missed early progression.

NeuroVista turns the OAK 4 D into a **passive, repeatable, in-clinic screening tool** that produces objective, longitudinal measurements of four facial biomarkers, runs entirely on-device (so no patient video ever leaves the camera), and exports a one-page report the neurologist can attach to the patient record.

## Why OAK 4 D specifically

| Judging dim. | How NeuroVista hits it |
|---|---|
| **30 % On-device** | The full pipeline (YuNet face detection, MediaPipe Face Landmarker, biomarker extraction, JSON storage) runs as an OAK App standalone on the camera. Patient data never leaves the device — the *primary* reason for choosing on-device is **medical privacy** (HIPAA / GDPR alignment). |
| **25 % Depth** | Hypomimia is measured in **real millimeters**, not pixels: facial landmark pairs (mouth corners, brow ↔ eyelid) are back-projected to 3D using the stereo depth aligned to RGB. A "smile width drop from 52 mm to 38 mm over 9 months" has clinical meaning; pixel deltas don't. |
| **25 % Advanced CV** | Stack of 4 models / nodes: YuNet detector → MediaPipe FaceLandmarker (468 pts) → custom EAR/asymmetry/gaze processors → Stereo depth → DepthMerger fusion. |
| **20 % Practical utility** | Mercato Parkinson: ~10 M patients worldwide, $52 B annual cost. Early-stage diagnosis delay averages 2.9 years (Rusz 2021). |

## Biomarkers measured

| Biomarker | Method | Clinical signal | Reference |
|---|---|---|---|
| **Blink rate** | Eye Aspect Ratio peak-detection over a sliding 60 s window | Healthy adult: 15–22/min · Parkinson: < 12/min | Karson, *Neurology* 1983 |
| **Hypomimia (smile)** | Mouth-corner spread amplitude over 30 s, in mm via depth | Reduced amplitude = facial bradykinesia | Bologna et al., *Brain* 2013 |
| **Hypomimia (brow)** | Brow ↔ eyelid distance amplitude over 30 s, in mm via depth | Same — independent muscle group | — |
| **Facial asymmetry** | Landmark mirror residuals across the nose-chin axis, normalized by face width | Asymmetric onset is typical of Parkinson | Djaldetti et al., *Lancet Neurol* 2006 |
| **Gaze stability** | Eye-center offset variance over 30 s | Increased saccade noise = motor control drift | Pretegiani & Optican 2017 |

All biomarkers are extracted in a single `BiomarkerExtractor` HostNode (`utils/biomarker_node.py`) and emitted as a JSON-encoded `dai.Buffer` every frame.

## Architecture

```
┌─ ON-DEVICE (OAK 4 D, runs as oakapp) ──────────────────┐
│                                                          │
│  Camera CAM_A ─→ resize ─→ YuNet face det ──┐           │
│                                                ↓           │
│  Stereo CAM_B/C → StereoDepth (HIGH_DETAIL,   Script crop │
│   aligned to CAM_A) ───┐                       ↓           │
│                         │       MediaPipe Face Landmarker │
│                         │             ↓                   │
│                         │       GatherData (sync)         │
│                         │             ↓                   │
│                         └→ BiomarkerExtractor (HostNode) │
│                                       ↓                   │
│                          ┌────────────┼─────────────┐    │
│                          ↓            ↓             ↓    │
│                   OverlayNode   VisitWriter    Visualizer │
│                  (HUD on feed)   (JSON disk)    (WebRTC)  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─ CLINICIAN DASHBOARD (Flask + Plotly, runs on laptop) ─┐
│   data/visits/<patient>/visit_<date>.json reads        │
│   • 4 KPI cards (latest value + Δ% vs baseline)        │
│   • 6 trend charts with normal-range bands             │
│   • PDF export (single-page report, ReportLab)         │
└─────────────────────────────────────────────────────────┘
```

## Folder layout

```
neurosight/
├── main.py                 # OAK App entry point — DepthAI v3 pipeline
├── oakapp.toml             # OAK App manifest
├── requirements.txt
├── depthai_models/         # Zoo model declarations (YAML)
│   ├── yunet.RVC4.yaml
│   └── mediapipe_face_landmarker.RVC4.yaml
├── utils/
│   ├── arguments.py        # CLI argparse
│   ├── biomarkers.py       # Pure-function biomarker math (EAR, asymmetry, …)
│   ├── biomarker_node.py   # BiomarkerExtractor HostNode (orchestrator)
│   ├── depth_utils.py      # Pixel + depth → 3D mm helpers
│   ├── annotation_node.py  # OverlayNode (HUD on visualizer)
│   └── visit_writer.py     # VisitWriter (JSON persistence)
├── backend/
│   ├── app.py              # Flask dashboard
│   ├── pdf_report.py       # ReportLab single-page PDF export
│   └── templates/dashboard.html
├── scripts/
│   ├── ingest_videos.sh    # Batch-process curated videos as past visits
│   └── seed_demo_visits.py # Synthetic visits for offline dashboard dev
├── data/
│   ├── visits/             # JSON output (gitignored)
│   └── reports/            # PDF output (gitignored)
├── videos_input/           # Drop curated patient videos here
├── RUN_BOOK.md             # Saturday step-by-step playbook
└── README.md
```

## Quick start

### Live preview against the camera (no recording)
```bash
pip install -r requirements.txt --user
python3 main.py --no_save
# open the printed WebRTC URL in Chrome
```

### Record a visit
```bash
python3 main.py --patient_id alice --visit_id 2026-05-09 --duration 60
# writes data/visits/alice/visit_2026-05-09.json
```

### Process a curated video as a past visit
```bash
python3 main.py --video videos_input/alice__2025-09-01.mp4 \
                --patient_id alice --visit_id 2025-09-01 --duration 60
# or batch:
./scripts/ingest_videos.sh
```

### Deploy as OAK App (on-device)
```bash
oakctl connect <DEVICE_IP>
oakctl app run .              # dev iteration
# or for permanent install:
oakctl app build .
oakctl app install neurosight.oakapp
oakctl app enable <app-id>    # auto-start on boot
```

### Start the clinician dashboard
```bash
python3 backend/app.py --port 8080
# open http://localhost:8080
```

### Generate fake visits for dashboard preview (no camera needed)
```bash
python3 scripts/seed_demo_visits.py
```

## What's NOT in the MVP (intentionally)

- **Tremor FFT 4–6 Hz analysis** — clinically valuable but requires 30 s+ stable footage and the signal is noisy at our frame rate. Skipped to keep the MVP credible.
- **Whisper voice analysis** — pitch variability and speech rate are great Parkinson biomarkers but adding audio doubled the integration cost. Logged as a v2 stretch goal.
- **Patient enrollment / authentication** — out of scope for a 24 h hack; would be required before any clinical pilot.
- **MDS-UPDRS scoring mapping** — we report raw biomarker values; mapping to UPDRS-III items needs validation we cannot do in 24 h.

## License

For demo / educational use during GDG AI Hack 2026. Not licensed for clinical use.

## Validation footage

Four 60-second clips from publicly available YouTube videos are used for pipeline validation (not redistributed; sources fully credited in `videos_input/SOURCES.md`):

- JAMA Network clinical case demonstration (Symmetric Parkinsonism)
- Ian Frizell self-vlog on facial masking
- Parkinson's UK / Havas Lynx patient ambassador interview
- Parkinson's Foundation "Faces of Parkinson's" testimonial

Use is non-commercial, educational, transformative (we extract numerical biomarkers, not redistribute footage). For any clinical use, fresh recordings with informed consent under IRB approval would be required.

## Acknowledgements

- Luxonis for hardware + the [oak-examples](https://github.com/luxonis/oak-examples) repo we forked from
- The DepthAI v3 SDK
- MediaPipe FaceMesh and YuNet face detection
- Patient ambassadors and clinical educators whose public videos enabled this prototype's validation
