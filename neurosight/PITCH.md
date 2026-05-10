# Pitch — 5 slide outline (≈ 4 min talk + 1 min demo)

Speaker notes embedded. Times are cumulative.

---

## Slide 1 — Hook · Problem  (0:00 → 0:45)

**Title:** Parkinson's hides on people's faces — and we miss it for years.

**Big number:** Average diagnostic delay = **2.9 years** from first motor symptom to clinical diagnosis (Rusz et al., *Mov Disord* 2021).

**Why:** Early Parkinson signs are *subtle facial changes* — reduced blink rate, smaller smile, slight asymmetry. Today these are scored subjectively during a 15-minute neurology visit using the MDS-UPDRS scale. Inter-rater variability is high. Slow progression goes unnoticed between visits.

**Speaker line:** *"Imagine if every visit produced an objective fingerprint of your face — and the neurologist could see exactly how much it changed since last time."*

---

## Slide 2 — Our solution  (0:45 → 1:30)

**Title:** NeuroVista — multi-modal Parkinson biomarker tracking on a single OAK 4 D.

A passive, in-clinic screening tool. Patient sits in front of an OAK 4 D for 60 seconds and reads a short standard sentence. We extract **six objective biomarker families**, all grounded in published literature:

| Biomarker | Reference | Pathological cutoff |
|---|---|---|
| **Blink rate** | Karson, *Neurology* 1983 | < 12 / min |
| **Hypomimia** (smile + brow amplitude, **mm**) | Bologna et al., *Brain* 2013 | smile < 35 mm, brow < 3 mm |
| **Facial asymmetry** (mirror residual) | Djaldetti et al., *Lancet Neurol* 2006 | > 0.05 |
| **Jaw / lip tremor** (FFT power 4–6 Hz) | Bologna 2013 + Bain 2003 | dominant freq locked in 4–6 Hz band |
| **Voice acoustics** — jitter, shimmer, HNR | Rusz et al. 2011 + Goberman 2002 | jitter > 1.04 %, shimmer > 3.81 %, HNR < 20 dB |
| **Voice prosody** — speech rate, pause ratio | Skodda et al. 2011 | < 130 wpm |

Every visit becomes a JSON fingerprint. Across visits, deltas tell the story.

**Speaker line:** *"Six biomarker families, comparable visit to visit, all extracted from a single 60-second recording."*

---

## Slide 3 — Why on a depth camera, why on-device  (1:30 → 2:15)

**Title:** Depth + on-device aren't decoration — they unlock the product.

| Decision | Why it matters |
|---|---|
| **Facial pipeline runs on-device (OAK App)** | Patient video never leaves the camera. HIPAA / GDPR alignment by design. No cloud bill scaling per scan. |
| **Voice runs locally on the host** | Whisper-tiny + Praat (parselmouth) execute on the doctor's laptop CPU. **No audio is uploaded** to OpenAI or any cloud — same privacy guarantee extended to voice. |
| **Stereo depth** | Smile amplitude in mm, not pixels. "Smile width dropped from 52 mm → 38 mm" has clinical meaning; pixel deltas don't. Tremor power normalized by face-diagonal so the 4–6 Hz band is camera-distance invariant. |
| **OAK 4 D specifically** | DSP runs YuNet + MediaPipe FaceMesh + StereoDepth + our BiomarkerExtractor + TremorTracker in real-time. PoE single-cable install at the doctor's desk. |

**Speaker line:** *"This wouldn't work as a webcam-plus-cloud app. Privacy is the product — and that includes voice, run locally."*

---

## Slide 4 — Live demo  (2:15 → 3:15)

**Switch to dashboard browser tab.**

1. Open dashboard, select **`demo_patient`** — 4 visits across 7 months, with the `SYNTHETIC` badge visible in the header and on every report.
2. Show two sections: **Facial biomarkers (on-device)** — blink ↓ 18.5→11.2 /min, smile ↓ 52→37 mm, asymmetry ↑ 0.04→0.09, jaw tremor power ↑ 9× and locked at ~5 Hz. **Voice biomarkers (local)** — jitter ↑ 0.55→1.85 %, shimmer ↑ 2.4→5.5 %, HNR ↓ 22.5→15.6 dB, speech rate ↓ 165→118 wpm. *"These four visits illustrate the progression signature our pipeline is built to detect."*
3. *"Now let's add today's live visit, end-to-end."*
4. Switch to terminal: `python3 main.py --patient_id demo_live --duration 60`
5. While the OAK records video, voice capture starts on the laptop in parallel. Narrate the patient asking-and-saying a short sentence (the standard "rainbow passage"). Point at the HUD: blink rate, smile mm, asymmetry update in real-time on-device.
6. After 60 s the visit closes; in another ~10 s, parselmouth and Whisper-tiny finish locally and the JSON is rewritten with `voice` features.
7. Refresh dashboard, switch to **`demo_live`** → healthy values across all 6 families.
8. Click **"Download PDF report"** → *"Two sections, ten cards. Facial computed on the OAK. Voice computed on this laptop, never uploaded. No patient data ever left this room."*

**Backup:** if camera misbehaves, play the pre-recorded demo screen capture.

---

## Slide 5 — What's next + ask  (3:15 → 4:00)

**What we validated today:** the full pipeline runs end-to-end on a Mac + OAK 4 D — face detection → 468-point FaceMesh → stereo-depth fusion → BiomarkerExtractor (blink, hypomimia, asymmetry, gaze, jaw/lip tremor FFT) → parallel host-side voice capture → Praat (jitter/shimmer/HNR/F0/intensity) + Whisper-tiny LOCAL (speech rate, pause ratio) → merged JSON visit → dashboard → PDF report. Reference ranges drawn from cited literature. The progression curve is illustrated with synthetic data; real-patient longitudinal validation is the v3 goal and requires IRB review.

**Roadmap:**
- v1 (today): **6 biomarker families**, single-camera + local voice, MVP dashboard, longitudinal trends
- v2: gait/posture biomarkers (full-body OAK), MDS-UPDRS automatic estimation, multilingual voice
- v3: longitudinal validation against PARK dataset (Univ. Rochester, request access pending) and clinical pilot
- v4: HL7 / FHIR export to EHR systems

**Looking for:** clinical collaborator (movement disorder neurologist) for a 50-patient pilot to validate against MDS-UPDRS scores.

**Speaker line:** *"Parkinson's research has the patients and the clinical questions. We have a tool that runs anywhere a power outlet does. Let's connect them."*

---

## Q&A prep — likely judge questions

**Q: "How accurate is this vs MDS-UPDRS?"**
A: We have NOT clinically validated. Our biomarkers are individually validated in cited literature; combining them is novel and would need a pilot. We're upfront about this — the disclaimer is on every report.

**Q: "Can someone fake healthy results?"**
A: Hypomimia (intentional dampening of expression) is the easiest to fake. Tremor and saccade noise are nearly impossible. The longitudinal nature is the defense — fakers can't fake their own past consistently.

**Q: "Why not just use a phone camera?"**
A: (1) Privacy: phones round-trip to cloud. (2) Depth in mm: phone TrueDepth has 30 cm range; OAK has 10 m. (3) Standardization: a fixed clinic camera gives consistent illumination + framing across visits.

**Q: "What about other neurodegenerative diseases?"**
A: The biomarker framework is disease-agnostic. Bell's palsy stands out instantly via asymmetry. Huntington shows in saccades. ALS shows in voice (v2). The platform extends.

**Q: "Edge cases — glasses, beards, dark skin?"**
A: YuNet + MediaPipe FaceMesh are trained on diverse datasets. We have not run our own bias audit — that's a v2 task before any clinical use.

**Q: "How long until clinical validation?"**
A: Realistic: 18–24 months for a single-site IRB-approved pilot. We're a hack project — we built the tool. The science needs the medical institution.
