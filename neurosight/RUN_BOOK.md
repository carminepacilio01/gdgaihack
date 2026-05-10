# NeuroVista — Saturday Run-Book

Step-by-step playbook for the hack day. Follow in order. Every step has copy-paste commands and expected output. **If something doesn't match the expected output, stop and read the troubleshooting at the bottom.**

Camera: OAK 4 D (RVC4) over PoE+. Laptop: macOS Apple Silicon, Python 3.9, depthai 3.6.1 already installed.

---

## T+0:00 → Hardware boot (3 min)

```bash
# Power the camera via PoE+ switch (NOT plain PoE, NOT USB)
# Wait for solid blue LED on the camera (~30s)
oakctl device list
```
**Expected:** one device listed with an IP. Note it as `$OAK_IP`.

```bash
export OAK_IP=<paste-the-ip-here>
oakctl device update     # flash latest Luxonis OS first thing
oakctl connect $OAK_IP   # set as default device
```

If `oakctl device list` shows nothing: set the laptop NIC to `169.254.10.50/16`, retry.

---

## T+0:05 → Smoke test (5 min)

```bash
open -a "OAK Viewer"     # GUI; pick your device, confirm RGB + stereo + point cloud render
```
Close the viewer when done — the OAK Viewer holds the camera and will block our pipeline.

```bash
cd ~/Downloads/GDG-AI-Hack-2026/neurosight
pip install -r requirements.txt --user
```

---

## T+0:15 → Run the pipeline locally (peripheral mode) — 10 min

This is the *single biggest de-risk*. We run the same `main.py` from the laptop, with the camera attached over Ethernet, before deploying as an OAK App.

```bash
python3 main.py --no_save
```
**Expected:** prints `[neurovista] Pipeline ready, starting…`, then a WebRTC URL. Open the URL in Chrome → 3 topics visible: **Video**, **Detections** (face boxes), **Biomarkers** (HUD with blink rate, smile mm, etc.).

Sit ~50 cm from the camera. After 10 s the HUD numbers should populate (blink_rate climbs, smile mm shows ~40-50, asymmetry < 0.05 if you face it straight).

If face detection works but landmarks don't: check `requirements.txt`, `pip install --upgrade depthai-nodes`.

---

## T+0:30 → Deploy as OAK App (on-device — the 30% judging path) — 15 min

```bash
oakctl app run .
```
**Expected:** build logs (~2 min), then `[neurovista] Pipeline ready`. The visualizer URL is now `https://$OAK_IP:8082/` (or `:9000/`, check the logs).

If build fails on `pip install`: read the error. Most common — the wheel for opencv on RVC4 is missing → already pinned to `opencv-python-headless`, but if it pukes try `apt-get install -y python3-opencv` in `oakapp.toml` before the pip line.

---

## T+0:45 → Verify synthetic visits exist (2 min)

Decision logged 2026-05-09: we do **not** ingest the 40 YouTube segments anymore. Quality of frontal-face windows in patient testimonial videos is too noisy to produce a credible trend, and our scope is longitudinal-within-one-patient, not patient comparison. Demo uses `demo_patient` (synthetic, with disclosure) + `demo_live` recorded on the day.

```bash
ls data/visits/demo_patient/
# -> visit_2025-09-01.json  visit_2025-11-15.json  visit_2026-01-30.json  visit_2026-04-12.json
```

If those four don't exist, regenerate:
```bash
python3 scripts/seed_demo_visits.py
```

`videos_input/` is intentionally empty. If the live recording fails on stage, fallback is the **pre-recorded screen capture of a previous successful run** (record this during demo rehearsal at T+1:15).

---

## T+1:00 → Start the dashboard (5 min)

```bash
python3 backend/app.py --port 8080
```
Open **http://localhost:8080** in Chrome. Select `demo_patient` from the dropdown.

**Expected:** 4 KPI cards at top + 6 trend charts. Deltas vs. baseline visible (red ↓ on declining metrics across the 4 synthetic visits).

Click "Download PDF report" to verify the export works.

---

## T+1:15 → Live demo run-through (15 min)

Now wire everything together for the pitch. **Goal:** in front of the judges, do this in 90 seconds:

1. Open dashboard showing `demo_patient` with declining trends across 4 synthetic visits. Disclose explicitly: *"these four visits are simulated to illustrate progression — our v3 plan is real-patient validation"*.
2. Switch to a fresh patient: `python3 main.py --patient_id demo_live --duration 60`
3. While it records, narrate: "for contrast, here's a healthy adult — the camera will record for 60 seconds and add this visit to our longitudinal record."
4. After 60 s, refresh the dashboard, switch dropdown to `demo_live` → values are healthy → contrast hits the synthetic patient.
5. Click PDF export → "and the neurologist gets this report, generated entirely on-device — no patient video leaves the camera."

---

## T+2:30 → Polish & rehearse (until pitch)

Things to check before pitch:
- [ ] OAK App is `oakctl app enable`d so it auto-starts if power is cycled
- [ ] Demo laptop screen mirrored to the projector / external monitor
- [ ] Dashboard browser already open + pre-zoomed to 110%
- [ ] Slides loaded (5 slides — see `pitch/`)
- [ ] One backup recording of the live demo in case the network flakes

```bash
oakctl app build .
oakctl app install neurosight.oakapp
oakctl app list                # find the app id
oakctl app enable <app-id>     # auto-start on next boot
```

---

## Troubleshooting

**Red flashing LED on camera** → wrong injector. Need PoE+ (30 W), not plain PoE. Some Ubiquiti injectors flash red briefly before going solid blue — wait 10 s.

**`oakctl device list` returns nothing** → DHCP issue. On macOS: System Settings → Network → Ethernet → Configure IPv4 → Manually → 169.254.10.50, 255.255.0.0. (Note: `oakctl list` is deprecated in 0.20.0 — use `oakctl device list`.)

**`oakctl app run` fails with depthai version mismatch** → in `oakapp.toml`, pin `depthai==3.4.0` and `depthai-nodes==0.4.0` to match the example zoo manifests we copied.

**`landmarks_list[0]` fails (no faces)** → face is too far / too small. Sit closer. The YuNet model expects faces ≥ 50 px wide.

**`face_distance_mm` is `None`** → stereo depth not aligning to RGB. Confirm `setDepthAlign(CAM_A)` in `main.py`. If still null, depth packets aren't reaching the HostNode → restart the pipeline.

**Pipeline runs but blink_rate stays 0** → user not actually blinking, or EAR threshold mis-tuned. Edit `BlinkCounter(ear_threshold=0.20, ...)` in `biomarker_node.py` — try 0.22 or 0.25.

**Dashboard shows `–` for all values** → no JSON files yet. Run `python3 scripts/seed_demo_visits.py` to seed a fake `demo_patient` while you fix the pipeline.

**Asymmetry score shoots over 0.5** → user's head was tilted during recording. Re-record with head straight, eyes level.
