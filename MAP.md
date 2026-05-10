# GDG AI Hack 2026 — Master Reference Map

**Track B — See Beyond** (Vision AI, sponsored by Luxonis)
**Hardware:** OAK 4 D camera + PoE+ switch + ETH cabling (provided Saturday)
**Start:** 2026-05-09 09:30
**Last updated:** 2026-05-09

> This is the single source of truth for the hack. All reference info lives here. Update it as we learn things on-site.

---

## Table of contents
1. [Judging criteria (memorize)](#judging-criteria-memorize)
2. [Mental model of the stack](#mental-model-of-the-stack)
3. [Saturday boot order](#saturday-boot-order)
4. [oakctl command cheatsheet](#oakctl-command-cheatsheet)
5. [OAK App scaffold (the 30% path)](#oak-app-scaffold-the-30-path)
6. [Top examples to fork](#top-examples-to-fork)
7. [Curated repo survey — luxonis/oak-examples](#curated-repo-survey--luxonisoak-examples)
8. [Hardware specs (OAK 4 D)](#hardware-specs-oak-4-d)
9. [All documentation links](#all-documentation-links)
10. [Gotchas & known issues](#gotchas--known-issues)
11. [Doc gaps — ask a Luxonis mentor](#doc-gaps--ask-a-luxonis-mentor)
12. [Project ideas brainstorm](#project-ideas-brainstorm)
13. [What's installed where](#whats-installed-where)

---

## Judging criteria (memorize)

| Weight | Dimension | What it means |
|---|---|---|
| **30%** | Creative On-Device Use | Camera is **not** used as a webcam. Pipeline runs as an **OAK App** on the camera. |
| **25%** | Depth Usage | Stereo or neural depth is part of the actual solution, not decoration. |
| **25%** | Advanced CV/AI Models | Custom-trained model OR multiple stacked models (detector → tracker → classifier). |
| **20%** | Practical Utility | Solves a real-world problem. |

**Plus general dimensions** (apply across all tracks): innovation, technical execution, real-world impact, presentation.

**Mandatory rules:** Have fun. Use the OAK 4 camera. That's it.

---

## Mental model of the stack

You write a DepthAI **Pipeline** (graph of nodes: cameras → stereo depth → neural net → tracker → image manip) on your **laptop** in Python. To run it locally for dev, just `python script.py` against the camera over Ethernet. To **run on-device** (the 30% judging path), wrap it as an **OAK App** — a folder with `main.py` + `requirements.txt` + `oakapp.toml` — and ship it with `oakctl`. The camera runs Luxonis OS (Linux 5.15) and executes your app standalone. **OAK Viewer** GUI is for poking at streams without code; it does not accept custom pipelines.

| Component | Where it runs | Purpose |
|---|---|---|
| DepthAI v3 SDK | Laptop | Build pipelines in Python/C++ |
| oakctl | Laptop | CLI: device mgmt + app deploy |
| OAK Viewer | Laptop | GUI for visualizing streams |
| OAK Apps | **Camera** | Containerized pipelines running on-device |
| Luxonis OS | Camera | Linux 5.15 base on the OAK 4 D |
| Model Conversion + Zoo | Laptop | Prep custom NNs for the camera's DSP/GPU |

---

## Saturday boot order

1. **Cable**: OAK 4 D → PoE+ switch → ETH → laptop. **Must be PoE+ (30W)**, not regular PoE — wrong injector = red flashing LED.  USB power crashes the camera (peaks 25 W).
2. **Wait for solid blue LED** (steady = ready).
3. `oakctl list` → find the camera. DHCP is tried first, then link-local `169.254.0.0/16`. If you're plugged direct laptop ↔ switch ↔ camera with no router, set your laptop NIC to `169.254.10.50/16` and the camera will be findable.
4. `oakctl device update` — flash the latest OS first thing (OS 1.20.5 has a static-IP bug — update before configuring).
5. Open **OAK Viewer**, select the device from the header dropdown, confirm RGB / stereo / point cloud streams.
6. `pip install depthai --force-reinstall` on your machine, then run **Spatial Detection** example to confirm the SDK end-to-end.
7. Once Spatial Detection works on the laptop, immediately wrap it as an OAK App and `oakctl app run ./folder` — getting one app running on-device early is the single biggest de-risk for the 30% judging weight.

---

## oakctl command cheatsheet

```bash
# ---- Discovery & device mgmt ----
oakctl list                                  # list cameras on network
oakctl device update                         # flash latest Luxonis OS
oakctl device unlock                         # enable SSH (run once)
ssh root@<IP>                                # shell on the camera
oakctl adb devices                           # USB-attached devices
oakctl adb shell                             # shell via USB (ADB)
oakctl adb shell agentconfd factory-reset    # if reset button is broken

# ---- Network config ----
agentconfd configure 192.168.10.15,255.255.255.0,192.168.10.1   # static IP (run on camera)
agentconfd configure dhcp                                        # back to DHCP
agentconfd configure dhcp --dns 10.12.102.2                      # DHCP + custom DNS

# ---- File transfer ----
scp -r folder/ root@<IP>:/data/
oakctl adb push folder/ /data/

# ---- OAK Apps lifecycle ----
oakctl app run ./my-oak-app                  # build + run from source (dev)
oakctl app build ./my-oak-app                # produces my-oak-app.oakapp
oakctl app install my-oak-app.oakapp         # persistent install
oakctl app list
oakctl app start <id>
oakctl app stop <id>
oakctl app logs <id>
oakctl app enable <id>                       # auto-start on boot (demo-day move)
```

**Update oakctl itself:** `oakctl self-update`

---

## OAK App scaffold (the 30% path)

This is the most important pattern in the entire hack. Project layout:

```
my-oak-app/
├── main.py            # DepthAI pipeline code (same API as laptop)
├── requirements.txt   # Python deps
├── oakapp.toml        # required manifest
└── .oakappignore      # optional: files to skip during build
```

### Minimal `oakapp.toml`

```toml
identifier = "com.yourteam.appname"
entrypoint = ["bash", "-c", "python3 /app/main.py"]

prepare_container = [
  { type = "COPY", source = "requirements.txt", target = "requirements.txt" },
  { type = "RUN", command = "pip3 install -r /app/requirements.txt --break-system-packages" },
]
```

### Dev loop

```bash
oakctl app run ./my-oak-app    # iterate fast
# then for the demo:
oakctl app build ./my-oak-app
oakctl app install my-oak-app.oakapp
oakctl app enable <id>          # auto-start when judges power it up
```

---

## Top examples to fork

Paths under `https://docs.luxonis.com/software-v3/depthai/examples/`:

| # | Example | Path | Hits |
|---|---|---|---|
| 1 | **Spatial Detection** | `spatial_detection_network/spatial_detection/` | On-device + Depth + Advanced CV — best baseline |
| 2 | **Object Tracker** | `object_tracker/object_tracker/` | 3D track IDs in mm — stack on top of #1 |
| 3 | **Detection Network Remap** | `detection_network/detection_network_remap/` | Cleanest 2D NN → 3D depth fusion |
| 4 | **Neural Depth RGBD** | `neural_depth/neural_depth_rgbd/` | Colored point cloud — visually impressive for demo |
| 5 | **Neural Assisted Stereo** | `stereo_depth/neural_assisted_stereo/` | Fuses neural + classical stereo (model stacking 25%) |
| 6 | **ImageManip All Ops** | `image_manip/image_manip_all_ops/` | Custom preprocessing chain |
| 7 | **Feature Tracker** | `feature_tracker/feature_tracker/` | Optical flow on-device |
| 8 | **IMU** | `imu/imu_accelerometer_gyroscope/` | Free signal — the camera has an IMU |
| 9 | **Camera Undistort** | `camera/camera_undistort/` | Lens correction |
| 10 | **Spatial Location Calculator** | `spatial_location_calculator/` | 3D coords for arbitrary points |

**Avoid the `rvc2/*` examples** — those target legacy RVC2 hardware (older OAKs), not the OAK 4 D.

---

## Curated repo survey — luxonis/oak-examples

**Local path:** `~/Downloads/GDG-AI-Hack-2026/oak-examples-main/`
**Source:** zip of `main` branch (DepthAI v3). The `master` branch is legacy DepthAI v2 — do not use.

### What you get out of the box
- **80+ examples**, all packaged as OAK Apps (each has its own `oakapp.toml`)
- All models are **pre-trained**, pulled from the [Luxonis Model Zoo](https://models.luxonis.com) by URL — no training scripts in this repo
- The OAK 4 D = **RVC4** platform. Filter compatibility tables for ✅ in the **RVC4 (standalone)** column — that's our deployment target

### Canonical OAK App structure (memorize)

```
my-app/
├── main.py            # DepthAI pipeline (entry point)
├── requirements.txt   # Python deps (depthai, depthai-nodes, opencv-python-headless...)
├── oakapp.toml        # manifest
├── depthai_models/    # YAML files declaring zoo models the app needs
├── utils/             # arguments parser, etc.
├── media/             # demo gifs/screenshots
└── README.md
```

Reference manifest (`apps/default-app/oakapp.toml` simplified):

```toml
identifier = "com.example.apps.default-app"
entrypoint = ["bash", "-c", "/usr/bin/runsvdir -P /etc/service"]
app_version = "1.0.0"
assign_frontend_port = true                                  # opens a web port for UI

prepare_container = [
  { type = "COPY", source = "./requirements.txt", target = "./requirements.txt" },
  { type = "RUN", command = "python3.12 -m pip install -r /app/requirements.txt --break-system-packages" },
]

build_steps = [
  "mkdir -p /etc/service/backend",
  "cp /app/backend-run.sh /etc/service/backend/run",
  "chmod +x /etc/service/backend/run",
]

depthai_models = { yaml_path = "./depthai_models" }          # zoo model declarations

[base_image]                                                  # always Luxonis base
image_name = "luxonis/oakapp-base"
image_tag = "1.2.5"                                           # or "1.2.6-py311"
```

**Apps with a React frontend** (e.g. `apps/object-volume-measurement-3d`) add `[static_frontend]` and a `prepare_build_container` that installs Node via nvm during build.

### Standalone deploy workflow on RVC4 (the 30% path)

```bash
oakctl connect <DEVICE_IP>     # one-time, sets the active device
oakctl app run .               # build + push + run from current folder
# or for a permanent install:
oakctl app build .             # creates my-app.oakapp
oakctl app install my-app.oakapp
oakctl app enable <id>         # auto-start on boot — flip this for the demo
```

### Top picks for the hackathon (RVC4 standalone ✓)

#### Tier 1 — fork-and-demo (already a complete app you can adapt in hours)
| Path | Hits | Why |
|---|---|---|
| `apps/object-volume-measurement-3d/` | On-device + Depth + UI | **Full-stack: Python backend + React frontend.** Measures real-world boxes in 3D. Perfect demo template. |
| `apps/p2p-measurement/` | On-device + Depth | Click two points → real-world distance. Trivially adaptable to AR-style overlays. |
| `apps/dino-tracking/` | On-device + Advanced CV | Click any object → DINO embeddings + FastSAM track it. **Zero-shot, no class list needed.** Killer demo. |
| `apps/people-demographics-and-sentiment-analysis/` | On-device + Stacked models | Person detect → face → age/gender/emotion + re-ID. 5-model stack. |
| `apps/focused-vision/` | On-device | 2-stage detection that crops + re-runs at higher res. Great for tiny-object problems. |
| `apps/data-collection/` | On-device + UI | YOLOE for auto data capture with interactive UI. Useful as scaffolding for any project. |
| `apps/qr-tiling/` | On-device + UI | High-res QR detection via dynamic tiling. Industrial scanning vibe. |

#### Tier 2 — stack ingredients (combine into your own pipeline)
| Path | Role |
|---|---|
| `neural-networks/object-detection/yolo-world/` | **Open-vocabulary detection** — write classes as text prompts at runtime. **RVC4-only**, hugely impressive. |
| `neural-networks/depth-estimation/neural-depth/` | RVC4-only neural depth (better than classical stereo). |
| `neural-networks/depth-estimation/crestereo-stereo-matching/` | Cross-platform neural stereo. |
| `neural-networks/object-detection/spatial-detections/` | YOLOv6 + 3D coords in mm. The base spatial-detection block. |
| `neural-networks/object-tracking/deepsort-tracking/` | DeepSORT with OSNet re-ID. Adds persistent IDs. |
| `neural-networks/object-tracking/collision-avoidance/` | Detection + 3D distance gate. Robotics-style. |
| `neural-networks/segmentation/depth-crop/` | Segment by class, then mask by depth. |
| `neural-networks/feature-detection/xfeat/` | XFeat keypoint matching — for SLAM-ish demos. |
| `neural-networks/pose-estimation/hand-pose/` | Mediapipe palm + landmarks. Gesture controllers. |
| `neural-networks/pose-estimation/human-pose/` | YOLOv6 person + Lite-HRNet. Form/posture scoring. |
| `neural-networks/speech-recognition/whisper-tiny-en/` | **Whisper running on the camera.** RVC4-only. Audio-input demos rare in vision hacks. |
| `neural-networks/ocr/license-plate-recognition/` | YOLO + LP detector + Paddle OCR. RVC4-only. |
| `neural-networks/counting/depth-people-counting/` | Counts via depth alone, **no NN model**. Light + accurate. |
| `neural-networks/3D-detection/objectron/` | 3D bounding boxes for everyday objects. |

#### Tier 3 — infrastructure (use these to polish)
| Path | Role |
|---|---|
| `tutorials/custom-models/` | PyTorch/Kornia → custom blob conversion. **The 25% custom-model path.** |
| `streaming/webrtc-streaming/` | Stream camera output to a browser — great for judges-on-the-floor demos. |
| `streaming/poe-mqtt/` | MQTT publish from camera — IoT integration demos. |
| `custom-frontend/open-vocabulary-object-detection/` | React + `@luxonis/depthai-viewer-common` template. |
| `custom-frontend/raw-stream/` | Minimal React frontend skeleton. |
| `integrations/rerun/` | [Rerun.io](https://rerun.io) for slick 3D viz. Great for presentation. |
| `integrations/foxglove/` | Foxglove robotics dashboards. |
| `integrations/roboflow-workflow/` | If you want to fine-tune via Roboflow then deploy here. |

### Surprising / inspiration finds
- **`neural-networks/face-detection/gaze-estimation/`** — eye-tracking on-device. Use for accessibility (eye-controlled UI), driver-monitoring, attention heatmaps.
- **`neural-networks/face-detection/fatigue-detection/`** — drowsiness/blink rate. Fleet/driver safety.
- **`neural-networks/pose-estimation/animal-pose/`** — pet tracking, livestock monitoring, wildlife.
- **`neural-networks/object-detection/human-machine-safety/`** — detects hands too close to industrial equipment. Real factory safety problem.
- **`neural-networks/object-detection/barcode-detection-conveyor-belt/`** — real warehouse use case.
- **`neural-networks/object-detection/social-distancing/`** — covid-era but the depth-violation pattern is reusable for any "things too close" alert.
- **`neural-networks/object-detection/text-blur/`** — privacy filter (auto-redact license plates/text in stream). GDPR demo.
- **`apps/ros/ros-follow-object/`** — actual mobile robot following. If anyone has a robot platform, this is plug-and-play.
- **`neural-networks/image-to-image-translation/` (zero-dce, esrgan, dncnn3)** — low-light enhancement, super-res, denoise. Run on-device for night-vision style demos.

### Setup pattern across examples
- Each example folder has its own `requirements.txt` — install per-example:
  ```bash
  cd oak-examples-main/apps/dino-tracking
  pip install -r requirements.txt
  ```
- Pinned versions are typically `depthai==3.4.0` and `depthai-nodes==0.4.0`. We have **3.6.1** installed — likely fine but if anything breaks, pin down to match the example.
- **Install `depthai-nodes` now** — it's used by every example for visualizer overlays:
  ```bash
  pip3 install --user depthai-nodes
  ```
- Tutorials require **Python 3.10+** (`tutorials/custom-models/README.md` says so). System Python 3.9 may work for most examples but custom-models won't.
- Standalone mode entry: `oakctl connect <DEVICE_IP> && oakctl app run .` from the example folder.

### Watch out
- **RVC4-only examples** (won't run on legacy OAKs but ✅ for our OAK 4 D): `yolo-world`, `whisper-tiny-en`, `neural-depth`, `depth-anything-v2`, `license-plate-recognition`, `fastsam-x`, all the Tier-1 apps marked `❌/❌/✅`.
- **`apps/conference-demos/`** — Luxonis shows these at trade shows. Judges have seen them. Don't copy directly.
- **`thermal-detection`** — RVC2/OAK Thermal only. Skip.
- **`foundation-stereo`** — runs on host, not on-device. Doesn't count toward the 30%.
- **Pinned depthai versions** in requirements.txt are slightly behind PyPI — if `import depthai_nodes` errors, downgrade with `pip3 install --user depthai==3.4.0 depthai-nodes==0.4.0`.

---

## Hardware specs (OAK 4 D)

- 6-core ARM CPU (Qualcomm 8-series), 8 GB RAM, 128 GB storage
- Luxonis OS (Linux 5.15 kernel)
- **DSP**: 48 TOPS (INT8) / 12 TOPS (FP16) — main inference target
- **GPU**: 4 TOPS (FP16)
- Stereo depth perception with filtering, post-processing, RGB-depth alignment
- 2D + 3D object tracking (`ObjectTracker` node)
- ImageManip node: warp/undistort, resize, crop, edge detection, feature tracking
- Power: PoE+ (802.3at, 30 W). Plain PoE (15 W) won't power it. USB peaks at 25 W → unstable.

---

## All documentation links

### Installers (already pulled — see `installers/`)
- OAK Viewer macOS Apple Silicon — https://oak-viewer-releases.luxonis.com/data/1.6.3/macos_arm_64/Viewer.zip
- OAK Viewer macOS Intel — https://oak-viewer-releases.luxonis.com/data/1.6.3/macos_x86_64/Viewer.zip
- OAK Viewer Windows — https://oak-viewer-releases.luxonis.com/data/1.6.3/windows_x86_64/Viewer.msi
- OAK Viewer Linux (deb) — https://oak-viewer-releases.luxonis.com/data/1.6.3/debian_x86_64/viewer.deb
- oakctl macOS/Linux installer — https://oakctl-releases.luxonis.com/oakctl-installer.sh
- oakctl Windows — https://oakctl-releases.luxonis.com/data/latest/windows_x86_64/oakctl.msi

### Saturday morning reads
- OAK4 Getting Started — https://docs.luxonis.com/hardware/platform/deploy/oak4-deployment-guide/oak4-getting-started/
- OAK4 Advanced Guide — https://docs.luxonis.com/hardware/platform/deploy/oak4-deployment-guide/oak4-advanced/

### Build pipelines
- DepthAI v3 SDK — https://docs.luxonis.com/software-v3/depthai/
- DepthAI examples (fork-and-adapt) — https://docs.luxonis.com/software-v3/depthai/examples/
- OAK Viewer docs — https://docs.luxonis.com/software-v3/depthai/tools/oak-viewer/

### On-device deployment (where the points are)
- OAK Apps — https://docs.luxonis.com/software-v3/oak-apps/
- luxonis/oak-examples (GitHub) — https://github.com/luxonis/oak-examples

### Reference
- Software stack overview — https://docs.luxonis.com/software-v3/
- Docs hub — https://docs.luxonis.com/
- Forum — https://discuss.luxonis.com/ (load in browser; SPA doesn't render in fetch)

---

## Gotchas & known issues

- **PoE wattage**: regular PoE (15 W) cannot power the OAK 4 D. Must be PoE+ (30 W). Symptom: red flashing LED.
- **USB power**: peaks at 25 W exceed standard USB → camera crashes mid-pipeline. Always use the PoE+ switch.
- **OS 1.20.5 static-IP bug**: setting a static IP on this version silently fails. Run `oakctl device update` before any network config.
- **Some PoE+ injectors don't signal compliance** (e.g. Ubiquiti) → red flash before solid blue. Wait through it.
- **Reset button can be flaky** — workaround: `oakctl adb shell agentconfd factory-reset`.
- **OAK Viewer is visualization-only** — you cannot load custom pipelines in it. Use it to confirm the camera works, not to develop.
- **DepthAI v2 examples won't run on OAK 4 D** — they target older RVC2 hardware. Stick to v3.
- **Laptop needs an Ethernet port.** MacBook Apple Silicon → bring a USB-C → Ethernet dongle. No dongle = no on-device dev.

---

## Doc gaps — ask a Luxonis mentor

- **Shipping a custom-trained model inside an OAK App** — the docs don't show this explicitly. Likely COPY the `.blob` / NN archive via `prepare_container` and reference it from `main.py`, but confirm the canonical pattern.
- **Per-app resource limits** (RAM/storage caps, max image size, allowed deps).
- **On-device debugging beyond `oakctl app logs`** — attaching to a running app, profiling, DSP/GPU usage telemetry.
- **Custom model conversion pipeline** — ONNX → blob/superblob workflow. Pull `https://docs.luxonis.com/software-v3/` Model Conversion section first thing tomorrow if we plan to fine-tune.
- **Multi-camera coordination** — if any teammates want to use 2 OAKs (probably not allowed, but ask).

---

## Project ideas brainstorm

> _Aim for: real problem + on-device + depth-as-signal + ≥2 stacked or fine-tuned models._

- **Posture/ergonomics coach** — pose estimation + depth gives you actual joint angles in 3D, not just 2D keypoints. Real-time feedback on slouching, screen distance, etc. (Practical utility ✓)
- **Warehouse/shelf depth audit** — detect missing items on a shelf using depth deltas vs. a reference scan. (On-device ✓ + depth ✓)
- **Accessibility — proximity announcer for visually impaired** — detection + 3D distance, audio cues for objects in the user's path. Stack object detector + scene classifier. (Utility ✓ + depth ✓)
- **Sports/form analysis** — squat / pushup / serve form, scored against ideal motion in 3D space.
- **Crowd density + flow map** — people detector + tracker + heatmap of dwell time. Useful for retail / events.
- **Industrial defect detection** — fine-tune a tiny classifier on conveyor-belt samples; reject by depth-of-field + color anomaly.
- **Volumetric package measurement** — point cloud → bounding box volume of a parcel. Stack depth filtering + plane detection. (Real shipping use case)
- **Gesture-controlled smart home / presentation remote** — hand pose + 3D depth for click-when-finger-touches-virtual-button.
- **Plant health monitor** — RGB + neural depth + small classifier for leaf disease / wilting.
- **Pet behavior tracker** — track pet in 3D, alert on falls / unusual postures (geriatric pet care is a real market).

---

## What's installed where

### macOS apps & CLI
- `OAK Viewer.app` → `/Applications/OAK Viewer.app` (quarantine attribute removed)
- `oakctl` 0.20.0 → `~/Library/Application Support/com.luxonis.oakctl/oakctl`, symlinked at `/usr/local/bin/oakctl` (on PATH)

### Python (system Python 3.9.6, user site-packages at `~/Library/Python/3.9/`)
- `depthai` 3.6.1 (DepthAI v3 SDK, OAK4-compatible)
- `opencv-python` 4.13.0.92
- `Flask` 3.1.3 + `numpy` 2.0.2
- ⚠ `depthai` CLI script lives in `~/Library/Python/3.9/bin` (not on PATH). Add to PATH if needed:
  `export PATH="$HOME/Library/Python/3.9/bin:$PATH"`
- ⚠ Pip is 21.2.4 (very old). Upgrade with `python3 -m pip install --user --upgrade pip` if any wheel install acts up.

### Project folder → `~/Downloads/GDG-AI-Hack-2026/`
- `MAP.md` — this file (master reference)
- `installers/Viewer.zip` (1.5 GB) — OAK Viewer source archive
- `installers/oakctl-installer.sh` — re-runnable / `self-update` capable
- `oak-examples/` — _(pending — to be cloned with `git clone --depth 1 --branch main https://github.com/luxonis/oak-examples.git`)_
