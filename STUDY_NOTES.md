# OAK Examples — Study Notes (Sentinel-focused)

**Goal:** offline-ready quick reference for building "Industrial Safety Sentinel" by adapting `human-machine-safety` + adding tracker + tuning depth zones.

**Last updated:** 2026-05-09 (pre-hack)

---

## TL;DR — what to fork

> **Base:** `oak-examples-main/neural-networks/object-detection/human-machine-safety/`
> **Add from:** `neural-networks/object-tracking/collision-avoidance/` (ObjectTracker + BirdsEyeView)
> **Optionally swap detector for:** `neural-networks/object-detection/yolo-world/` (open-vocab classes)

`human-machine-safety` already implements the exact architecture we need: 2 parallel detectors → fuse with depth → merge → filter → measure 3D distance → state-queued alert overlay. We change the labels, the threshold, and add tracking.

---

## Recurring patterns across ALL examples (memorize these)

### 1. Boilerplate top of every `main.py`

```python
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsFilter, DepthMerger, ...

_, args = initialize_argparser()                          # always argparse via utils
visualizer = dai.RemoteConnection(httpPort=8082)          # WebRTC viewer at :8082
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name                      # "RVC2" or "RVC4"; ours is RVC4
frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
```

### 2. Loading a model from the Zoo (always YAML-driven)

```python
model_description = dai.NNModelDescription.fromYamlFile(f"yolov6_nano_r2_coco.{platform}.yaml")
nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
classes = nn_archive.getConfig().model.heads[0].metadata.classes   # COCO labels for YOLO
nn_w, nn_h = nn_archive.getInputSize()
```

YAML format (`depthai_models/yolov6_nano_r2_coco.RVC4.yaml`) is **2 lines**:
```yaml
model: luxonis/yolov6-nano:r2-coco-512x288
platform: RVC4
```

### 3. Camera + Stereo setup (canonical 3-cam pattern)

```python
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)            # RGB
left_cam  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)      # stereo L
right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)      # stereo R

stereo = pipeline.create(dai.node.StereoDepth).build(
    left=left_cam.requestOutput((640, 400), fps=args.fps_limit),
    right=right_cam.requestOutput((640, 400), fps=args.fps_limit),
    presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)   # align depth to RGB
stereo.setLeftRightCheck(True)
stereo.setRectification(True)
```

**Use HIGH_DETAIL preset** — it's the default in every depth-using example.

### 4. Two paths to fuse 2D detection + depth

**Path A — single node (simplest):** `SpatialDetectionNetwork` does everything in one box
```python
nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
    input=cam, stereo=stereo, nnArchive=nn_archive, fps=args.fps_limit
)
nn.setBoundingBoxScaleFactor(0.5)   # shrink bbox for depth sampling
```
Output: `dai.SpatialImgDetections` with `.spatialCoordinates` (Point3f, mm) on each detection.
Used by: `spatial-detections`, `collision-avoidance`.

**Path B — manual (when you need 2 detectors):** detect with `ParsingNeuralNetwork`, then fuse with `DepthMerger`
```python
det_nn = pipeline.create(ParsingNeuralNetwork).build(manip.out, nn_archive)

depth_merger = pipeline.create(DepthMerger).build(
    output_2d=det_nn.out,
    output_depth=stereo.depth,
    calib_data=device.readCalibration2(),
    depth_alignment_socket=dai.CameraBoardSocket.CAM_A,
    shrinking_factor=0.1,
)
```
Used by: `human-machine-safety` (twice — once per detector, then merge results).
**Use this when running 2+ detectors in parallel.**

### 5. Object Tracking (built-in, no NN cost)

```python
tracker = pipeline.create(dai.node.ObjectTracker)
tracker.setDetectionLabelsToTrack([person_label])
tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)   # RVC4
# RVC2 uses ZERO_TERM_COLOR_HISTOGRAM
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

img_detections_filter.out.link(tracker.inputDetections)
nn.passthrough.link(tracker.inputTrackerFrame)
nn.passthrough.link(tracker.inputDetectionFrame)
```
Output: `dai.Tracklets` with `tracklet.id`, `tracklet.spatialCoordinates`, `tracklet.roi`.

### 6. HostNode skeleton (write your own logic node)

```python
class MyNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.output = self.createOutput(possibleDatatypes=[
            dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
        ])

    def build(self, input1: dai.Node.Output, input2: dai.Node.Output) -> "MyNode":
        self.link_args(input1, input2)
        return self

    def process(self, msg1: dai.Buffer, msg2: dai.Buffer):
        # do logic
        out_msg = SomeBuffer()
        out_msg.setTimestamp(msg1.getTimestamp())
        out_msg.setSequenceNum(msg1.getSequenceNum())
        self.output.send(out_msg)
```
Always preserve `Timestamp` and `SequenceNum` from upstream. Always emit on `self.output.send()`.

### 7. Visualizer wiring (WebRTC viewer auto-served)

```python
visualizer.addTopic("Color", camera_output)
visualizer.addTopic("Detections", annotation_node.out_detections)
visualizer.addTopic("Distances", visualize_distances.output)
visualizer.addTopic("Alert", show_alert.output)

pipeline.start()
visualizer.registerPipeline(pipeline)
while pipeline.isRunning():
    if visualizer.waitKey(1) == ord("q"): break
```
URL: `https://<OAK4_IP>:9000/` (the actual port is shown in oakctl logs).

### 8. AnnotationHelper for overlays (rectangles, lines, text)

```python
from depthai_nodes.utils import AnnotationHelper

helper = AnnotationHelper()
helper.draw_rectangle(top_left=(0, 0), bottom_right=(1, 1),
                      outline_color=dai.Color(1, 0, 0, 1),
                      fill_color=dai.Color(1, 0, 0, 0.1), thickness=10)
helper.draw_text(text="Too close!", position=(0.3, 0.5),
                 color=dai.Color(1, 0, 0, 1), size=64)
helper.draw_line(pt1=(x1, y1), pt2=(x2, y2), thickness=2)

img_annotations = helper.build(timestamp=ts, sequence_num=seq)
self.output.send(img_annotations)
```
Coordinates are **normalized 0..1** (frame-relative).

---

## human-machine-safety — anatomical breakdown (this is OUR base)

### File map
| File | Role | Reusable as-is? |
|---|---|---|
| `main.py` | Pipeline graph | Customize: change `DANGEROUS_OBJECTS`, add tracker, add LOW_LIGHT branch |
| `utils/measure_object_distance.py` | 3D Euclidean dist between every detection pair | ✅ keep |
| `utils/show_alert.py` | State-queue stability filter + red overlay | ✅ keep, tune `DISTANCE_THRESHOLD` |
| `utils/visualize_object_distances.py` | Draws line + "X.X m" between detections | ✅ keep |
| `utils/detection_merger.py` | Merges two detection messages into one | ✅ keep |
| `utils/annotation_node.py` | Converts SpatialImgDetections → ImgDetectionsExtended; colorizes depth | ✅ keep |
| `utils/arguments.py` | argparse (-d device, -fps) | ✅ keep |
| `depthai_models/*.yaml` | YOLOv6 + Palm declarations | ✅ keep, optionally swap YOLOv6 for yolo-world |
| `oakapp.toml` | Manifest (no frontend, simplest variant) | ✅ keep |

### Key constants to tune (in `show_alert.py`)
```python
DISTANCE_THRESHOLD = 500      # mm — alert when palm < 50cm from machine
ALERT_THRESHOLD = 0.3         # state queue % over which alert fires
STATE_QUEUE_LENGTH = 5        # frames of stability to avoid jitter
```

### How its detections compose

```
RGB ─→ ImageManip (512×288) ─→ YOLOv6-nano ─┐
                                              ├─→ DetectionMerger ─→ filter ─→ MeasureObjectDistance ─┬─→ ShowAlert
RGB ─→ ImageManip (192×192) ─→ MediaPipe-Palm ┘                              │                       └─→ VisualizeObjectDistances
                                                                              └─→ AnnotationNode
StereoDepth ─→ DepthMerger (×2, one per branch above)                              ↓
                                                                          colored depth + bboxes
```

### Sentinel modifications needed

1. **Replace dangerous objects list:** instead of `["bottle", "cup"]`, use whatever YOLOv6 detects in your scene. Easiest: pick "scissors", "knife", "laptop" if visible. Or swap detector to `yolo-world` and prompt classes as text strings ("hot machine", "moving belt").
2. **Add ObjectTracker** between the merged detections and the visualizer for persistent IDs (copy from `collision-avoidance/main.py` lines 74-84).
3. **Tune zone:** `DISTANCE_THRESHOLD = 300` (mm) for tighter zone in demo.
4. **Add bird's-eye-view** overlay (copy `BirdsEyeView` HostNode from `collision-avoidance/utils/host_bird_eye_view.py`).
5. **Optional GPIO trigger:** in `show_alert.process()`, if `_should_alert`, also write to `/sys/class/gpio/...` if M8 cable available. (TBD on Saturday.)

---

## Useful patterns from collision-avoidance

| Pattern | What it gives | File |
|---|---|---|
| `SpatialDetectionNetwork` (single node, depth fused inside) | Simpler than DepthMerger when only 1 detector | `main.py:59` |
| `ObjectTracker` setup | ID persistence for tracked entities | `main.py:74-84` |
| `BirdsEyeView` HostNode | Top-down map (great for dashboards) | `utils/host_bird_eye_view.py` |
| `CollisionAvoidanceNode` | Per-tracklet trajectory + speed + TTI via `np.polyfit` | `utils/collision_avoidance_node.py` |
| Position history dict per tracklet | `{id: {"x": [...], "z": [...], "timestamp": [...]}}` | same |

The trajectory pattern is **gold** for predictive alerts: "person approaching zone, ETA 1.2s" beats "person in zone."

---

## YOLO-World — open-vocabulary upgrade (if YOLOv6 classes aren't enough)

```python
from utils.helper_functions import extract_text_embeddings

text_features = extract_text_embeddings(
    class_names=args.class_names,         # ["worker without helmet", "robot arm", ...]
    max_num_classes=80
)

nn_with_parser = pipeline.create(ParsingNeuralNetwork)
nn_with_parser.setNNArchive(model_nn_archive)
nn_with_parser.setBackend("snpe")                                        # RVC4-specific
nn_with_parser.setBackendProperties({"runtime": "dsp", "performance_profile": "default"})
nn_with_parser.getParser(0).setConfidenceThreshold(args.confidence_thresh)

input_node.link(nn_with_parser.inputs["images"])
text_input_q = nn_with_parser.inputs["texts"].createInputQueue()
nn_with_parser.inputs["texts"].setReusePreviousMessage(True)

# After pipeline.start():
inputNNData = dai.NNData()
inputNNData.addTensor("texts", text_features, dataType=dai.TensorInfo.DataType.U8F)
text_input_q.send(inputNNData)
```

**Pros:** zero training, classes are runtime args.
**Cons:** RVC4-only (we have RVC4 ✓), default 5 FPS in the example, slower than YOLOv6.

**For Sentinel:** if we want PPE detection ("person without helmet") without training a custom model, this is the path.

---

## hand-pose — multi-stage detection→landmark pipeline

Architecture is more complex (palm detector → crop config → Script node + ImageManip → hand-landmarker → GatherData sync). **Use only if we want gesture controls in the Sentinel UI**, otherwise skip.

Key insight: the `dai.node.Script` runs **on-device Python** to dynamically configure ImageManip (e.g. crop the palm out of the full frame). Pattern useful for any 2-stage detect→classify pipeline where the second stage needs runtime crops.

---

## oakapp.toml — three real variants

### Minimal Python only (human-machine-safety)
```toml
identifier = "com.example.object-detection.human-machine-safety"
app_version = "1.0.0"

prepare_container = [
    { type = "RUN", command = "apt-get update" },
    { type = "RUN", command = "apt-get install -y python3-pip" },
    { type = "COPY", source = "requirements.txt", target = "requirements.txt" },
    { type = "RUN", command = "pip3 install -r /app/requirements.txt --break-system-packages" },
]

depthai_models = { yaml_path = "./depthai_models" }
entrypoint = ["bash", "-c", "python3 -u /app/main.py"]
```
**Use for Sentinel.** Simplest. No frontend = no Node build = ~2 min build time.

### With React frontend (dino-tracking, p2p, object-volume)
Adds:
- `prepare_build_container` — installs nvm + Node 24
- `[static_frontend]` block with `dist_path` and build steps
- `entrypoint` becomes `runsvdir -P /etc/service` + a `backend-run.sh` script
- `image_tag = "1.2.6"` (or `1.2.6-py311` for Python 3.11)
- `assign_frontend_port = true` for some apps

**Use only if** we want a slick custom UI. Default RemoteConnection viewer at port 8082/9000 is enough for most demos. Skip the frontend overhead unless we need interactive controls.

### Python 3.11 base (p2p-measurement)
Same structure but `image_tag = "1.2.6-py311"`. Use if we hit a Python 3.12 compat issue with a dependency.

---

## Version pinning — what's installed vs what examples expect

| | We have | Examples pin |
|---|---|---|
| `depthai` | **3.6.1** | 3.0.0 / 3.2.1 / 3.3.0 / 3.4.0 |
| `depthai-nodes` | **0.3.6** | 0.3.4 / 0.3.7 / 0.4.0 |

**Likely fine** (newer = backward compatible). If `import` breaks, downgrade to match the specific example: `pip3 install --user depthai==3.4.0 depthai-nodes==0.4.0`.

---

## Sentinel pipeline (final mental model)

```
                        ┌─→ YOLOv6-nano ─┐  (or yolo-world)
                        │                  ├─→ DepthMerger ─┐
                        │   ┌──────────────┘                 │
RGB CAM_A ──→ ImageManip│   │                                ├─→ DetectionMerger
              ─────────→│   │                                │   ├─→ ImgDetectionsFilter (palm + dangerous)
                        │   │                                │   │   ├─→ MeasureObjectDistance ─┬─→ ShowAlert ──→ Visualizer "Alert"
                        │   │                                │   │   │                          └─→ VisualizeObjectDistances ──→ Visualizer "Distances"
                        │   │                                │   │   └─→ ObjectTracker ────────→ Visualizer "Tracklets"
                        └─→ MediaPipe-Palm ──→ DepthMerger ──┘   │
                                                                  └─→ AnnotationNode ──→ Visualizer "Detections" + "Depth"

Stereo: CAM_B + CAM_C ──→ StereoDepth(HIGH_DETAIL, align CAM_A) ──→ DepthMerger inputs (×2)

Optional add-ons:
  - BirdsEyeView from tracker.out → "BirdView" topic (top-down map)
  - CollisionAvoidanceNode for trajectory + ETA prediction
  - GPIO trigger inside ShowAlert (Saturday TBD)
```

---

## Saturday execution plan (when you have the camera)

1. **00:00** — Boot OAK4, `oakctl device update`, OAK Viewer smoke test (RGB + depth visible).
2. **00:30** — `cd oak-examples-main/neural-networks/object-detection/human-machine-safety/`, `pip install -r requirements.txt`, run **peripheral mode** first: `python3 main.py`. Confirm baseline works on laptop ↔ camera. **Single biggest de-risk.**
3. **01:00** — `oakctl connect <IP>` then `oakctl app run .` from same folder. Wait for build. Confirm WebRTC view at the URL printed.
4. **01:30** — Copy folder to `~/Downloads/GDG-AI-Hack-2026/sentinel/` and start mods:
   - Change `DANGEROUS_OBJECTS` to objects you actually have on the table
   - Drop `DISTANCE_THRESHOLD` to 300 mm
   - Add ObjectTracker block from collision-avoidance
5. **02:30** — IR low-light test: dim the room, verify depth still streams (it should — OV9282 is 940nm-sensitive).
6. **03:00+** — Polish: BirdsEyeView overlay, slide deck, demo rehearsal.

---

## Gotchas to remember

- The `requestOutput()` needs both **size** and **fps** for stereo to behave: `left_cam.requestOutput((640, 400), fps=args.fps_limit)`.
- `setBoundingBoxScaleFactor(0.5)` shrinks bbox before depth sampling — needed because edge pixels have unreliable depth.
- `ImgDetectionsFilter` takes `labels_to_keep`, NOT `labels_to_drop`.
- In `DetectionMerger` you must call `set_detection_2_label_offset(len(classes_of_first))` to avoid label collisions when merging two detector outputs.
- HostNodes execute **on-host**, not on-device — at scale they can bottleneck. Keep their logic light or convert to `Script` node (on-device Python).
- `depthai_models/*.yaml` filenames must match the string passed to `fromYamlFile()` — **including** the `.{platform}.yaml` suffix.
- WebRTC viewer port differs between local dev (`:8082` from `RemoteConnection`) and standalone OAK App (`:9000`). Both are normal.
