"""
NeuroVista — Parkinson facial biomarker screening (OAK App).

Pipeline:
  Camera CAM_A (RGB) ─┬─ resize ─→ YuNet face det ─┬─→ Script crop ─→ MediaPipe face landmarker
                      │                              │
  Stereo CAM_B/CAM_C ─→ StereoDepth (HIGH_DETAIL, aligned to CAM_A)
                                                     │
                     ┌─────────────── GatherData ────┘
                     ▼
       ┌─ BiomarkerExtractor (HostNode) ──────────────────────────────┐
       │  • BlinkRateNode  — eye aspect ratio + sliding window        │
       │  • HypomimiaNode  — smile/brow amplitude in mm via depth     │
       │  • AsymmetryNode  — L/R landmark mirror score                │
       │  • GazeStabilityNode — saccade frequency (later)             │
       │  Emits VisitMetrics buffer every second                       │
       └─────────────────────────────────────────────────────────────┘
                     │
                     ▼
       VisitWriter — accumulates duration seconds, persists JSON to
       data/visits/<patient_id>/visit_<visit_id>.json
"""
import json
import os
import time
from pathlib import Path

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge, GatherData
from depthai_nodes.node.utils import generate_script_content

from utils.arguments import initialize_argparser
from utils.biomarker_node import BiomarkerExtractor
from utils.visit_writer import VisitWriter, DATA_DIR
from utils.annotation_node import OverlayNode
from utils.voice_capture import VoiceCaptureSession, extract_voice_features

REQ_WIDTH, REQ_HEIGHT = 1024, 768
STEREO_W, STEREO_H = 640, 400


def main():
    _, args = initialize_argparser()
    visualizer = dai.RemoteConnection(httpPort=8082)
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatform().name
    print(f"[neurovista] Platform: {platform}")
    if platform != "RVC4":
        print("[neurovista] WARNING: this app is tuned for RVC4 (OAK 4 D).")

    if args.fps_limit is None:
        args.fps_limit = 5 if platform == "RVC2" else 30

    frame_type = (
        dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
    )

    with dai.Pipeline(device) as pipeline:
        # ─── Models ───
        det_desc = dai.NNModelDescription.fromYamlFile(f"yunet.{platform}.yaml")
        det_archive = dai.NNArchive(dai.getModelFromZoo(det_desc))
        det_w, det_h = det_archive.getInputSize()

        rec_desc = dai.NNModelDescription.fromYamlFile(
            f"mediapipe_face_landmarker.{platform}.yaml"
        )
        rec_archive = dai.NNArchive(dai.getModelFromZoo(rec_desc))
        rec_w, rec_h = rec_archive.getInputSize()

        # ─── Input: live camera or replay video ───
        if args.video:
            replay = pipeline.create(dai.node.ReplayVideo)
            replay.setReplayVideoFile(Path(args.video))
            replay.setOutFrameType(frame_type)
            replay.setLoop(False)
            replay.setFps(args.fps_limit)
            replay.setSize(REQ_WIDTH, REQ_HEIGHT)
            input_out = replay.out
            stereo_depth = None  # depth not available on replay
            calib = device.readCalibration2()
        else:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            input_out = cam.requestOutput(
                size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=args.fps_limit
            )

            # ─── Stereo depth aligned to RGB ───
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth).build(
                left=left.requestOutput((STEREO_W, STEREO_H), fps=args.fps_limit),
                right=right.requestOutput((STEREO_W, STEREO_H), fps=args.fps_limit),
                presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            )
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setLeftRightCheck(True)
            stereo.setRectification(True)
            stereo_depth = stereo.depth
            calib = device.readCalibration2()

        # ─── Face detection ───
        resize = pipeline.create(dai.node.ImageManip)
        resize.initialConfig.setOutputSize(det_w, det_h)
        resize.initialConfig.setReusePreviousImage(False)
        resize.inputImage.setBlocking(True)
        input_out.link(resize.inputImage)

        det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
            resize.out, det_archive
        )

        # ─── Crop face → landmarks ───
        det_bridge = pipeline.create(ImgDetectionsBridge).build(det_nn.out)
        script = pipeline.create(dai.node.Script)
        det_bridge.out.link(script.inputs["det_in"])
        input_out.link(script.inputs["preview"])
        script.setScript(
            generate_script_content(resize_width=rec_w, resize_height=rec_h)
        )

        crop = pipeline.create(dai.node.ImageManip)
        crop.inputConfig.setWaitForMessage(True)
        script.outputs["manip_cfg"].link(crop.inputConfig)
        script.outputs["manip_img"].link(crop.inputImage)

        landmark_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
            crop.out, rec_desc
        )

        # ─── Sync detections + landmarks ───
        gather = pipeline.create(GatherData).build(args.fps_limit)
        landmark_nn.out.link(gather.input_data)
        det_nn.out.link(gather.input_reference)

        # ─── Biomarker extraction (custom HostNode) ───
        biomarkers = pipeline.create(BiomarkerExtractor).build(
            gather_out=gather.out,
            depth_out=stereo_depth,
            calib=calib,
            fps=args.fps_limit,
        )

        # ─── Overlay annotations on the live feed ───
        overlay = pipeline.create(OverlayNode).build(
            metrics_out=biomarkers.metrics_out,
            gather_out=gather.out if args.debug else None,
            debug=args.debug,
            show_hud=not args.no_hud,
            clean_mask=args.clean_mask,
        )

        # ─── Persist visit JSON ───
        writer = None
        if not args.no_save:
            writer = pipeline.create(VisitWriter).build(
                metrics_out=biomarkers.metrics_out,
                patient_id=args.patient_id,
                visit_id=args.visit_id,
                duration_s=args.duration,
                sex=args.sex,
                age=args.age,
                label=args.label,
                state=args.state,
            )

        # ─── Visualizer topics ───
        visualizer.addTopic("Video", det_nn.passthrough, "images")
        visualizer.addTopic("Detections", det_nn.out, "images")
        visualizer.addTopic("Biomarkers", overlay.overlay_out, "images")

        # ─── Voice capture (host-side, parallel thread) ───
        # Disabled when: no_save (preview mode), no_voice flag, or processing a video
        # file (no synced mic for offline ingestion).
        voice_sess = None
        run_voice = (
            not args.no_save and not args.no_voice and args.video is None
        )
        if run_voice:
            voice_sess = VoiceCaptureSession(duration_s=args.duration)
            if not voice_sess.start():
                print(f"[neurovista] voice capture skipped: {voice_sess.error}")
                voice_sess = None

        print("[neurovista] Pipeline ready, starting…")
        pipeline.start()
        visualizer.registerPipeline(pipeline)
        while pipeline.isRunning():
            # Auto-stop when the visit JSON has been written (--save mode)
            if writer is not None and getattr(writer, "flushed", False):
                print("[neurovista] visit completed — wrapping up.")
                break
            time.sleep(0.05)

        # ─── Voice post-processing & merge into visit JSON ───
        if voice_sess is not None:
            wav = voice_sess.stop()
            print("[voice] running parselmouth + whisper …")
            t0 = time.monotonic()
            features = extract_voice_features(
                wav, run_whisper=not args.no_whisper
            )
            print(f"[voice] features computed in {time.monotonic() - t0:.1f}s")
            try:
                if wav and Path(wav).exists():
                    os.remove(wav)
            except Exception:
                pass
            _merge_voice_into_visit(
                args.patient_id, args.visit_id or _today(), features
            )


def _today():
    from datetime import date
    return date.today().isoformat()


def _merge_voice_into_visit(patient_id: str, visit_id: str, voice_features: dict):
    """Open the just-written visit JSON, attach voice features, rewrite."""
    path = Path(DATA_DIR) / patient_id / f"visit_{visit_id}.json"
    if not path.exists():
        print(f"[voice] visit JSON not found at {path}; skipping merge")
        return
    try:
        with open(path) as f:
            visit = json.load(f)
        visit["voice"] = voice_features
        with open(path, "w") as f:
            json.dump(visit, f, indent=2)
        print(f"[voice] merged into {path}")
    except Exception as e:
        print(f"[voice] merge failed: {e}")


if __name__ == "__main__":
    main()
