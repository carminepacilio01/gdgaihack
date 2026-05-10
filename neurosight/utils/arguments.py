import argparse


def initialize_argparser():
    parser = argparse.ArgumentParser(
        description="NeuroVista — Parkinson facial biomarker screening tool"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="OAK device IP address. If omitted, uses first detected device.",
    )
    parser.add_argument(
        "-fps",
        "--fps_limit",
        type=int,
        default=None,
        help="FPS cap for inference. Default 30 on RVC4, 5 on RVC2.",
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default=None,
        help="Process a video file instead of live camera (offline visit ingestion).",
    )
    parser.add_argument(
        "-pid",
        "--patient_id",
        type=str,
        default="demo_patient",
        help="Patient identifier — used for storage path.",
    )
    parser.add_argument(
        "-vid",
        "--visit_id",
        type=str,
        default=None,
        help="Visit identifier (e.g. 2026-05-09). Defaults to today.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Recording duration in seconds for one visit. Default 60.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Run pipeline without persisting visit JSON (live preview only).",
    )
    parser.add_argument(
        "--no_voice",
        action="store_true",
        help="Skip voice analysis (no microphone capture / no Whisper). Useful "
             "when running on video file or when mic is unavailable.",
    )
    parser.add_argument(
        "--no_whisper",
        action="store_true",
        help="Capture audio + acoustic features (jitter/shimmer/HNR) but skip "
             "Whisper transcription. Faster, no PyTorch needed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show all 468 facial landmarks live on the visualizer feed, with "
             "key biomarker landmarks (eyes/mouth/brows) highlighted.",
    )
    parser.add_argument(
        "--no_hud",
        action="store_true",
        help="Suppress the HUD metrics text overlay on the visualizer feed. "
             "Used by the clinician portal so the doctor sees only the "
             "face mask, no debug numbers.",
    )
    parser.add_argument(
        "--clean_mask",
        action="store_true",
        help="Render only the salient biomarker landmarks (eyes, mouth, brows, "
             "tremor anchors, axis), thinner. Skip the 463 grey background "
             "points. Used by the clinician view so the mask is minimalist.",
    )
    # Metadata for the ML training CSV embedding (lower-face landmarks).
    parser.add_argument(
        "--sex", type=int, choices=[0, 1], default=None,
        help="Patient sex for embed CSV: 0=M, 1=F. Omit if unknown.",
    )
    parser.add_argument(
        "--age", type=int, default=None,
        help="Patient age in years for embed CSV.",
    )
    parser.add_argument(
        "--label", type=int, choices=[0, 1], default=None,
        help="Ground-truth label: 0=no PD, 1=PD. Omit if unknown.",
    )
    parser.add_argument(
        "--state", type=str,
        choices=["rest", "talk", "smile", "visit"],
        default="visit",
        help="Recording state for baseline / visit context. Used to filter "
             "training data downstream.",
    )
    return parser, parser.parse_args()
