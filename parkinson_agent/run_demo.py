"""Synthetic capture session for offline demos and tests.

We don't try to render an anatomically faithful face — the metrics operate
on relative motion of region centroids, so what matters is that the
regions of interest (chin/jaw, lips, mouth corners, eyes) get plausible
motion patterns.

The synthetic patient has:
- Reduced amplitude in the chin/jaw and lower-lip regions during the
  expression task (hypomimia signature, weighted heavy in our priors).
- Mild 4.5 Hz jaw tremor at rest (MDS-UPDRS 3.17).
- Asymmetric mouth-corner mobility, weaker on the LEFT side.
- Reduced blink rate (8/min).

Run end-to-end with `python -m parkinson_agent.run_demo` (requires Ollama).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .oak_adapter import CaptureSession, FaceTimeSeries, TaskWindow

N_LANDMARKS = 478
FPS = 30.0


def _base_face() -> np.ndarray:
    """Anchor positions (478, 3) for a neutral, forward-facing template.

    Real FaceMesh values would come from the network. For our metrics it is
    sufficient that the *indices we read* (eyes, chin, lips, corners,
    cheeks) sit in a roughly correct geometric layout so that ranges,
    asymmetries and centroids behave like real face data.
    """
    rng = np.random.default_rng(seed=42)
    base = rng.uniform(0.4, 0.6, size=(N_LANDMARKS, 3))
    base[:, 2] = 0.0  # ignore depth for synthetic data

    # Anchor specific landmarks at canonical positions (normalized image coords).
    # Eyes (outer corners): horizontal separator for inter-ocular reference.
    base[33] = [0.40, 0.45, 0.0]   # right eye outer
    base[263] = [0.60, 0.45, 0.0]  # left eye outer
    base[159] = [0.42, 0.43, 0.0]  # right upper eyelid
    base[145] = [0.42, 0.47, 0.0]  # right lower eyelid
    base[386] = [0.58, 0.43, 0.0]  # left upper eyelid
    base[374] = [0.58, 0.47, 0.0]  # left lower eyelid
    base[158] = [0.43, 0.435, 0.0]
    base[153] = [0.43, 0.465, 0.0]
    base[385] = [0.57, 0.435, 0.0]
    base[380] = [0.57, 0.465, 0.0]

    # Nose tip and bridge — not used by metrics but stabilizes appearance.
    base[1] = [0.50, 0.55, 0.0]

    # Mouth corners.
    base[61] = [0.45, 0.70, 0.0]    # right corner (image-right == anatomical-left)
    base[291] = [0.55, 0.70, 0.0]   # left corner

    # Upper lip points.
    for idx in [0, 11, 12, 37, 39, 40, 267, 269, 270, 13]:
        base[idx] = [0.50 + (idx % 5 - 2) * 0.005, 0.68, 0.0]

    # Lower lip points.
    for idx in [14, 87, 178, 88, 95, 17, 84, 181, 91, 146, 317, 402, 318, 324, 308]:
        base[idx] = [0.50 + (idx % 7 - 3) * 0.006, 0.72, 0.0]

    # Chin/jaw lower contour.
    for idx in [152, 175, 199, 200, 17, 148, 176, 149, 150, 377, 400, 378, 379]:
        base[idx] = [0.50 + (idx % 9 - 4) * 0.01, 0.85, 0.0]

    # Cheeks.
    for idx, x in [(50, 0.38), (280, 0.62), (117, 0.40), (346, 0.60),
                   (205, 0.42), (425, 0.58), (187, 0.39), (411, 0.61)]:
        base[idx] = [x, 0.58, 0.0]

    return base


def _render_session(seed: int = 0) -> CaptureSession:
    """Build the synthetic CaptureSession described in the module docstring."""
    rng = np.random.default_rng(seed)
    base = _base_face()

    # Task plan: rest_seated 10s, facial_expression 10s, speech 8s.
    tasks = [
        TaskWindow("rest_seated", 0.0, 10.0),
        TaskWindow("facial_expression", 10.0, 20.0),
        TaskWindow("speech", 20.0, 28.0),
    ]
    duration = tasks[-1].end
    n_frames = int(duration * FPS)
    timestamps = np.arange(n_frames) / FPS

    # Precompute per-task masks.
    def in_task(t_arr: np.ndarray, name: str) -> np.ndarray:
        w = next(t for t in tasks if t.name == name)
        return (t_arr >= w.start) & (t_arr < w.end)

    rest_mask = in_task(timestamps, "rest_seated")
    expr_mask = in_task(timestamps, "facial_expression")
    speech_mask = in_task(timestamps, "speech")

    landmarks = np.tile(base, (n_frames, 1, 1))  # (N, 478, 3)

    # --- Expression task: reduced amplitude in chin/jaw and lower lip ---
    # Smile-like motion: mouth corners pull outward, cheeks rise.
    # Hypomimia: chin/lower-lip RoM is damped to ~30% of normal.
    expr_t = timestamps[expr_mask] - 10.0
    smile_envelope = 0.5 * (1 - np.cos(2 * np.pi * 0.4 * expr_t))  # 0.4 Hz, 2.5 s

    # Mouth corner motion — asymmetric: right corner (291) normal, left
    # corner (61) reduced. Convention follows signal_processing.py:
    # index 61 is labeled "left" (image-left), 291 is "right".
    NORMAL_AMP = 0.020
    LEFT_DAMPING = 0.3   # left mouth corner moves only 30% of right
    landmarks[expr_mask, 61, 0] += LEFT_DAMPING * NORMAL_AMP * smile_envelope   # left corner damped
    landmarks[expr_mask, 291, 0] += NORMAL_AMP * smile_envelope                 # right corner full

    # Lower lip — globally damped (hypomimia).
    LOWER_LIP_AMP = 0.005  # tiny vertical motion (would be ~0.015 in healthy)
    for idx in [14, 87, 178, 88, 95, 17, 84, 181, 91, 146, 317, 402, 318, 324, 308]:
        landmarks[expr_mask, idx, 1] += LOWER_LIP_AMP * smile_envelope

    # Chin/jaw — barely moves during expression (hypomimia).
    CHIN_EXPR_AMP = 0.003
    for idx in [152, 175, 199, 200, 17, 148, 176, 149, 150, 377, 400, 378, 379]:
        landmarks[expr_mask, idx, 1] += CHIN_EXPR_AMP * smile_envelope

    # Upper lip — moderate motion.
    UPPER_LIP_AMP = 0.008
    for idx in [0, 11, 12, 37, 39, 40, 267, 269, 270, 13]:
        landmarks[expr_mask, idx, 1] -= UPPER_LIP_AMP * smile_envelope

    # Cheeks — small rise.
    CHEEK_AMP = 0.006
    for idx in [50, 280, 117, 346, 205, 425, 187, 411]:
        landmarks[expr_mask, idx, 1] -= CHEEK_AMP * smile_envelope

    # --- Rest task: 4.5 Hz jaw tremor on chin region ---
    rest_t = timestamps[rest_mask] - 0.0
    JAW_TREMOR_FREQ = 4.5
    JAW_TREMOR_AMP = 0.003   # 0.3% of frame, ~3% of inter-ocular distance
    tremor = JAW_TREMOR_AMP * np.sin(2 * np.pi * JAW_TREMOR_FREQ * rest_t)
    for idx in [152, 175, 199, 200, 17, 148, 176, 149, 150, 377, 400, 378, 379]:
        landmarks[rest_mask, idx, 1] += tremor

    # --- Speech task: small mouth motion (we don't deeply use it) ---
    sp_t = timestamps[speech_mask] - 20.0
    speech_env = 0.005 * np.sin(2 * np.pi * 1.5 * sp_t)
    for idx in [14, 87, 178, 0, 11, 12, 13]:
        landmarks[speech_mask, idx, 1] += speech_env

    # Tiny global noise so per-frame deltas don't all collapse to zero.
    landmarks += rng.normal(0, 0.0002, landmarks.shape)

    # Reduced blink rate: ~8/min over 28s ≈ 4 blinks. Place them in rest+expr.
    blink_events = [3.0, 6.5, 13.5, 22.0]

    face = FaceTimeSeries(
        timestamps=timestamps,
        landmarks=landmarks,
        blink_events=blink_events,
    )

    return CaptureSession(
        session_id="demo-session-001",
        patient_id="DEMO-P-0001",
        duration_s=duration,
        capture_fps=FPS,
        tasks=tasks,
        face=face,
    )


def synthetic_session() -> CaptureSession:
    """Public factory used by tests, the Streamlit UI, and __main__."""
    return _render_session(seed=0)


def main() -> int:
    """End-to-end run against a live Ollama server.

    Use it as a smoke test:
        python -m parkinson_agent.run_demo
    """
    from .agent import run_screening_agent  # local import to keep deps lazy

    session = synthetic_session()
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    print(f"[run_demo] Running screening agent with model={model}...")
    result = run_screening_agent(session, model=model)
    print(result.report.model_dump_json(indent=2))
    print(f"[run_demo] iterations={result.iterations}, tool_calls={len(result.tool_calls)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
