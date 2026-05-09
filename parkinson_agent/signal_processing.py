"""Face feature extraction from OAK MediaPipe FaceMesh time-series.

Each function returns a JSON-serializable dict that the agent ingests as a
tool result. Numbers only — no thresholds applied here. Clinical reasoning
lives in the tool descriptions and the LLM, not in this layer.

Regional landmark groupings follow MediaPipe FaceMesh (478 points). The
weights reflect clinical priors supplied by the project owner: chin/jaw and
lower lip dominate (jaw tremor and lower-face hypomimia are the strongest
PD signals at the face); cheeks, eyelids and neck contribute less.

References:
- Bandini et al., 2017 — automated facial expression analysis in PD.
- Ricciardi et al., 2020 — hypomimia quantification with consumer cameras.
- Almeida et al., 2018 — jaw tremor as MDS-UPDRS 3.17 feature.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch

from .oak_adapter import CaptureSession, FaceTimeSeries

# ---------------------------------------------------------------------------
# Regional weights (clinical prior). Higher = more informative for screening.
# ---------------------------------------------------------------------------

REGION_WEIGHTS: dict[str, float] = {
    "chin_jaw":      1.0,   # HIGH
    "lower_lip":     1.0,   # HIGH
    "upper_lip":     0.6,   # MID
    "mouth_corners": 0.6,   # MID
    "cheeks":        0.4,   # LOW-MID
    "eyelids":       0.2,   # LOW
    "neck":          0.2,   # LOW (approximated from low jaw points; FaceMesh has no neck)
}


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices, grouped by anatomical region.
# Curated subsets — enough points per region for a stable centroid.
# ---------------------------------------------------------------------------

REGION_INDICES: dict[str, list[int]] = {
    "chin_jaw":      [152, 175, 199, 200, 17, 148, 176, 149, 150, 377, 400, 378, 379],
    "lower_lip":     [14, 87, 178, 88, 95, 17, 84, 181, 91, 146, 317, 402, 318, 324, 308],
    "upper_lip":     [0, 11, 12, 37, 39, 40, 267, 269, 270, 13],
    "mouth_corners": [61, 291],
    "cheeks":        [50, 280, 117, 346, 205, 425, 187, 411],
    "eyelids":       [159, 145, 386, 374, 158, 153, 385, 380],
    # FaceMesh has no true neck landmark — approximate with the lowest jawline
    # points. Clinically this is a weak proxy, hence the LOW weight.
    "neck":          [152, 175, 199, 200],
}

# Reference points used to normalize by inter-ocular distance (camera-distance
# invariance). 33 = right eye outer corner, 263 = left eye outer corner.
RIGHT_EYE_OUTER = 33
LEFT_EYE_OUTER = 263


def _safe_fs(timestamps: np.ndarray, fallback: float = 30.0) -> float:
    if len(timestamps) < 2:
        return fallback
    dt = np.diff(timestamps)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return fallback
    return float(1.0 / np.median(dt))


def _interocular_distance(landmarks: np.ndarray) -> float:
    """Per-frame mean distance between outer eye corners; used as a length unit."""
    if len(landmarks) == 0:
        return 1.0
    eye_l = landmarks[:, LEFT_EYE_OUTER, :2]
    eye_r = landmarks[:, RIGHT_EYE_OUTER, :2]
    d = np.linalg.norm(eye_l - eye_r, axis=1)
    d = d[d > 0]
    if len(d) == 0:
        return 1.0
    return float(np.median(d))


def _region_centroid(landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
    """Mean (x, y) per frame across the points of a region."""
    return landmarks[:, indices, :2].mean(axis=1)


def _region_dynamics(
    centroid: np.ndarray, timestamps: np.ndarray, ref_len: float
) -> dict:
    """Range-of-motion and velocity stats, normalized by inter-ocular distance."""
    if len(centroid) < 2:
        return {"range_of_motion": 0.0, "velocity_p95": 0.0, "velocity_median": 0.0}

    rom = float(np.linalg.norm(centroid.max(axis=0) - centroid.min(axis=0)) / ref_len)

    dt = np.diff(timestamps)
    dt = np.where(dt > 0, dt, np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0)
    speed = np.linalg.norm(np.diff(centroid, axis=0), axis=1) / dt / ref_len

    return {
        "range_of_motion": rom,
        "velocity_p95": float(np.percentile(speed, 95)),
        "velocity_median": float(np.median(speed)),
    }


# ---------------------------------------------------------------------------
# Public feature extractors. Each returns a JSON-friendly dict.
# ---------------------------------------------------------------------------

def regional_motion(face: FaceTimeSeries) -> dict:
    """Per-region range-of-motion and velocity, weighted by clinical priors.

    Reduced motion in the chin/jaw and lower-lip regions is the strongest
    face-only proxy for hypomimia / lower-face bradykinesia. The composite
    score is a weighted average where each region's contribution is its
    normalized range-of-motion times its clinical weight.
    """
    if face is None or len(face.timestamps) < 30 or face.coverage() < 0.5:
        return {
            "valid": False,
            "reason": "insufficient_face_data",
            "coverage": face.coverage() if face is not None else 0.0,
        }

    ref = _interocular_distance(face.landmarks)
    per_region: dict[str, dict] = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for region, indices in REGION_INDICES.items():
        centroid = _region_centroid(face.landmarks, indices)
        stats = _region_dynamics(centroid, face.timestamps, ref)
        per_region[region] = {
            "weight": REGION_WEIGHTS[region],
            **stats,
        }
        weighted_sum += REGION_WEIGHTS[region] * stats["range_of_motion"]
        weight_total += REGION_WEIGHTS[region]

    composite = weighted_sum / max(weight_total, 1e-9)

    return {
        "valid": True,
        "coverage": face.coverage(),
        "duration_s": float(face.timestamps[-1] - face.timestamps[0]),
        "per_region": per_region,
        "composite_expressivity_score": composite,
        "weights_applied": REGION_WEIGHTS,
    }


def jaw_tremor(face: FaceTimeSeries) -> dict:
    """Detect rhythmic 3–7 Hz oscillation of the chin (MDS-UPDRS 3.17, jaw).

    Jaw tremor at rest is a classic PD finding. We FFT the chin centroid
    (anchored to the inter-ocular midpoint to remove head sway) and look for
    spectral peaks in the parkinsonian band.
    """
    if face is None or len(face.timestamps) < 64 or face.coverage() < 0.5:
        return {"valid": False, "reason": "insufficient_face_data"}

    chin = _region_centroid(face.landmarks, REGION_INDICES["chin_jaw"])
    eye_mid = (
        face.landmarks[:, LEFT_EYE_OUTER, :2]
        + face.landmarks[:, RIGHT_EYE_OUTER, :2]
    ) / 2.0
    motion = chin - eye_mid          # remove rigid head motion
    motion = motion - motion.mean(axis=0)

    fs = _safe_fs(face.timestamps)
    nperseg = min(256, len(motion))

    pxx_total = None
    f = None
    for axis in range(motion.shape[1]):
        f_axis, pxx_axis = welch(motion[:, axis], fs=fs, nperseg=nperseg)
        if pxx_total is None:
            f, pxx_total = f_axis, pxx_axis.copy()
        else:
            pxx_total += pxx_axis
    pxx = pxx_total

    band = (f >= 3) & (f <= 7)
    if not band.any() or pxx[band].sum() == 0:
        return {"valid": False, "reason": "no_in_band_power"}

    peak_idx = int(np.argmax(pxx[band]))
    peak_freq = float(f[band][peak_idx])
    in_band = float(pxx[band].sum())
    total = float(pxx.sum())
    near = (f >= peak_freq - 0.5) & (f <= peak_freq + 0.5)
    peakedness = float(pxx[near].sum() / max(in_band, 1e-12))

    return {
        "valid": True,
        "dominant_frequency_hz": peak_freq,
        "in_parkinsonian_range_4_6hz": 4.0 <= peak_freq <= 6.0,
        "in_band_fraction_of_total": in_band / max(total, 1e-12),
        "spectral_peakedness": peakedness,
    }


def blink_rate(face: FaceTimeSeries) -> dict:
    """Blinks per minute. Reduced rate is supportive of hypomimia (3.2)."""
    if face is None or len(face.timestamps) < 30:
        return {"valid": False, "reason": "insufficient_face_data"}

    duration = float(face.timestamps[-1] - face.timestamps[0])
    if duration <= 0:
        return {"valid": False, "reason": "zero_duration"}

    rate = len(face.blink_events) / max(duration / 60.0, 1e-9)
    return {
        "valid": True,
        "duration_s": duration,
        "n_blinks": len(face.blink_events),
        "blink_rate_per_min": rate,
        "weight_eyelids": REGION_WEIGHTS["eyelids"],
    }


def mouth_corner_asymmetry(face: FaceTimeSeries) -> dict:
    """Compare left vs right mouth-corner range of motion.

    Asymmetric facial bradykinesia is consistent with unilateral PD onset.
    """
    if face is None or len(face.timestamps) < 30 or face.coverage() < 0.5:
        return {"valid": False, "reason": "insufficient_face_data"}

    ref = _interocular_distance(face.landmarks)
    left = face.landmarks[:, 61, :2]
    right = face.landmarks[:, 291, :2]

    rom_left = float(np.linalg.norm(left.max(axis=0) - left.min(axis=0)) / ref)
    rom_right = float(np.linalg.norm(right.max(axis=0) - right.min(axis=0)) / ref)
    asym = abs(rom_left - rom_right) / max(rom_left, rom_right, 1e-9)

    return {
        "valid": True,
        "rom_left": rom_left,
        "rom_right": rom_right,
        "asymmetry_ratio": asym,
        "less_mobile_side": "left" if rom_left < rom_right else "right",
        "weight_mouth_corners": REGION_WEIGHTS["mouth_corners"],
    }


def session_summary(session: CaptureSession) -> dict:
    """High-level metadata about what's available in this session."""
    return {
        "patient_id": session.patient_id,
        "session_id": session.session_id,
        "duration_s": session.duration_s,
        "capture_fps": session.capture_fps,
        "tasks": [
            {"name": t.name, "start_s": t.start, "end_s": t.end}
            for t in session.tasks
        ],
        "face_available": session.face is not None,
        "face_coverage": session.face.coverage() if session.face is not None else 0.0,
        "regional_weights": REGION_WEIGHTS,
    }


def face_for_task(session: CaptureSession, task_name: str | None) -> FaceTimeSeries | None:
    """Helper used by tools: face slice for a named task, or full session if None."""
    if task_name is None:
        return session.face
    return session.face_in_task(task_name)
