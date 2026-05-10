"""Upstream model: OAK landmark CSV → data.json (the agent's input).

This is the boundary script the rest of the project keys off. It:
  1. Loads an OAK landmark CSV (per-frame x_{i}, y_{i}, z_{i} columns).
  2. Computes clinical features (regional motion, jaw tremor,
     mouth-corner asymmetry) using the project's clinical regional weights.
  3. Optionally runs the trained TCN classifier (`pd_tcn_weights.pth`)
     to get a PD probability — skipped when weights are absent.
  4. Writes a single JSON document matching `parkinson_agent.input_schema.
     KnowledgePayload` so the agent can ingest it.

Usage:
    python -m models.generate_knowledge \\
        --csv data/raw/embed_2026-05-10_013137.csv \\
        --out data/data.json

If a CSV contains multiple (patient_id, visit_id) sessions, pick one with
`--patient` / `--visit`; otherwise the first session is used.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Clinical priors (must stay in sync with parkinson_agent's expectations).
# ---------------------------------------------------------------------------

REGION_WEIGHTS: dict[str, float] = {
    "chin_jaw":      1.0,   # HIGH
    "lower_lip":     1.0,   # HIGH
    "upper_lip":     0.6,   # MID
    "mouth_corners": 0.6,   # MID
    "cheeks":        0.4,   # LOW-MID
}

REGION_INDICES_FULL: dict[str, list[int]] = {
    "chin_jaw": [
        152, 175, 199, 17, 148, 149, 150,
        377, 397, 365, 376,
        132, 138, 140, 169, 170, 171, 172, 215,
        213, 211, 32, 36, 192, 208, 207, 187, 205,
        50, 280, 425, 411, 432, 287, 416, 427,
        433, 435, 436, 262, 266, 288, 361, 352,
    ],
    "lower_lip": [
        14, 87, 178, 88, 95, 17, 84, 181, 91, 146,
        317, 402, 318, 324, 308, 405, 321, 375,
    ],
    "upper_lip": [0, 11, 12, 37, 39, 40, 267, 269, 270, 13],
    "mouth_corners": [61, 291],
    "cheeks": [50, 280, 205, 425, 187, 411, 207, 427],
}

NOSE_BRIDGE = 168
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

EXCLUDED_MODALITIES = [
    "blink_rate (no eyelid landmarks 159/145/386/374 in OAK sparse set)",
    "neck (not in MediaPipe FaceMesh)",
]


# ---------------------------------------------------------------------------
# CSV → in-memory session
# ---------------------------------------------------------------------------

_LM_PATTERN = re.compile(r"^(x|y|z)_(\d+)$")


@dataclass
class _Session:
    patient_id: str
    visit_id: str
    timestamps: np.ndarray
    landmarks: np.ndarray            # (T, L, 3)
    landmark_indices: list[int]
    age: float | None
    sex: str | None
    label: int | None
    state: str | None


def _discover_landmarks(df: pd.DataFrame):
    seen: dict[int, set] = {}
    for col in df.columns:
        m = _LM_PATTERN.match(col)
        if m:
            ax, idx = m.group(1), int(m.group(2))
            seen.setdefault(idx, set()).add(ax)
    indices = sorted(i for i, axes in seen.items() if {"x", "y", "z"}.issubset(axes))
    if not indices:
        raise ValueError("No x_{i}/y_{i}/z_{i} triplets found in CSV.")
    cols = {ax: [f"{ax}_{i}" for i in indices] for ax in ("x", "y", "z")}
    return cols, indices


def _first_non_null(s: pd.Series):
    nn = s.dropna()
    return nn.iloc[0] if len(nn) else None


def _load_session(csv_path: Path, patient: str | None, visit: str | None) -> _Session:
    df = pd.read_csv(csv_path, low_memory=False)
    cols, indices = _discover_landmarks(df)

    if patient is None:
        patient = str(df["patient_id"].iloc[0])
    if visit is None:
        visit = str(df[df["patient_id"] == patient]["visit_id"].iloc[0])

    grp = df[(df["patient_id"] == patient) & (df["visit_id"] == visit)]
    grp = grp.sort_values("t").reset_index(drop=True)
    if len(grp) == 0:
        raise ValueError(f"No rows for patient={patient!r} visit={visit!r}.")

    landmarks = np.stack(
        [
            grp[cols["x"]].values.astype(np.float32),
            grp[cols["y"]].values.astype(np.float32),
            grp[cols["z"]].values.astype(np.float32),
        ],
        axis=2,
    )
    age_v = _first_non_null(grp["age"]) if "age" in grp else None
    sex_v = _first_non_null(grp["sex"]) if "sex" in grp else None
    label_v = _first_non_null(grp["label"]) if "label" in grp else None
    state_v = _first_non_null(grp["state"]) if "state" in grp else None

    return _Session(
        patient_id=str(patient),
        visit_id=str(visit),
        timestamps=grp["t"].values.astype(np.float64),
        landmarks=landmarks,
        landmark_indices=indices,
        age=float(age_v) if age_v is not None and not pd.isna(age_v) else None,
        sex=str(sex_v) if sex_v is not None else None,
        label=int(label_v) if label_v is not None and not pd.isna(label_v) else None,
        state=str(state_v) if state_v is not None else None,
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _idx_pos(session: _Session) -> dict[int, int]:
    return {lm: pos for pos, lm in enumerate(session.landmark_indices)}


def _anchored_landmarks(session: _Session, ipos: dict[int, int]) -> np.ndarray:
    """Subtract the nose-bridge trajectory from every landmark.

    Removes rigid head motion (translation + most of the head-bob) so the
    remaining signal is facial expression. Falls back to per-frame
    landmark centroid when nose bridge is unavailable.
    """
    if NOSE_BRIDGE in ipos:
        anchor = session.landmarks[:, ipos[NOSE_BRIDGE], :2]
    else:
        anchor = session.landmarks[:, :, :2].mean(axis=1)
    return session.landmarks[:, :, :2] - anchor[:, np.newaxis, :]


def _face_size(session: _Session, ipos: dict[int, int], rel: np.ndarray) -> float:
    """Stable face-size reference: median nose-bridge → chin distance.

    Used as the unit that RoM values are reported in. Falls back to
    mouth-corner distance, then to a constant.
    """
    chin_id = next((i for i in (152, 175, 199) if i in ipos), None)
    if chin_id is not None and NOSE_BRIDGE in ipos:
        d = np.linalg.norm(rel[:, ipos[chin_id], :], axis=1)
        d = d[d > 0]
        if len(d):
            return float(np.median(d))
    if MOUTH_LEFT in ipos and MOUTH_RIGHT in ipos:
        L = session.landmarks[:, ipos[MOUTH_LEFT], :2]
        R = session.landmarks[:, ipos[MOUTH_RIGHT], :2]
        d = np.linalg.norm(L - R, axis=1)
        d = d[d > 0]
        if len(d):
            return float(np.median(d))
    return 0.1


def _region_centroid(landmarks: np.ndarray, positions: list[int]) -> np.ndarray:
    return landmarks[:, positions, :2].mean(axis=1)


def _region_dynamics(centroid: np.ndarray, ts: np.ndarray, ref: float) -> dict:
    if len(centroid) < 2:
        return {"range_of_motion": 0.0, "velocity_p95": 0.0, "velocity_median": 0.0}
    rom = float(np.linalg.norm(centroid.max(axis=0) - centroid.min(axis=0)) / max(ref, 1e-9))
    dt = np.diff(ts)
    valid = dt > 0
    if not valid.any():
        return {"range_of_motion": rom, "velocity_p95": 0.0, "velocity_median": 0.0}
    speed = (
        np.linalg.norm(np.diff(centroid, axis=0)[valid], axis=1)
        / dt[valid] / max(ref, 1e-9)
    )
    return {
        "range_of_motion": rom,
        "velocity_p95": float(np.percentile(speed, 95)),
        "velocity_median": float(np.median(speed)),
    }


def _fps(ts: np.ndarray) -> float:
    if len(ts) < 2:
        return 30.0
    dt = np.diff(ts)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else 30.0


def _regional_motion(session: _Session) -> dict:
    if len(session.timestamps) < 30:
        return {"valid": False, "reason": "insufficient_frames"}
    ipos = _idx_pos(session)
    rel = _anchored_landmarks(session, ipos)         # (T, L, 2), head-anchored
    face_size = _face_size(session, ipos, rel)
    fps = _fps(session.timestamps)
    duration = float(session.timestamps[-1] - session.timestamps[0])

    per_region: dict[str, dict] = {}
    weighted_sum = 0.0
    total_w = 0.0
    for region, ids in REGION_INDICES_FULL.items():
        positions = [ipos[i] for i in ids if i in ipos]
        entry: dict[str, Any] = {
            "weight": REGION_WEIGHTS[region],
            "available_landmarks": len(positions),
        }
        if not positions:
            entry["range_of_motion"] = 0.0
        else:
            cent = _region_centroid(rel, positions)  # head-anchored centroid
            stats = _region_dynamics(cent, session.timestamps, face_size)
            entry.update(stats)
            weighted_sum += REGION_WEIGHTS[region] * stats["range_of_motion"]
            total_w += REGION_WEIGHTS[region]
        per_region[region] = entry

    composite = weighted_sum / max(total_w, 1e-9)
    return {
        "valid": True,
        "duration_s": duration,
        "fps": fps,
        "n_frames": int(len(session.timestamps)),
        "per_region": per_region,
        "composite_expressivity_score": composite,
    }


def _jaw_tremor(session: _Session) -> dict:
    if len(session.timestamps) < 64:
        return {"valid": False, "reason": "insufficient_frames"}
    ipos = _idx_pos(session)
    chin_pos = [ipos[i] for i in REGION_INDICES_FULL["chin_jaw"] if i in ipos]
    if not chin_pos:
        return {"valid": False, "reason": "no_chin_landmarks"}
    chin = _region_centroid(session.landmarks, chin_pos)

    if NOSE_BRIDGE in ipos:
        anchor = session.landmarks[:, ipos[NOSE_BRIDGE], :2]
        anchor_used = "nose_bridge_168"
    elif MOUTH_LEFT in ipos and MOUTH_RIGHT in ipos:
        anchor = (
            session.landmarks[:, ipos[MOUTH_LEFT], :2]
            + session.landmarks[:, ipos[MOUTH_RIGHT], :2]
        ) / 2.0
        anchor_used = "mouth_corner_midpoint"
    else:
        anchor = np.zeros_like(chin)
        anchor_used = "none"

    motion = chin - anchor
    motion = motion - motion.mean(axis=0)
    fs = _fps(session.timestamps)
    nperseg = min(256, len(motion))
    pxx_total = None
    f = None
    for ax in range(motion.shape[1]):
        fa, pa = welch(motion[:, ax], fs=fs, nperseg=nperseg)
        if pxx_total is None:
            f, pxx_total = fa, pa.copy()
        else:
            pxx_total += pa
    pxx = pxx_total
    band = (f >= 3) & (f <= 7)
    if not band.any() or pxx[band].sum() == 0:
        return {"valid": False, "reason": "no_in_band_power", "fps": fs}

    peak_freq = float(f[band][int(np.argmax(pxx[band]))])
    in_band = float(pxx[band].sum())
    total = float(pxx.sum())
    near = (f >= peak_freq - 0.5) & (f <= peak_freq + 0.5)
    return {
        "valid": True,
        "anchor_used": anchor_used,
        "fps": fs,
        "dominant_frequency_hz": peak_freq,
        "in_parkinsonian_range_4_6hz": 4.0 <= peak_freq <= 6.0,
        "in_band_fraction_of_total": in_band / max(total, 1e-12),
        "spectral_peakedness": float(pxx[near].sum() / max(in_band, 1e-12)),
    }


def _mouth_asymmetry(session: _Session) -> dict:
    if len(session.timestamps) < 30:
        return {"valid": False, "reason": "insufficient_frames"}
    ipos = _idx_pos(session)
    if MOUTH_LEFT not in ipos or MOUTH_RIGHT not in ipos:
        return {"valid": False, "reason": "missing_mouth_corner_landmarks"}

    rel = _anchored_landmarks(session, ipos)
    face_size = _face_size(session, ipos, rel)
    L = rel[:, ipos[MOUTH_LEFT], :]
    R = rel[:, ipos[MOUTH_RIGHT], :]
    rom_l = float(np.linalg.norm(L.max(axis=0) - L.min(axis=0)) / max(face_size, 1e-9))
    rom_r = float(np.linalg.norm(R.max(axis=0) - R.min(axis=0)) / max(face_size, 1e-9))
    asym = abs(rom_l - rom_r) / max(rom_l, rom_r, 1e-9)
    return {
        "valid": True,
        "rom_left": rom_l,
        "rom_right": rom_r,
        "asymmetry_ratio": asym,
        "less_mobile_side": "left" if rom_l < rom_r else "right",
        "weight_mouth_corners": REGION_WEIGHTS["mouth_corners"],
    }


# ---------------------------------------------------------------------------
# Optional: TCN classifier inference. Skipped when weights are absent.
# ---------------------------------------------------------------------------

def _maybe_run_tcn(
    session: _Session, weights_path: Path | None
) -> dict | None:
    """Run the TCN model on the session windows. Returns None if unavailable.

    This is a stub: actually plugging in `models/model.py` requires running
    the same preprocessing the model was trained with (windows of 4s of
    feature vectors, FFT band-power per region) and matching dims. Hook
    `models/dataset_preprocessing.py + models/model.py + weights.pth` here
    when you have a trained checkpoint.
    """
    if weights_path is None or not weights_path.exists():
        return None
    return {
        "model_name": "SmallPDTCN",
        "version": "0.1",
        "pd_probability": None,
        "n_windows_analyzed": None,
        "notes": (
            "Weights file found but inference not wired yet. "
            "See models/generate_knowledge.py:_maybe_run_tcn — plug in "
            "preprocessing + model.forward + window aggregation here."
        ),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _build_payload(session: _Session, weights_path: Path | None) -> dict:
    fps = _fps(session.timestamps)
    duration = float(session.timestamps[-1] - session.timestamps[0]) if len(session.timestamps) > 1 else 0.0
    payload: dict[str, Any] = {
        "patient_id": session.patient_id,
        "session_id": session.visit_id,
        "captured_at": None,
        "duration_s": duration,
        "n_frames": int(len(session.timestamps)),
        "fps": fps,
        "metadata": {
            "age": session.age,
            "sex": session.sex,
            "ground_truth_label": session.label,
        },
        "regional_weights": REGION_WEIGHTS,
        "clinical_features": {
            "regional_motion": _regional_motion(session),
            "jaw_tremor": _jaw_tremor(session),
            "mouth_asymmetry": _mouth_asymmetry(session),
        },
        "model_inference": _maybe_run_tcn(session, weights_path),
        "quality": {
            "face_coverage": float(np.mean(np.any(session.landmarks != 0, axis=(1, 2)))),
            "n_landmarks_available": len(session.landmark_indices),
            "missing_modalities": EXCLUDED_MODALITIES,
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, default=Path("data/raw/embed_2026-05-10_013137.csv"))
    p.add_argument("--out", type=Path, default=Path("data/data.json"))
    p.add_argument("--patient", default=None)
    p.add_argument("--visit", default=None)
    p.add_argument("--weights", type=Path, default=Path("models/pd_tcn_weights.pth"),
                   help="Optional TCN weights. Inference is skipped if missing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"[generate_knowledge] CSV not found: {args.csv}", file=sys.stderr)
        return 1

    session = _load_session(args.csv, args.patient, args.visit)
    weights = args.weights if args.weights and args.weights.exists() else None
    payload = _build_payload(session, weights)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)

    has_model = payload["model_inference"] is not None
    print(
        f"[generate_knowledge] wrote {args.out}  "
        f"(patient={session.patient_id} visit={session.visit_id} "
        f"frames={len(session.timestamps)} model_inference={'yes' if has_model else 'no'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
