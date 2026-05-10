"""
inference.py — Parkinson Detection · Real-Time Inference
=========================================================

Reads a 4-second window JSON produced by the camera pipeline.

REAL INPUT FORMAT (array of frame-rows, landmarks as flat columns):
    [
        {
            "patient_id": "lisanew",
            "visit_id":   "2026-05-10_035241",
            "timestamp_ms": 1778377969427,
            "timestamp_iso": "2026-05-10T03:52:49.427",
            "sex": 1,
            "age": 65,
            "label": 1,
            "state": "visit",
            "x_0": 0.43261, "y_0": 0.6104, "z_0": 0.0,
            "x_1": 0.43426, "y_1": 0.59795, "z_1": 0.0,
            ...
        },
        ...  (one dict per frame, ~120 rows for a 4-second / 30fps window)
    ]

OUTPUT JSON:
    {
        "patient_id": "lisanew",
        "visit_id": "2026-05-10_035241",
        "window_start_ms": ...,
        "window_end_ms": ...,
        "ground_truth": {"label": 1, "sex": 1, "age": 65},
        "prediction": {
            "label": "Parkinson",
            "confidence_pct": 83.4,
            "pd_probability": 0.834,
            "threshold": 0.5
        },
        "feature_importance": { ... },
        "preprocessing": { ... },
        "raw_features": { ... },
        "inference_time_ms": 12.3,
        "model_path": "pd_tcn_model_acc_XX.onnx"
    }

Usage:
    python inference.py \
        --model    pd_tcn_model_acc_78.50.onnx \
        --scaler   preprocessed/scaler.pkl \
        --features preprocessed/feature_names.txt \
        --input    window_capture.json \
        --output   result.json

    # Batch: point --input at a directory of .json files
    python inference.py --model ... --scaler ... --features ... --input captures/
"""

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARNING] onnxruntime not found. Install with: pip install onnxruntime")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match preprocess.py / train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
FPS_DEFAULT    = 30
WINDOW_SEC     = 4.0
WINDOW_FRAMES  = int(FPS_DEFAULT * WINDOW_SEC)   # 120
ANCHOR_LM_PREF = 168
TREMOR_BAND_HZ = (3.0, 7.0)
LOWPASS_CUTOFF = 15.0

REGION_DEFS = {
    "left_eye":      [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    "right_eye":     [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398],
    "lips":          [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,
                      14,87,178,88,95,185,40,39,37,0,267,269,270,409],
    "left_eyebrow":  [276,283,282,295,285,300,293,334,296,336],
    "right_eyebrow": [46,53,52,65,55,70,63,105,66,107],
    "nose":          [1,2,3,4,5,6,19,20,44,45,48,51,94,97,98,115,131,134,
                      174,195,197,198,209,236,240,242,250],
    "jaw":           [0,17,57,61,84,91,146,152,172,175,176,178,181,
                      199,200,208,210,211,214,262,271,397,400,421,428,430,431],
    "forehead":      [10,67,103,109,338,297,332,333,334,296,336,9,8],
}


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING  (mirrors preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────

def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if cutoff >= nyq:
        return data
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


def fft_band(signal_1d: np.ndarray, fs: float, band: tuple) -> dict:
    n = len(signal_1d)
    if n < 8:
        return {"dominant_freq_hz": 0.0, "band_power": 0.0, "band_ratio": 0.0}
    yf   = np.abs(rfft(signal_1d - signal_1d.mean())) ** 2
    xf   = rfftfreq(n, d=1.0 / fs)
    mask = (xf >= band[0]) & (xf <= band[1])
    if not mask.any() or yf.sum() == 0:
        return {"dominant_freq_hz": 0.0, "band_power": 0.0, "band_ratio": 0.0}
    dom_f = float(xf[mask][np.argmax(yf[mask])])
    bp    = float(yf[mask].sum())
    ratio = float(bp / (yf.sum() + 1e-12))
    return {"dominant_freq_hz": dom_f, "band_power": bp, "band_ratio": ratio}


# ─────────────────────────────────────────────────────────────────────────────
# INPUT PARSER  — handles the real flat-row format
# ─────────────────────────────────────────────────────────────────────────────

def parse_window_json(json_path: str) -> dict:
    """
    Parses the real camera JSON: a list of frame-row dicts where each landmark
    appears as flat columns x_{i}, y_{i}, z_{i}.
    """
    with open(json_path, "r") as f:
        rows = json.load(f)

    if not rows:
        raise ValueError(f"Empty JSON: {json_path}")

    # detect which landmark indices are actually present
    lm_pattern = re.compile(r"^x_(\d+)$")
    lm_idxs = sorted(
        int(m.group(1))
        for key in rows[0].keys()
        for m in [lm_pattern.match(key)]
        if m
    )
    if not lm_idxs:
        raise ValueError("No x_{i} landmark columns found in JSON rows.")

    L     = max(lm_idxs) + 1
    T     = len(rows)
    xyz   = np.zeros((T, L, 3), dtype=np.float32)
    ts_ms = np.zeros(T, dtype=np.float64)

    for fi, row in enumerate(rows):
        if "timestamp_ms" in row:
            ts_ms[fi] = float(row["timestamp_ms"])
        elif "timestamp_iso" in row:
            from datetime import datetime
            dt = datetime.fromisoformat(row["timestamp_iso"])
            ts_ms[fi] = dt.timestamp() * 1000.0
        else:
            ts_ms[fi] = fi * (1000.0 / FPS_DEFAULT)

        for idx in lm_idxs:
            x = row.get(f"x_{idx}", 0.0) or 0.0
            y = row.get(f"y_{idx}", 0.0) or 0.0
            z = row.get(f"z_{idx}", 0.0) or 0.0
            xyz[fi, idx, 0] = float(x)
            xyz[fi, idx, 1] = float(y)
            xyz[fi, idx, 2] = float(z)

    # estimate FPS from actual timestamps
    if T > 1:
        diffs = np.diff(ts_ms)
        median_gap_ms = float(np.median(diffs[diffs > 0])) if (diffs > 0).any() else (1000.0 / FPS_DEFAULT)
        fps = round(1000.0 / median_gap_ms, 1)
    else:
        fps = FPS_DEFAULT

    first  = rows[0]
    label  = first.get("label", float("nan"))
    sex    = first.get("sex",   float("nan"))
    age    = first.get("age",   float("nan"))

    def safe_float(v):
        try: return float(v)
        except: return float("nan")

    return {
        "patient_id":         str(first.get("patient_id", "unknown")),
        "visit_id":           str(first.get("visit_id",   "unknown")),
        "fps":                fps,
        "timestamp_start_ms": float(ts_ms[0]),
        "timestamp_end_ms":   float(ts_ms[-1]),
        "xyz":                xyz,
        "lm_idxs":            lm_idxs,
        "T":                  T,
        "L":                  L,
        "label":              safe_float(label),
        "sex":                safe_float(sex),
        "age":                safe_float(age),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (mirrors preprocess.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(xyz: np.ndarray, lm_idxs: list, fps: float, apply_filter: bool = True):
    T, L, _ = xyz.shape

    anchor_pos = ANCHOR_LM_PREF if ANCHOR_LM_PREF in lm_idxs else lm_idxs[len(lm_idxs) // 2]

    lm_set = set(lm_idxs)
    region_map = {
        name: [i for i in ids if i in lm_set]
        for name, ids in REGION_DEFS.items()
        if any(i in lm_set for i in ids)
    }

    anchor = xyz[:, anchor_pos, :]
    rel    = xyz - anchor[:, np.newaxis, :]

    if apply_filter and T > 12:
        for li in lm_idxs:
            for ax in range(3):
                rel[:, li, ax] = butter_lowpass(rel[:, li, ax], LOWPASS_CUTOFF, fps)

    dt  = 1.0 / fps
    vel = np.zeros_like(rel)
    acc = np.zeros_like(rel)
    if T > 1:
        vel[1:] = np.diff(rel, axis=0) / dt
    if T > 2:
        acc[2:] = np.diff(vel[1:], axis=0) / dt

    vel_mag = np.linalg.norm(vel, axis=2)

    reg_means = [vel_mag[:, pos].mean(axis=1, keepdims=True) for pos in region_map.values()]
    reg_stds  = [vel_mag[:, pos].std( axis=1, keepdims=True) for pos in region_map.values()]
    region_feats = np.concatenate(reg_means + reg_stds, axis=1)

    frame_feats = np.concatenate([
        rel.reshape(T, -1),
        vel.reshape(T, -1),
        acc.reshape(T, -1),
        vel_mag,
        region_feats,
    ], axis=1).astype(np.float32)

    fft_values, fft_rich = [], {}
    for name, positions in region_map.items():
        mean_vel  = vel_mag[1:, positions].mean(axis=1) if T > 1 else np.zeros(1)
        band_info = fft_band(mean_vel, fps, TREMOR_BAND_HZ)
        fft_values.extend([band_info["dominant_freq_hz"],
                           band_info["band_power"],
                           band_info["band_ratio"]])
        fft_rich[name] = band_info

    fft_feats = np.array(fft_values, dtype=np.float32)

    rich_stats = {
        "velocity_per_region": {
            name: {
                "mean_velocity": float(vel_mag[:, pos].mean()),
                "std_velocity":  float(vel_mag[:, pos].std()),
                "max_velocity":  float(vel_mag[:, pos].max()),
            }
            for name, pos in region_map.items()
        },
        "fft_per_region": fft_rich,
        "anchor_landmark": anchor_pos,
        "num_landmarks":   len(lm_idxs),
        "num_regions":     len(region_map),
    }

    return frame_feats, fft_feats, rich_stats, anchor_pos, region_map


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN-TIME BLINDFOLD  (mirrors train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def apply_blindfold(X_temporal: np.ndarray, feature_names: list) -> tuple:
    temporal_names = [n for n in feature_names if not n.startswith("fft_")]
    frame_0_names  = [n for n in temporal_names if n.startswith("t0_")]
    keep_indices   = [i for i, name in enumerate(frame_0_names) if "_rel_" not in name]
    return X_temporal[:, keep_indices, :], len(keep_indices)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX
# ─────────────────────────────────────────────────────────────────────────────

def load_onnx_session(model_path: str):
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"[model] Loaded {model_path}")
    return sess


def run_onnx(sess, X_temporal: np.ndarray, X_fft: np.ndarray) -> float:
    outputs = sess.run(
        ["pd_probability"],
        {"temporal_input": X_temporal.astype(np.float32),
         "fft_input":      X_fft.astype(np.float32)}
    )
    return float(outputs[0][0][0])


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_importance(rich_stats: dict, pd_prob: float) -> dict:
    fft_data = rich_stats.get("fft_per_region", {})
    vel_data = rich_stats.get("velocity_per_region", {})

    ranked_tremor = sorted(fft_data.items(), key=lambda x: x[1].get("band_ratio", 0), reverse=True)
    top_tremor = []
    for name, info in ranked_tremor[:4]:
        ratio = info["band_ratio"]
        interp = (f"Strong 3–7 Hz tremor in {name} — primary PD indicator" if ratio > 0.4
                  else f"Moderate tremor signal in {name}" if ratio > 0.2
                  else f"Low tremor in {name}")
        top_tremor.append({
            "region": name,
            "tremor_band_ratio": round(ratio, 4),
            "dominant_freq_hz":  round(info["dominant_freq_hz"], 2),
            "band_power":        round(info["band_power"], 6),
            "interpretation":    interp,
        })

    vel_ranked = sorted(vel_data.items(), key=lambda x: x[1].get("std_velocity", 0), reverse=True)
    top_velocity = [
        {"region": name,
         "mean_velocity": round(info["mean_velocity"], 6),
         "std_velocity":  round(info["std_velocity"],  6),
         "max_velocity":  round(info["max_velocity"],  6)}
        for name, info in vel_ranked[:4]
    ]

    top_r = ranked_tremor[0][0] if ranked_tremor else "N/A"
    top_v = ranked_tremor[0][1].get("band_ratio", 0) if ranked_tremor else 0
    if pd_prob >= 0.7:
        note = (f"High PD probability driven by elevated tremor-band energy "
                f"(top region: {top_r}, ratio={top_v:.3f}). "
                f"Micro-tremors in the 3–7 Hz range are a hallmark of Parkinson's.")
    elif pd_prob >= 0.5:
        note = (f"Borderline PD signal. Tremor activity in {top_r} is present "
                f"but not dominant. Consider a longer observation window.")
    else:
        note = (f"Low PD probability. Facial movement patterns appear within healthy "
                f"norms. Tremor-band ratios are low across all regions.")

    return {"top_tremor_regions": top_tremor, "top_velocity_regions": top_velocity, "decision_note": note}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def infer_one(json_path, onnx_session, scaler, feature_names,
              threshold=0.5, apply_filter=True, model_name="unknown"):

    t0 = time.perf_counter()

    window  = parse_window_json(json_path)
    xyz     = window["xyz"]
    lm_idxs = window["lm_idxs"]
    fps     = window["fps"]
    T_orig  = window["T"]

    # Pad or trim to exactly WINDOW_FRAMES
    if T_orig < WINDOW_FRAMES:
        print(f"  [warn] {T_orig} frames received, need {WINDOW_FRAMES}. Zero-padding.")
        pad = np.zeros((WINDOW_FRAMES - T_orig, xyz.shape[1], 3), dtype=np.float32)
        xyz = np.concatenate([xyz, pad], axis=0)
    elif T_orig > WINDOW_FRAMES:
        xyz = xyz[:WINDOW_FRAMES]

    frame_feats, fft_feats, rich_stats, anchor_pos, region_map = extract_features(
        xyz, lm_idxs, fps, apply_filter=apply_filter
    )

    num_frame_features_raw = frame_feats.shape[1]
    num_fft_features       = fft_feats.shape[0]

    flat  = np.concatenate([frame_feats.flatten(), fft_feats]).reshape(1, -1)

    scaler_applied = False
    if scaler is not None:
        try:
            flat = scaler.transform(flat).astype(np.float32)
            scaler_applied = True
        except Exception as e:
            print(f"  [warn] Scaler failed ({e})")
            flat = flat.astype(np.float32)
    else:
        flat = flat.astype(np.float32)

    split_idx  = WINDOW_FRAMES * num_frame_features_raw
    X_t        = flat[:, :split_idx].reshape(1, WINDOW_FRAMES, num_frame_features_raw)
    X_t        = np.transpose(X_t, (0, 2, 1))    # (1, F, T)
    X_f        = flat[:, split_idx:]

    X_t_blind, num_frame_features_blind = apply_blindfold(X_t, feature_names)

    pd_prob    = run_onnx(onnx_session, X_t_blind, X_f)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label      = "Parkinson" if pd_prob >= threshold else "Healthy"
    confidence = pd_prob if pd_prob >= threshold else 1.0 - pd_prob

    def _int_or_none(v):
        return int(v) if not np.isnan(v) else None

    return {
        "patient_id":      window["patient_id"],
        "visit_id":        window["visit_id"],
        "window_start_ms": window["timestamp_start_ms"],
        "window_end_ms":   window["timestamp_end_ms"],
        "ground_truth": {
            "label": _int_or_none(window["label"]),
            "sex":   _int_or_none(window["sex"]),
            "age":   _int_or_none(window["age"]),
        },
        "prediction": {
            "label":          label,
            "confidence_pct": round(confidence * 100, 2),
            "pd_probability": round(pd_prob, 4),
            "threshold":      threshold,
        },
        "feature_importance": build_feature_importance(rich_stats, pd_prob),
        "preprocessing": {
            "frames_in_file":     T_orig,
            "frames_used":        WINDOW_FRAMES,
            "landmarks_in_file":  len(lm_idxs),
            "estimated_fps":      fps,
            "anchor_landmark":    anchor_pos,
            "filter_applied":     apply_filter,
            "num_frame_features": num_frame_features_blind,
            "num_fft_features":   num_fft_features,
            "scaler_applied":     scaler_applied,
        },
        "raw_features": {
            "fft_per_region": {
                k: {kk: round(vv, 6) for kk, vv in v.items()}
                for k, v in rich_stats["fft_per_region"].items()
            },
            "velocity_per_region": {
                k: {kk: round(vv, 6) for kk, vv in v.items()}
                for k, v in rich_stats["velocity_per_region"].items()
            },
        },
        "inference_time_ms": round(elapsed_ms, 2),
        "model_path":        model_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PD TCN Real-Time Inference")
    p.add_argument("--model",     required=True)
    p.add_argument("--scaler",    required=True)
    p.add_argument("--features",  required=True)
    p.add_argument("--input",     required=True)
    p.add_argument("--output",    default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-filter", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    session    = load_onnx_session(args.model)
    model_name = Path(args.model).name

    scaler = None
    if Path(args.scaler).exists():
        with open(args.scaler, "rb") as f:
            scaler = pickle.load(f)
        print(f"[scaler] Loaded {args.scaler}")
    else:
        print(f"[warn] Scaler not found at {args.scaler}")

    with open(args.features) as f:
        feature_names = f.read().splitlines()
    print(f"[features] {len(feature_names)} names loaded")

    input_path = Path(args.input)
    json_files = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    print(f"[batch] {len(json_files)} file(s)")

    results = []
    for jf in json_files:
        print(f"\n[infer] {jf.name}")
        try:
            r = infer_one(str(jf), session, scaler, feature_names,
                          threshold=args.threshold,
                          apply_filter=not args.no_filter,
                          model_name=model_name)
            results.append(r)
            pred = r["prediction"]
            print(f"  ➜  {pred['label']}  |  PD prob: {pred['pd_probability']:.3f}  |  "
                  f"Confidence: {pred['confidence_pct']:.1f}%  |  {r['inference_time_ms']:.1f} ms")
            top = r["feature_importance"]["top_tremor_regions"]
            if top:
                print(f"  Top tremor: {top[0]['region']} "
                      f"(ratio={top[0]['tremor_band_ratio']:.3f}, {top[0]['dominant_freq_hz']} Hz)")
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"input": str(jf), "error": str(e)})

    out = (Path(args.output) if args.output
           else (input_path / "batch_results.json" if input_path.is_dir()
                 else input_path.with_name(input_path.stem + "_result.json")))
    data = results[0] if len(results) == 1 else {"results": results, "count": len(results)}
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Result saved to {out}")


if __name__ == "__main__":
    main()