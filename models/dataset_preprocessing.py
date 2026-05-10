"""
preprocess.py — Parkinson Detection · Face Landmark Preprocessing Pipeline
===========================================================================

Adapted to the real dataset schema:

    patient_id   : str   — unique per person
    visit_id     : str   — datetime string e.g. '2026-05-10_013137' (one session)
    frame_idx    : int   — frame counter within the visit
    t            : float — time in seconds since session start
    sex          : str/int/float — 'M'/'F' or 0/1 (may be sparse → propagated)
    age          : float — years (may be sparse → propagated)
    label        : int   — 0=healthy, 1=Parkinson (may be sparse → propagated)
    state        : str   — visit-level tag e.g. 'visit' (kept for traceability only)
    x_{i}, y_{i}, z_{i} — sparse landmark set (only the indices captured, not all 468)

Key differences from generic template:
  - Sparse landmark columns  — only what's present in the file
  - Anchor = landmark 168 if present, else centroid proxy of available set
  - sex / age / label propagated forward+backward within each patient_id group
  - No inter-landmark count assumptions

OUTPUT  preprocessed/
    windows_X.npy        (N, W*F + K)   float32   flattened frame feats + FFT feats
    windows_meta.npy     (N, 3)         float32   [age, sex, label]
    windows_index.csv    traceability
    scaler.pkl           StandardScaler (apply at inference time)
    pca.pkl              PCA (if --pca N flag used)
    landmark_cols.json   exact landmark indices and anchor used
    feature_names.txt    one name per column of windows_X

USAGE:
    python preprocess.py --input data.csv  --output preprocessed/
    python preprocess.py --input data.parquet --output preprocessed/ --window 4 --stride 2 --pca 128

REQUIREMENTS:
    pip install pandas numpy scipy scikit-learn tqdm pyarrow
"""

import argparse
import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_LM_PREF = 168         # preferred anchor: nose bridge
TREMOR_BAND_HZ = (3.0, 7.0)  # Parkinson tremor range
LOWPASS_CUTOFF = 15.0        # Hz — sensor noise above this is suppressed

# Anatomical regions (MediaPipe indices).
# Only indices actually present in the file will contribute.
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
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Parkinson face-landmark preprocessing (sparse schema)")
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     default="preprocessed")
    p.add_argument("--window",     type=float, default=4.0,  help="Window length in seconds")
    p.add_argument("--stride",     type=float, default=2.0,  help="Stride in seconds (50%% overlap default)")
    p.add_argument("--fps",        type=float, default=30.0, help="Expected capture rate")
    p.add_argument("--max-gap-ms", type=float, default=200,  help="Max timestamp gap before splitting segment")
    p.add_argument("--pca",        type=int,   default=0,    help="PCA components after scaling (0 = off)")
    p.add_argument("--no-filter",  action="store_true",      help="Skip Butterworth low-pass filter")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & SCHEMA DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    print(f"[load] Reading {path.name} …")
    df = pd.read_parquet(path) if path.suffix in (".parquet", ".pq") else pd.read_csv(path, low_memory=False)
    print(f"[load] {len(df):,} rows  x  {len(df.columns)} columns")
    return df


def discover_landmark_cols(df: pd.DataFrame):
    """
    Detect sparse landmark columns in format x_{i}, y_{i}, z_{i}.
    Returns:
        lm_cols   dict  axis -> [col_name, ...]  ordered by landmark index
        lm_idxs   list  sorted landmark indices present with all 3 axes
    """
    pattern = re.compile(r"^(x|y|z)_(\d+)$")
    found = {}
    for col in df.columns:
        m = pattern.match(col)
        if m:
            ax, idx = m.group(1), int(m.group(2))
            found.setdefault(idx, set()).add(ax)

    complete = sorted(k for k, axes in found.items() if {"x", "y", "z"}.issubset(axes))
    if not complete:
        raise ValueError("No complete x_{i}/y_{i}/z_{i} triplets found. Check column names.")

    lm_cols = {
        "x": [f"x_{i}" for i in complete],
        "y": [f"y_{i}" for i in complete],
        "z": [f"z_{i}" for i in complete],
    }
    sample = complete[:6]
    print(f"[landmarks] {len(complete)} complete triplets found: {sample}{'...' if len(complete)>6 else ''}")
    return lm_cols, complete


def resolve_anchor(lm_idxs: list) -> tuple:
    """Return (anchor_landmark_id, anchor_position_in_array)."""
    if ANCHOR_LM_PREF in lm_idxs:
        pos = lm_idxs.index(ANCHOR_LM_PREF)
        print(f"[anchor] Using landmark {ANCHOR_LM_PREF} (nose bridge) at array position {pos}")
        return ANCHOR_LM_PREF, pos
    fallback = lm_idxs[len(lm_idxs) // 2]
    pos = lm_idxs.index(fallback)
    print(f"[anchor] Landmark {ANCHOR_LM_PREF} absent — using {fallback} as anchor (array pos {pos})")
    return fallback, pos


def build_region_map(lm_idxs: list) -> dict:
    """Map region_name -> list of array positions for landmarks in that region."""
    idx_to_pos = {lm: pos for pos, lm in enumerate(lm_idxs)}
    region_map = {}
    for name, ids in REGION_DEFS.items():
        positions = [idx_to_pos[i] for i in ids if i in idx_to_pos]
        if positions:
            region_map[name] = positions
    print(f"[regions] {len(region_map)} active regions covering "
          f"{sum(len(v) for v in region_map.values())} landmark positions")
    return region_map


# ─────────────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────────────
def encode_sex(series: pd.Series) -> pd.Series:
    # Try numeric first; fall back to string mapping
    try:
        return pd.to_numeric(series, errors="raise").astype(float)
    except (ValueError, TypeError):
        return series.astype(str).str.strip().str.upper().map(
            {"F": 0.0, "M": 1.0, "FEMALE": 0.0, "MALE": 1.0, "0": 0.0, "1": 1.0}
        )


def propagate_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Forward+backward fill sex/age/label within each patient_id."""
    for col in ("sex", "age", "label"):
        if col not in df.columns:
            print(f"[meta] '{col}' column missing — will be NaN")
            df[col] = np.nan
        else:
            df[col] = df.groupby("patient_id")[col].transform(lambda s: s.ffill().bfill())
    df["sex"] = encode_sex(df["sex"])
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("patient_id", "visit_id", "t"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found.")
    df = df.rename(columns={"t": "timestamp"})
    df = df.sort_values(["patient_id", "visit_id", "timestamp"]).reset_index(drop=True)

    if df["label"].notna().any():
        counts = df.groupby("patient_id")["label"].first().value_counts()
        print(f"[meta] Subjects per label: {dict(counts)}")
    else:
        print("[meta] WARNING: all label values are NaN.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if cutoff >= nyq:
        return data
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


def fft_band(signal_1d: np.ndarray, fs: float, band: tuple) -> tuple:
    n = len(signal_1d)
    if n < 8:
        return 0.0, 0.0, 0.0
    yf  = np.abs(rfft(signal_1d - signal_1d.mean())) ** 2
    xf  = rfftfreq(n, d=1.0 / fs)
    mask = (xf >= band[0]) & (xf <= band[1])
    if not mask.any() or yf.sum() == 0:
        return 0.0, 0.0, 0.0
    dom_f = float(xf[mask][np.argmax(yf[mask])])
    bp    = float(yf[mask].sum())
    ratio = float(bp / (yf.sum() + 1e-12))
    return dom_f, bp, ratio


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def frame_features(
    xyz: np.ndarray,       # (T, L, 3)
    anchor_pos: int,
    region_map: dict,
    dt: float,
    fps: float,
    apply_filter: bool,
) -> np.ndarray:
    """
    Per-frame features for one continuous segment.
    Returns (T, F) float32.

    Feature groups:
        relative position  (T, L*3)  — anchor-subtracted
        velocity           (T, L*3)  — 1st finite diff / dt
        acceleration       (T, L*3)  — 2nd finite diff / dt
        velocity magnitude (T, L)    — Euclidean norm of vel
        region mean velMag (T, R)    — mean per anatomical region
        region std  velMag (T, R)    — std  per anatomical region
    """
    T, L, _ = xyz.shape
    anchor   = xyz[:, anchor_pos, :]
    rel      = xyz - anchor[:, np.newaxis, :]

    if apply_filter and T > 12:
        for li in range(L):
            for ax in range(3):
                rel[:, li, ax] = butter_lowpass(rel[:, li, ax], LOWPASS_CUTOFF, fps)

    vel = np.zeros_like(rel)
    acc = np.zeros_like(rel)
    if T > 1:
        vel[1:] = np.diff(rel, axis=0) / max(dt, 1e-6)
    if T > 2:
        acc[2:] = np.diff(vel[1:], axis=0) / max(dt, 1e-6)

    vel_mag = np.linalg.norm(vel, axis=2)   # (T, L)

    reg_means = [vel_mag[:, pos].mean(axis=1, keepdims=True) for pos in region_map.values()]
    reg_stds  = [vel_mag[:, pos].std( axis=1, keepdims=True) for pos in region_map.values()]
    region_feats = np.concatenate(reg_means + reg_stds, axis=1)

    return np.concatenate([
        rel.reshape(T, -1),
        vel.reshape(T, -1),
        acc.reshape(T, -1),
        vel_mag,
        region_feats,
    ], axis=1).astype(np.float32)


def window_fft(
    xyz: np.ndarray,   # (W, L, 3)
    anchor_pos: int,
    region_map: dict,
    fps: float,
) -> np.ndarray:
    """FFT features over the full window, per region. Returns (R*3,) float32."""
    anchor  = xyz[:, anchor_pos, :]
    rel     = xyz - anchor[:, np.newaxis, :]
    if xyz.shape[0] > 1:
        vm = np.linalg.norm(np.diff(rel, axis=0), axis=2)   # (W-1, L)
    else:
        return np.zeros(len(region_map) * 3, dtype=np.float32)

    feats = []
    for positions in region_map.values():
        mean_vel = vm[:, positions].mean(axis=1)
        feats.extend(fft_band(mean_vel, fps, TREMOR_BAND_HZ))
    return np.array(feats, dtype=np.float32)


def slide(pf, xyz, anchor_pos, region_map, win_f, stride_f, fps):
    """Yield windows as dicts with frame_feats and fft_feats."""
    T, windows = pf.shape[0], []
    start = 0
    while start + win_f <= T:
        end = start + win_f
        windows.append({
            "frame_feats": pf[start:end],
            "fft_feats":   window_fft(xyz[start:end], anchor_pos, region_map, fps),
        })
        start += stride_f
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps      = args.fps
    win_f    = int(round(args.window * fps))
    stride_f = int(round(args.stride * fps))
    dt       = 1.0 / fps
    max_gap  = args.max_gap_ms / 1000.0

    print(f"\n[config] window={args.window}s ({win_f} frames) | stride={args.stride}s ({stride_f} frames) | fps={fps}\n")

    df = load_data(args.input)
    lm_cols, lm_idxs = discover_landmark_cols(df)
    anchor_id, anchor_pos = resolve_anchor(lm_idxs)
    region_map = build_region_map(lm_idxs)
    L, R = len(lm_idxs), len(region_map)

    print(f"\n[dims] L={L} landmarks | F/frame = {L*3*3 + L + 2*R} | FFT feats = {R*3}\n")

    with open(out_dir / "landmark_cols.json", "w") as f:
        json.dump({"landmark_indices": lm_idxs, "anchor": anchor_id}, f, indent=2)

    df = propagate_meta(df)
    df = validate(df)

    all_X, all_meta, index_rows = [], [], []

    for (patient, visit), grp in tqdm(df.groupby(["patient_id", "visit_id"]), desc="Sessions"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        age   = float(grp["age"].iloc[0])   if grp["age"].notna().any()   else np.nan
        sex   = float(grp["sex"].iloc[0])   if grp["sex"].notna().any()   else np.nan
        label = float(grp["label"].iloc[0]) if grp["label"].notna().any() else np.nan

        xyz = np.stack([
            grp[lm_cols["x"]].values.astype(np.float32),
            grp[lm_cols["y"]].values.astype(np.float32),
            grp[lm_cols["z"]].values.astype(np.float32),
        ], axis=2)   # (T, L, 3)

        T = xyz.shape[0]
        if T < win_f:
            tqdm.write(f"  [skip] {patient}/{visit}: {T} frames < window {win_f}")
            continue

        ts     = grp["timestamp"].values
        splits = np.where(np.diff(ts) > max_gap)[0] + 1
        bounds = [0] + list(splits) + [T]

        for si in range(len(bounds) - 1):
            s, e = bounds[si], bounds[si + 1]
            if e - s < win_f:
                continue
            xyz_s = xyz[s:e]
            ts_s  = ts[s:e]
            dt_s  = max(float(np.median(np.diff(ts_s))) if len(ts_s) > 1 else dt, 1e-6)

            pf   = frame_features(xyz_s, anchor_pos, region_map, dt_s, fps, not args.no_filter)
            wins = slide(pf, xyz_s, anchor_pos, region_map, win_f, stride_f, fps)

            for wi, win in enumerate(wins):
                combined = np.concatenate([win["frame_feats"].flatten(), win["fft_feats"]])
                all_X.append(combined)
                all_meta.append([age, sex, label])
                index_rows.append({
                    "patient_id": patient, "visit_id": visit,
                    "segment": si, "window": wi,
                    "label": int(label) if not np.isnan(label) else -1,
                    "age": age, "sex": int(sex) if not np.isnan(sex) else -1,
                })

    n = len(all_X)
    print(f"\n[done] {n:,} windows extracted")
    if n == 0:
        print("[ERROR] No windows produced. Check --window / --fps or data length.")
        return

    X    = np.stack(all_X).astype(np.float32)
    meta = np.array(all_meta, dtype=np.float32)

    bad = np.isnan(X).sum() + np.isinf(X).sum()
    if bad:
        print(f"[clean] Replacing {bad:,} NaN/Inf values with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[shape] X={X.shape}  meta={meta.shape}")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X).astype(np.float32)

    pca = None
    if args.pca > 0:
        nc   = min(args.pca, X_sc.shape[1], X_sc.shape[0] - 1)
        pca  = PCA(n_components=nc, random_state=42)
        X_sc = pca.fit_transform(X_sc).astype(np.float32)
        print(f"[pca] {nc} components | cumulative variance {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Feature names
    F_frame = L*3*3 + L + 2*R
    if pca is None:
        frame_names = (
            [f"rel_{i}_{ax}" for i in lm_idxs for ax in ("x","y","z")] +
            [f"vel_{i}_{ax}" for i in lm_idxs for ax in ("x","y","z")] +
            [f"acc_{i}_{ax}" for i in lm_idxs for ax in ("x","y","z")] +
            [f"vmag_{i}"     for i in lm_idxs] +
            [f"reg_{n}_mean" for n in region_map] +
            [f"reg_{n}_std"  for n in region_map]
        )
        names = (
            [f"t{f}_{fn}" for f in range(win_f) for fn in frame_names] +
            [f"fft_{n}_{k}" for n in region_map for k in ("domfreq","bandpow","ratio")]
        )
    else:
        names = [f"pca_{i}" for i in range(X_sc.shape[1])]
    names = names[:X_sc.shape[1]]

    np.save(out_dir / "windows_X.npy",    X_sc)
    np.save(out_dir / "windows_meta.npy", meta)
    pd.DataFrame(index_rows).to_csv(out_dir / "windows_index.csv", index=False)
    with open(out_dir / "scaler.pkl",        "wb") as f: pickle.dump(scaler, f)
    if pca:
        with open(out_dir / "pca.pkl",       "wb") as f: pickle.dump(pca, f)
    with open(out_dir / "feature_names.txt", "w")  as f: f.write("\n".join(names))

    labels = meta[:, 2]
    lab    = labels[~np.isnan(labels)].astype(int)
    print("\n" + "="*58)
    print("PREPROCESSING COMPLETE")
    print("="*58)
    print(f"  Total windows  : {X_sc.shape[0]:,}")
    print(f"  Feature dims   : {X_sc.shape[1]:,}")
    if len(lab):
        print(f"  PD windows     : {(lab==1).sum():,}")
        print(f"  Healthy windows: {(lab==0).sum():,}")
        print(f"  Class ratio    : {(lab==1).mean()*100:.1f}% PD")
    print(f"\n  Output: {out_dir.resolve()}/")
    for fn in ("windows_X.npy","windows_meta.npy","windows_index.csv",
               "scaler.pkl","landmark_cols.json","feature_names.txt"):
        print(f"    {fn}")
    if pca:
        print(f"    pca.pkl")
    print("="*58)


if __name__ == "__main__":
    run(parse_args())