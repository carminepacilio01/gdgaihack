"""VisitWriter — accumulates per-frame metrics emitted by BiomarkerExtractor and
persists a single visit summary JSON when the recording duration is reached.

JSON schema (data/visits/<patient_id>/visit_<visit_id>.json):
{
  "patient_id": "demo_patient",
  "visit_id": "2026-05-09",
  "recorded_at": "2026-05-09T11:30:42",
  "duration_s": 60,
  "n_frames": 1812,
  "metrics": {
    "blink_rate":        { "mean": 11.4, "std": 2.1, "p95": 14.1, "trace": [...] },
    "smile_amplitude_mm":{ "mean": 38.5, ... },
    "brow_amplitude_mm": { "mean": 4.2,  ... },
    "asymmetry":         { "mean": 0.07, ... },
    "gaze_x_std":        { "mean": 0.04, ... },
    "gaze_y_std":        { "mean": 0.03, ... },
    "face_distance_mm":  { "mean": 540,  ... }
  },
  "ranges": { ... clinical normal ranges from biomarkers.py ... },
  "disclaimer": "Screening tool only — not a medical diagnostic device."
}
"""
import csv
import json
import time
import os
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import depthai as dai

from .biomarkers import LOWER_FACE_FIXED_INDICES


DATA_DIR = Path(os.environ.get("NEUROSIGHT_DATA_DIR", "data/visits"))
PATIENT_INDEX_PATH = Path(os.environ.get(
    "NEUROSIGHT_PATIENT_INDEX",
    str(DATA_DIR.parent / "patient_index.json"),
))
MASTER_DATASET_PATH = Path(os.environ.get(
    "NEUROSIGHT_MASTER_DATASET",
    str(DATA_DIR.parent / "master_dataset.csv"),
))

# Embed / master CSV columns: ML training-ready, one row per frame.
# `patient_num` is a stable monotonic integer registered in patient_index.json.
# `timestamp_iso` + `timestamp_ms` together replace the previous (visit_id, t)
# pair as the precise wall-clock identifier for each frame.
EMBED_META_COLS = [
    "patient_num", "patient_id", "visit_id",
    "timestamp_iso", "timestamp_ms",
    "sex", "age", "label", "state",
]
EMBED_KP_COLS = []
for _i in LOWER_FACE_FIXED_INDICES:
    EMBED_KP_COLS.extend([f"x_{_i}", f"y_{_i}", f"z_{_i}"])
EMBED_HEADER = EMBED_META_COLS + EMBED_KP_COLS


def _load_patient_index() -> dict:
    try:
        with open(PATIENT_INDEX_PATH) as f:
            idx = json.load(f)
        if "next_id" not in idx or "by_name" not in idx:
            raise ValueError("malformed index")
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        idx = {"next_id": 1, "by_name": {}, "labels": {}}
    idx.setdefault("labels", {})
    return idx


def _save_patient_index(idx: dict) -> None:
    PATIENT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATIENT_INDEX_PATH, "w") as f:
        json.dump(idx, f, indent=2)


def patient_num_for(alias: str) -> int:
    """Return monotonic numeric ID for a patient string alias.
    Registers the alias on first use. Persistent in patient_index.json."""
    if not alias:
        alias = "_unknown"
    idx = _load_patient_index()
    by_name = idx["by_name"]
    if alias in by_name:
        return int(by_name[alias])
    num = int(idx["next_id"])
    by_name[alias] = num
    idx["next_id"] = num + 1
    _save_patient_index(idx)
    return num


def pd_label_for(patient_num: int):
    """Return preset PD label (0 or 1) for a numeric patient ID, or None
    if no preset is registered. Used to pre-fill the record modal."""
    idx = _load_patient_index()
    return idx.get("labels", {}).get(str(int(patient_num)))

# Numeric keys persisted as columns in the per-frame CSV. Order = column order.
CSV_COLUMNS = [
    "t", "frame_idx",
    "ear", "ear_threshold", "blink_rate",
    "smile_mm", "smile_amplitude_mm",
    "brow_mm", "brow_amplitude_mm",
    "asymmetry", "asymmetry_std",
    "gaze_dx", "gaze_dy", "gaze_x_std", "gaze_y_std",
    "face_distance_mm",
    "tremor_chin_power_4_6hz", "tremor_chin_dominant_hz",
    "tremor_chin_motion_power", "tremor_chin_lock_in",
    "tremor_chin_pd_likelihood",
    "tremor_lip_power_4_6hz", "tremor_lip_dominant_hz",
    "tremor_lip_motion_power", "tremor_lip_lock_in",
    "tremor_lip_pd_likelihood",
    "head_yaw_deg", "off_axis",
]


class VisitWriter(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.t0 = None
        self.traces = defaultdict(list)
        self.n_frames = 0
        self.flushed = False
        self._csv_file = None
        self._csv_writer = None
        self._embed_file = None
        self._embed_writer = None
        self._master_file = None
        self._master_writer = None
        # Mirror JSON array buffers (written at flush time so the JSON is a
        # well-formed array, not partial NDJSON)
        self._frames_json_path = None
        self._embed_json_path = None
        self._frames_buf = []
        self._embed_buf = []

    def build(self, metrics_out, patient_id="demo_patient", visit_id=None,
              duration_s=60, sex=None, age=None, label=None, state="visit"):
        self.patient_id = patient_id
        self.patient_num = patient_num_for(patient_id)
        self.visit_id = visit_id or date.today().isoformat()
        self.duration_s = duration_s
        self.sex = sex          # 0/1/None
        self.age = age          # int/None
        self.label = label      # 0/1/None
        self.state = state      # rest|talk|smile|visit
        self.link_args(metrics_out)
        return self

    def _open_csv(self):
        out_dir = DATA_DIR / self.patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # Biomarker per-frame CSV (existing schema, unchanged)
        path = out_dir / f"visit_{self.visit_id}.csv"
        self._csv_path = path
        self._csv_file = open(path, "w", buffering=1, newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(CSV_COLUMNS)
        # Per-visit ML embedding CSV (lower-face landmarks + metadata)
        epath = out_dir / f"embed_{self.visit_id}.csv"
        self._embed_path = epath
        self._embed_file = open(epath, "w", buffering=1, newline="")
        self._embed_writer = csv.writer(self._embed_file)
        self._embed_writer.writerow(EMBED_HEADER)
        # Master cross-patient dataset (append-only, header on first create)
        MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        master_existed = MASTER_DATASET_PATH.exists()
        self._master_file = open(MASTER_DATASET_PATH, "a", buffering=1, newline="")
        self._master_writer = csv.writer(self._master_file)
        if not master_existed:
            self._master_writer.writerow(EMBED_HEADER)
        # JSON mirrors — accumulated in memory, dumped at flush
        self._frames_json_path = out_dir / f"visit_{self.visit_id}_frames.json"
        self._embed_json_path = out_dir / f"embed_{self.visit_id}.json"
        print(f"[visit_writer] live CSV log    → {path}")
        print(f"[visit_writer] embed CSV       → {epath}")
        print(f"[visit_writer] frames JSON     → {self._frames_json_path}")
        print(f"[visit_writer] embed JSON      → {self._embed_json_path}")
        print(f"[visit_writer] master append   → {MASTER_DATASET_PATH} "
              f"(patient_num={self.patient_num})")

    def process(self, metrics_buf):
        if self.flushed:
            return
        if self.t0 is None:
            self.t0 = time.monotonic()
            self._open_csv()

        try:
            payload = json.loads(bytes(metrics_buf.getData()).decode("utf-8"))
        except Exception as e:
            print(f"[visit_writer] decode failed: {e}")
            return

        for k, v in payload.items():
            if isinstance(v, (int, float)):
                self.traces[k].append(v)
            elif v is None:
                pass
        self.n_frames += 1

        # Append a CSV row in real time (one per emitted metrics frame)
        if self._csv_writer is not None:
            row = []
            for col in CSV_COLUMNS:
                val = payload.get(col)
                row.append("" if val is None else val)
            try:
                self._csv_writer.writerow(row)
            except Exception as e:
                print(f"[visit_writer] csv write failed: {e}")

        # JSON mirror of the per-frame biomarker CSV — strip the heavy landmark
        # array (kept separately in the embed JSON) to avoid duplication.
        try:
            frame_lite = {k: v for k, v in payload.items() if k != "landmarks_lower"}
            self._frames_buf.append(frame_lite)
        except Exception:
            pass

        # Append embedding row: metadata + flat landmark XYZ
        # Same row goes to per-visit embed CSV AND master dataset CSV.
        if self._embed_writer is not None:
            kp_lookup = {kp["i"]: kp for kp in (payload.get("landmarks_lower") or [])}
            ts_ms = payload.get("timestamp_ms")
            if ts_ms is None:
                ts_ms = int(time.time() * 1000)
            ts_iso = datetime.fromtimestamp(ts_ms / 1000.0).isoformat(timespec="milliseconds")
            row = [
                self.patient_num,
                self.patient_id,
                self.visit_id,
                ts_iso,
                ts_ms,
                "" if self.sex is None else self.sex,
                "" if self.age is None else self.age,
                "" if self.label is None else self.label,
                self.state,
            ]
            for i in LOWER_FACE_FIXED_INDICES:
                kp = kp_lookup.get(i)
                if kp is None:
                    row.extend(["", "", ""])
                else:
                    row.extend([kp["x"], kp["y"], kp["z"]])
            try:
                self._embed_writer.writerow(row)
                if self._master_writer is not None:
                    self._master_writer.writerow(row)
                # JSON mirror of the embed CSV — same fields, dict per frame
                self._embed_buf.append(dict(zip(EMBED_HEADER, row)))
            except Exception as e:
                print(f"[visit_writer] embed write failed: {e}")

        elapsed = time.monotonic() - self.t0
        if elapsed >= self.duration_s:
            self._flush()

    def _flush(self):
        from .biomarkers import (
            NORMAL_BLINK_RANGE,
            PARKINSON_BLINK_THRESHOLD,
            SMILE_AMPL_NORMAL_MM,
            ASYMMETRY_NORMAL,
            GAZE_STABILITY_NORMAL,
            TREMOR_PARKINSON_BAND_HZ,
            TREMOR_NORMAL_POWER_MAX,
            VOICE_JITTER_NORMAL_PCT,
            VOICE_SHIMMER_NORMAL_PCT,
            VOICE_HNR_NORMAL_DB,
            VOICE_SPEECH_RATE_NORMAL_WPM,
        )

        def summarize(name):
            vals = np.asarray(self.traces.get(name, []), dtype=np.float64)
            if vals.size == 0:
                return None
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p95": float(np.percentile(vals, 95)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "trace": vals.round(3).tolist()[-600:],  # cap stored trace length
            }

        out = {
            "patient_id": self.patient_id,
            "patient_num": self.patient_num,
            "visit_id": self.visit_id,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": self.duration_s,
            "n_frames": self.n_frames,
            "sex": self.sex,
            "age": self.age,
            "label": self.label,
            "state": self.state,
            "metrics": {
                "blink_rate": summarize("blink_rate"),
                "smile_amplitude_mm": summarize("smile_amplitude_mm"),
                "brow_amplitude_mm": summarize("brow_amplitude_mm"),
                "asymmetry": summarize("asymmetry"),
                "asymmetry_std": summarize("asymmetry_std"),
                "gaze_x_std": summarize("gaze_x_std"),
                "gaze_y_std": summarize("gaze_y_std"),
                "face_distance_mm": summarize("face_distance_mm"),
                "ear": summarize("ear"),
                "tremor_chin_power_4_6hz": summarize("tremor_chin_power_4_6hz"),
                "tremor_chin_dominant_hz": summarize("tremor_chin_dominant_hz"),
                "tremor_chin_motion_power": summarize("tremor_chin_motion_power"),
                "tremor_chin_lock_in": summarize("tremor_chin_lock_in"),
                "tremor_chin_pd_likelihood": summarize("tremor_chin_pd_likelihood"),
                "tremor_lip_power_4_6hz": summarize("tremor_lip_power_4_6hz"),
                "tremor_lip_dominant_hz": summarize("tremor_lip_dominant_hz"),
                "tremor_lip_motion_power": summarize("tremor_lip_motion_power"),
                "tremor_lip_lock_in": summarize("tremor_lip_lock_in"),
                "tremor_lip_pd_likelihood": summarize("tremor_lip_pd_likelihood"),
                "head_yaw_deg": summarize("head_yaw_deg"),
                "off_axis": summarize("off_axis"),
            },
            "ranges": {
                "blink_rate_normal": list(NORMAL_BLINK_RANGE),
                "blink_rate_parkinson_threshold": PARKINSON_BLINK_THRESHOLD,
                "smile_amplitude_normal_mm": list(SMILE_AMPL_NORMAL_MM),
                "asymmetry_normal_max": ASYMMETRY_NORMAL,
                "gaze_stability_normal_max": GAZE_STABILITY_NORMAL,
                "tremor_parkinson_band_hz": list(TREMOR_PARKINSON_BAND_HZ),
                "tremor_normal_power_max": TREMOR_NORMAL_POWER_MAX,
                "voice_jitter_normal_pct": VOICE_JITTER_NORMAL_PCT,
                "voice_shimmer_normal_pct": VOICE_SHIMMER_NORMAL_PCT,
                "voice_hnr_normal_db": VOICE_HNR_NORMAL_DB,
                "voice_speech_rate_normal_wpm": list(VOICE_SPEECH_RATE_NORMAL_WPM),
            },
            "disclaimer": (
                "NeuroVista is a research/screening prototype. "
                "It is NOT a medical device and does NOT provide a diagnosis. "
                "Clinical interpretation requires a licensed neurologist."
            ),
        }

        out_dir = DATA_DIR / self.patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"visit_{self.visit_id}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[visit_writer] wrote {out_path}  ({self.n_frames} frames)")

        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            print(f"[visit_writer] sealed CSV log {self._csv_path}")
        if self._embed_file is not None:
            try:
                self._embed_file.flush()
                self._embed_file.close()
            except Exception:
                pass
            print(f"[visit_writer] sealed embed CSV {self._embed_path}")
        if self._master_file is not None:
            try:
                self._master_file.flush()
                self._master_file.close()
            except Exception:
                pass
            print(f"[visit_writer] flushed master append → {MASTER_DATASET_PATH}")
        # Dump the JSON mirrors as a single well-formed array
        if self._frames_json_path is not None:
            try:
                with open(self._frames_json_path, "w") as f:
                    json.dump(self._frames_buf, f, indent=2, default=str)
                print(f"[visit_writer] wrote frames JSON → {self._frames_json_path} "
                      f"({len(self._frames_buf)} frames)")
            except Exception as e:
                print(f"[visit_writer] frames JSON write failed: {e}")
        if self._embed_json_path is not None:
            try:
                with open(self._embed_json_path, "w") as f:
                    json.dump(self._embed_buf, f, indent=2, default=str)
                print(f"[visit_writer] wrote embed JSON → {self._embed_json_path} "
                      f"({len(self._embed_buf)} frames)")
            except Exception as e:
                print(f"[visit_writer] embed JSON write failed: {e}")
        self.flushed = True
