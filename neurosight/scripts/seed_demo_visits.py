"""Generate synthetic visit JSONs for offline dashboard preview.

Useful when you don't yet have curated patient videos but want to develop the
dashboard. Produces a fictional 'demo_patient' with 4 visits showing a
mild downward trend in blink_rate, smile amplitude and brow amplitude
(consistent with progressive Parkinson facial bradykinesia).

NOT to be used in the live demo — these numbers are SIMULATED, not measured.

Run:
  python3 scripts/seed_demo_visits.py
"""
import json
from datetime import date
from pathlib import Path
import random

random.seed(42)

VISITS = [
    "2025-09-01",
    "2025-11-15",
    "2026-01-30",
    "2026-04-12",
]

# baseline → progression curves (mean values per visit)
TRENDS = {
    "blink_rate":             [18.5, 16.0, 13.5, 11.2],     # decreasing
    "smile_amplitude_mm":     [52.0, 47.5, 42.0, 37.5],     # decreasing
    "brow_amplitude_mm":      [9.0,  7.5,  5.5,  4.2],      # decreasing
    "asymmetry":              [0.04, 0.05, 0.07, 0.09],     # increasing
    "asymmetry_std":          [0.01, 0.012, 0.018, 0.022],
    "gaze_x_std":             [0.02, 0.025, 0.032, 0.040],
    "gaze_y_std":             [0.018, 0.022, 0.028, 0.034],
    "face_distance_mm":       [600,  590,  580,  590],
    "ear":                    [0.27, 0.25, 0.24, 0.22],
    # Tremor — power in 4-6 Hz band rises, dominant freq locks into the band
    "tremor_chin_power_4_6hz":  [1.0e-4, 2.5e-4, 5.5e-4, 9.0e-4],
    "tremor_chin_dominant_hz":  [3.2,    4.1,    4.6,    5.0],
    "tremor_lip_power_4_6hz":   [0.8e-4, 1.8e-4, 4.0e-4, 7.5e-4],
    "tremor_lip_dominant_hz":   [3.0,    4.0,    4.5,    4.9],
}

# Voice features (Praat + Whisper) — degrade with PD progression.
# Refs: Rusz 2011 (jitter/shimmer), Goberman 2002 (HNR), Skodda 2011 (rate slowing)
VOICE_TRENDS = {
    "jitter_local_pct":  [0.55, 0.85, 1.30, 1.85],   # > 1.04 = pathological
    "shimmer_local_pct": [2.40, 3.20, 4.40, 5.50],   # > 3.81 = pathological
    "hnr_db":            [22.5, 20.8, 18.2, 15.6],   # < 20 = breathy
    "f0_mean_hz":        [148,  142,  136,  130],
    "f0_std_hz":         [22,   18,   14,   11],     # monotone-ization
    "intensity_db":      [68.0, 65.5, 62.8, 60.4],
    "speech_rate_wpm":   [165,  150,  135,  118],    # < 130 abnormal
    "pause_ratio":       [0.18, 0.24, 0.32, 0.41],
    "wav_seconds":       [60.0, 60.0, 60.0, 60.0],
    "transcript_word_count": [165, 150, 135, 118],
}

RANGES = {
    "blink_rate_normal": [15, 22],
    "blink_rate_parkinson_threshold": 12,
    "smile_amplitude_normal_mm": [35.0, 60.0],
    "asymmetry_normal_max": 0.05,
    "gaze_stability_normal_max": 0.15,
    "tremor_parkinson_band_hz": [4.0, 6.0],
    "tremor_normal_power_max": 5e-4,
    "voice_jitter_normal_pct": 1.04,
    "voice_shimmer_normal_pct": 3.81,
    "voice_hnr_normal_db": 20.0,
    "voice_speech_rate_normal_wpm": [130, 180],
}


def synthesize(mean, n=300, noise=0.05):
    return [round(mean * (1 + random.uniform(-noise, noise)), 3) for _ in range(n)]


def main():
    out_dir = Path("data/visits/demo_patient")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, vid in enumerate(VISITS):
        metrics = {}
        for k, trend in TRENDS.items():
            mean = trend[i]
            trace = synthesize(mean)
            metrics[k] = {
                "mean": mean,
                "std": round(abs(mean) * 0.05, 6),
                "p95": round(mean * 1.07, 6),
                "min": round(min(trace), 6),
                "max": round(max(trace), 6),
                "trace": trace,
            }
        # Voice features come as a single flat dict, not per-frame summaries
        voice = {k: VOICE_TRENDS[k][i] for k in VOICE_TRENDS}
        voice["transcript_excerpt"] = (
            "the rainbow is a division of white light into many beautiful colors"
            if i < 2
            else "the rainbow division of light into colors"
        )
        voice["errors"] = []
        visit = {
            "patient_id": "demo_patient",
            "visit_id": vid,
            "recorded_at": f"{vid}T10:00:00",
            "duration_s": 60,
            "n_frames": 1800,
            "metrics": metrics,
            "voice": voice,
            "ranges": RANGES,
            "disclaimer": "SYNTHETIC visit — for dashboard development only.",
            "synthetic": True,
        }
        path = out_dir / f"visit_{vid}.json"
        with open(path, "w") as f:
            json.dump(visit, f, indent=2)
        print(f"  wrote {path}")
    print("done.")


if __name__ == "__main__":
    main()
