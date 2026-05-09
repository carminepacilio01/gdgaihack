"""Unit tests for face signal processing.

Run without LLM dependencies. Validate that our feature extractors produce
the expected numerical signature on synthetic face landmark inputs.
"""
from __future__ import annotations

import numpy as np

from parkinson_agent.oak_adapter import FaceTimeSeries
from parkinson_agent.run_demo import _base_face, synthetic_session
from parkinson_agent.signal_processing import (
    REGION_INDICES,
    REGION_WEIGHTS,
    blink_rate,
    jaw_tremor,
    mouth_corner_asymmetry,
    regional_motion,
)


N_LANDMARKS = 478


def _static_face_timeseries(n_frames: int, fps: float = 30.0) -> FaceTimeSeries:
    """A perfectly still face — useful as a control."""
    base = _base_face()
    landmarks = np.tile(base, (n_frames, 1, 1))
    timestamps = np.arange(n_frames) / fps
    return FaceTimeSeries(timestamps=timestamps, landmarks=landmarks, blink_events=[])


class TestRegionalMotion:
    def test_static_face_has_low_composite(self):
        face = _static_face_timeseries(300)
        out = regional_motion(face)
        assert out["valid"] is True
        # Static face: composite expressivity should be ~0.
        assert out["composite_expressivity_score"] < 0.01

    def test_weights_match_clinical_priors(self):
        face = _static_face_timeseries(300)
        out = regional_motion(face)
        # Weights are reported alongside metrics.
        assert out["weights_applied"]["chin_jaw"] == REGION_WEIGHTS["chin_jaw"]
        assert out["per_region"]["lower_lip"]["weight"] == REGION_WEIGHTS["lower_lip"]

    def test_synthetic_session_shows_reduced_lower_face_motion(self):
        session = synthetic_session()
        expr_face = session.face_in_task("facial_expression")
        out = regional_motion(expr_face)
        assert out["valid"] is True
        # Per the synthetic patient design, chin/lower-lip RoM is small.
        chin = out["per_region"]["chin_jaw"]["range_of_motion"]
        lower_lip = out["per_region"]["lower_lip"]["range_of_motion"]
        upper_lip = out["per_region"]["upper_lip"]["range_of_motion"]
        # High-weight regions should move LESS than the upper lip in our hypomimia synth.
        assert chin < upper_lip
        assert lower_lip < upper_lip


class TestJawTremor:
    def test_clean_4_5hz_tremor_detected(self):
        # Synthesize chin-only oscillation at 4.5 Hz.
        fps, duration, freq = 60.0, 10.0, 4.5
        n = int(fps * duration)
        t = np.arange(n) / fps
        base = _base_face()
        landmarks = np.tile(base, (n, 1, 1))
        amp = 0.005
        for idx in REGION_INDICES["chin_jaw"]:
            landmarks[:, idx, 1] += amp * np.sin(2 * np.pi * freq * t)
        landmarks += np.random.default_rng(0).normal(0, 0.0001, landmarks.shape)
        face = FaceTimeSeries(timestamps=t, landmarks=landmarks, blink_events=[])

        out = jaw_tremor(face)
        assert out["valid"] is True
        assert abs(out["dominant_frequency_hz"] - freq) < 0.5
        assert out["in_parkinsonian_range_4_6hz"] is True

    def test_no_tremor_low_in_band(self):
        # Pure low-frequency drift on chin — should NOT look like tremor.
        fps, duration = 60.0, 10.0
        n = int(fps * duration)
        t = np.arange(n) / fps
        base = _base_face()
        landmarks = np.tile(base, (n, 1, 1))
        for idx in REGION_INDICES["chin_jaw"]:
            landmarks[:, idx, 1] += 0.001 * np.sin(2 * np.pi * 0.3 * t)
        landmarks += np.random.default_rng(1).normal(0, 0.0001, landmarks.shape)
        face = FaceTimeSeries(timestamps=t, landmarks=landmarks, blink_events=[])

        out = jaw_tremor(face)
        if out["valid"]:
            assert out["in_band_fraction_of_total"] < 0.4


class TestMouthAsymmetry:
    def test_static_face_is_symmetric(self):
        face = _static_face_timeseries(300)
        out = mouth_corner_asymmetry(face)
        # No motion either side -> ratio undefined; expect small or zero.
        assert out["valid"] is True
        assert out["asymmetry_ratio"] < 0.5

    def test_synthetic_session_flags_left_side(self):
        session = synthetic_session()
        face_expr = session.face_in_task("facial_expression")
        out = mouth_corner_asymmetry(face_expr)
        assert out["valid"] is True
        # The synthetic patient has reduced LEFT-side mobility (291 in image coords).
        assert out["less_mobile_side"] == "left"
        assert out["asymmetry_ratio"] > 0.3


class TestBlinkRate:
    def test_blink_rate_per_minute(self):
        # 4 blinks in 60s -> 4 bpm
        n_frames = 1800  # 60s @ 30fps
        face = _static_face_timeseries(n_frames)
        face.blink_events = [5.0, 20.0, 35.0, 50.0]
        out = blink_rate(face)
        assert out["valid"] is True
        assert abs(out["blink_rate_per_min"] - 4.0) < 0.2
