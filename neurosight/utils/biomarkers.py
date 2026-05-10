"""Pure-function biomarker calculators. Stateless per-frame computations.

MediaPipe Face Landmarker landmark indices used:
  Left eye  EAR pts  : 33, 160, 158, 133, 144, 153
  Right eye EAR pts  : 263, 387, 385, 362, 373, 380
  Mouth corners      : 61 (left), 291 (right)
  Upper / lower lip  : 13 (upper inner), 14 (lower inner)
  Brow inner L / R   : 65, 295
  Brow outer L / R   : 70, 300
  Nose tip / bridge  : 4, 168
  Face mid-axis      : 168 (top of nose), 152 (chin)

References:
  • MediaPipe FaceMesh canonical landmark map (468 pts)
  • Soukupová & Čech (2016) — Eye Aspect Ratio for blink detection
  • Bologna et al., Brain 2013 — Hypomimia in Parkinson's
  • Karson 1983 — Blink rate < 12/min as Parkinson indicator
"""
from collections import deque
import numpy as np

# ─── Landmark indices (MediaPipe FaceMesh) ──────────────────────────
LEFT_EYE_EAR = [33, 160, 158, 133, 144, 153]
RIGHT_EYE_EAR = [263, 387, 385, 362, 373, 380]

# Full eye + iris regions, used to EXCLUDE eyes from the lower-face mask.
# We pick all standard MediaPipe eye-contour indices (not just EAR points).
EYE_REGION_INDICES = set([
    # Left eye contour
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
    # Right eye contour
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
    # EAR specials (already in above, kept for safety)
    160, 158, 144, 153, 387, 385, 373, 380,
])

# Curated lower-face landmark whitelist for ML training embedding.
# Selected from MediaPipe FaceMesh canonical regions: lips (outer+inner), nose,
# lower cheeks, chin, jaw line. Eyes and brows are deliberately excluded — PD
# facial masking dominates the lower face (Bologna 2013).
# Order is fixed and reproducible → CSV columns are stable across visits.
LOWER_FACE_FIXED_INDICES = sorted(set([
    # Lips outer
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
    # Lips inner
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
    # Nose
    1, 2, 4, 5, 19, 20, 94, 97, 98, 168, 195, 197, 326, 327,
    # Lower cheeks (left)
    50, 101, 36, 205, 187, 207, 213, 192, 147, 123,
    # Lower cheeks (right)
    280, 330, 266, 425, 411, 427, 433, 416, 376, 352,
    # Chin / jaw line
    18, 32, 83, 89, 96, 132, 134, 138, 140, 148, 149, 150, 152, 169, 170,
    171, 172, 175, 176, 177, 199, 208, 211, 215,
    262, 287, 288, 313, 318, 365, 397, 432, 435, 436, 361,
]))

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LIP_UPPER = 13
LIP_LOWER = 14

BROW_INNER_L = 65
BROW_OUTER_L = 70
BROW_INNER_R = 295
BROW_OUTER_R = 300
EYE_TOP_L = 159  # upper eyelid mid, used as reference for brow lift
EYE_TOP_R = 386

NOSE_BRIDGE = 168
CHIN = 152
NOSE_TIP = 4

# Pairs for L/R asymmetry (mirror across mid-axis)
ASYMMETRY_PAIRS = [
    (33, 263),    # outer eye corners
    (133, 362),   # inner eye corners
    (61, 291),    # mouth corners
    (70, 300),    # outer brow
    (65, 295),    # inner brow
    (234, 454),   # cheek edges
]

# Clinical reference ranges (for dashboard color coding)
NORMAL_BLINK_RANGE = (15, 22)        # blinks/min, healthy adult resting
PARKINSON_BLINK_THRESHOLD = 12       # < this = suggestive
SMILE_AMPL_NORMAL_MM = (35.0, 60.0)  # peak-to-peak mouth corner spread Δ
ASYMMETRY_NORMAL = 0.05              # < this = symmetric
GAZE_STABILITY_NORMAL = 0.15         # saccade std (lower = more stable)

# Tremor (Bologna et al., Brain 2013 — jaw/lip tremor in PD around 4-6 Hz,
# matching Bain 2003 limb resting-tremor band)
TREMOR_PARKINSON_BAND_HZ = (4.0, 6.0)
TREMOR_NORMAL_POWER_MAX = 5e-4       # combined X+Y normalized — empirically tuned

# Voice (Rusz et al., J Acoust Soc Am 2011; Goberman & Coelho 2002)
VOICE_JITTER_NORMAL_PCT = 1.04       # > this = pathological dysphonia
VOICE_SHIMMER_NORMAL_PCT = 3.81      # > this = pathological
VOICE_HNR_NORMAL_DB = 20.0           # < this = breathy / hoarse
VOICE_SPEECH_RATE_NORMAL_WPM = (130, 180)


def _kp_to_xy(landmarks, idx, w, h):
    kp = landmarks[idx]
    return np.array([kp.x * w, kp.y * h], dtype=np.float32)


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    """Soukupová–Čech EAR. Lower = eye more closed."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C == 0:
        return 0.0
    return float((A + B) / (2.0 * C))


def both_eyes_ear(landmarks, w, h):
    return (
        eye_aspect_ratio(landmarks, LEFT_EYE_EAR, w, h)
        + eye_aspect_ratio(landmarks, RIGHT_EYE_EAR, w, h)
    ) / 2.0


def asymmetry_score(landmarks, w, h):
    """Mirror landmarks across the nose mid-axis, compare normalized residuals.

    Returns scalar in [0, 1+]; 0 = perfectly symmetric.
    """
    nose = _kp_to_xy(landmarks, NOSE_BRIDGE, w, h)
    chin = _kp_to_xy(landmarks, CHIN, w, h)
    axis = chin - nose
    if np.linalg.norm(axis) < 1e-6:
        return 0.0
    axis_norm = axis / np.linalg.norm(axis)
    perp = np.array([-axis_norm[1], axis_norm[0]])

    residuals = []
    for li, ri in ASYMMETRY_PAIRS:
        pl = _kp_to_xy(landmarks, li, w, h) - nose
        pr = _kp_to_xy(landmarks, ri, w, h) - nose
        # Reflect right point across the axis: mirror = p - 2(p·perp)perp
        pr_mirror = pr - 2 * np.dot(pr, perp) * perp
        residuals.append(np.linalg.norm(pl - pr_mirror))

    face_diag = np.linalg.norm(_kp_to_xy(landmarks, 234, w, h) - _kp_to_xy(landmarks, 454, w, h))
    if face_diag < 1e-3:
        return 0.0
    return float(np.mean(residuals) / face_diag)


def head_yaw_deg(landmarks, w, h):
    """Approximate head yaw in degrees from a small set of landmarks.

    Positive = head turned to the camera-right. Uses the nose-tip horizontal
    offset relative to the midpoint of the cheek edges, scaled by the face
    width. Cheap, no PnP solver needed; accurate to ±5° in the ±45° range.
    """
    L_cheek = _kp_to_xy(landmarks, 234, w, h)   # left cheek edge
    R_cheek = _kp_to_xy(landmarks, 454, w, h)   # right cheek edge
    nose = _kp_to_xy(landmarks, NOSE_TIP, w, h)
    face_w = float(np.linalg.norm(R_cheek - L_cheek))
    if face_w < 1e-3:
        return 0.0
    midline_x = (L_cheek[0] + R_cheek[0]) / 2.0
    # Normalized horizontal offset of nose from face midline ([-0.5, +0.5])
    offset = (nose[0] - midline_x) / face_w
    # Empirical mapping: full-profile (offset ~ ±0.5) ≈ ±90°
    return float(offset * 180.0)


def gaze_proxy(landmarks, w, h):
    """Cheap gaze proxy from face landmarks: iris approximated by eye-center.

    Returns (gaze_x, gaze_y) normalized; for stability we only need temporal var.
    """
    # eye centers
    le = (_kp_to_xy(landmarks, 33, w, h) + _kp_to_xy(landmarks, 133, w, h)) / 2
    re = (_kp_to_xy(landmarks, 263, w, h) + _kp_to_xy(landmarks, 362, w, h)) / 2
    nose = _kp_to_xy(landmarks, NOSE_TIP, w, h)
    # offset of eye-center pair from nose-bridge as crude gaze direction
    return ((le + re) / 2 - nose) / max(w, h)


# ─── Stateful trackers (sliding-window aggregates over a visit) ─────

class BlinkCounter:
    """Counts blinks via EAR threshold crossings + minimum frame gap (debounce).

    Uses a relative threshold: tracks the rolling min/max EAR over the last
    ~10 s and triggers a blink when EAR drops below `mid - margin*range`.
    This is robust to subjects with naturally narrow eyes (e.g. wearing
    glasses), where a fixed absolute 0.20 threshold never triggers.
    """

    def __init__(self, ear_threshold=0.22, min_frames_open=2, fps=30,
                 adaptive=True, calib_window_s=10.0):
        self.thr = ear_threshold
        self.min_open = min_frames_open
        self.fps = fps
        self.adaptive = adaptive
        self.calib_window_s = calib_window_s
        self.was_closed = False
        self.frames_open_since_blink = 0
        self.timestamps = deque()  # sec since start, recent only
        self.recent_ear = deque()  # (t, ear) for adaptive threshold
        self.last_threshold = ear_threshold  # exposed for HUD debug

    # Warm-up: don't count blinks until the adaptive threshold has stabilized.
    # ~3 s of accumulated EAR samples covers a full calibration window even
    # when the subject keeps eyes open the whole time.
    WARMUP_FRAMES = 90

    @property
    def is_calibrated(self):
        return self.adaptive and len(self.recent_ear) >= self.WARMUP_FRAMES

    def update(self, ear_value, t_seconds):
        # Maintain rolling window of recent EAR values for adaptive thresholding
        self.recent_ear.append((t_seconds, ear_value))
        while self.recent_ear and t_seconds - self.recent_ear[0][0] > self.calib_window_s:
            self.recent_ear.popleft()

        if self.adaptive and len(self.recent_ear) >= 30:
            vals = [v for _, v in self.recent_ear]
            v_min = min(vals)
            v_max = max(vals)
            # Trigger when EAR drops below 70% of the open-eye baseline.
            # Falls back to fixed threshold if range is too small (no blinks yet).
            if (v_max - v_min) > 0.04:
                threshold = v_max - 0.30 * (v_max - v_min)
            else:
                threshold = self.thr
        else:
            threshold = self.thr
        self.last_threshold = threshold

        # During warm-up: track edge state (so we don't double-count the first
        # blink after calibration), but DON'T enqueue any blink timestamps.
        is_closed = ear_value < threshold
        blinked = False
        if not self.is_calibrated:
            self.was_closed = is_closed
            self.frames_open_since_blink = 0
            return blinked

        if is_closed and not self.was_closed:
            if self.frames_open_since_blink >= self.min_open:
                self.timestamps.append(t_seconds)
                blinked = True
            self.frames_open_since_blink = 0
        elif not is_closed:
            self.frames_open_since_blink += 1
        self.was_closed = is_closed
        # purge older than 60s
        while self.timestamps and t_seconds - self.timestamps[0] > 60.0:
            self.timestamps.popleft()
        return blinked

    def rate_per_minute(self, t_seconds):
        # Don't emit a rate during warm-up — caller can show "—" instead of 0
        if not self.is_calibrated:
            return None
        if t_seconds < 1.0:
            return None
        if len(self.timestamps) == 0:
            return 0.0
        # Window grows from 0 to 60s; before then, scale to per-minute by elapsed
        window_s = min(60.0, t_seconds)
        return float(len(self.timestamps) * (60.0 / window_s))



class AmplitudeTracker:
    """Tracks min/max of a scalar (e.g. mouth width in mm) over a sliding window."""

    def __init__(self, window_s=30.0):
        self.window_s = window_s
        self.history = deque()  # (t, value)

    def update(self, value, t_seconds):
        if value is None or np.isnan(value):
            return
        self.history.append((t_seconds, value))
        while self.history and t_seconds - self.history[0][0] > self.window_s:
            self.history.popleft()

    def amplitude(self):
        if len(self.history) < 5:
            return 0.0
        vals = [v for _, v in self.history]
        return float(max(vals) - min(vals))

    def std(self):
        if len(self.history) < 5:
            return 0.0
        vals = [v for _, v in self.history]
        return float(np.std(vals))


class TremorTracker:
    """FFT-based tremor detection on a single 2D landmark trajectory.

    Position samples are stored as a deque of (x, y) values that should already
    be normalized by face size (e.g. face-diagonal in pixels) so the metric is
    invariant to camera distance. Detrend + Hanning window + real FFT, returns
    the spectral power in the Parkinson resting-tremor band 4–6 Hz, plus the
    dominant frequency in 0.5–12 Hz.

    Bain 2003: limb resting tremor 4–6 Hz.
    Bologna et al., Brain 2013: jaw / lip tremor in PD shares the same band.
    """

    def __init__(self, fps=30, window_s=10.0, freq_band=(4.0, 6.0)):
        self.fps = float(fps)
        self.window_n = int(fps * window_s)
        self.freq_band = freq_band
        self.buf_x = deque(maxlen=self.window_n)
        self.buf_y = deque(maxlen=self.window_n)
        # Lock-in tracking: a tremor that "locks" into the PD band stays there
        # for several seconds. Chewing / talking transit through the band but
        # don't dwell. ~3 s window of dominant-freq history.
        self.dom_history = deque(maxlen=int(fps * 3.0))

    def update(self, x_norm, y_norm):
        if x_norm is None or y_norm is None:
            return
        if np.isnan(x_norm) or np.isnan(y_norm):
            return
        self.buf_x.append(float(x_norm))
        self.buf_y.append(float(y_norm))

    def _axis_psd(self, samples):
        n = len(samples)
        if n < 32:  # need at least ~1 s of data
            return None, None
        s = np.asarray(samples, dtype=np.float64)
        s = s - np.mean(s)            # detrend
        win = np.hanning(n)
        s = s * win
        spectrum = np.fft.rfft(s)
        psd = (np.abs(spectrum) ** 2) / max(np.sum(win ** 2) * self.fps, 1e-12)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fps)
        return freqs, psd

    def power_band(self):
        """Multi-feature spectral analysis for PD-tremor classification.

        Returns dict with:
          - band_power_pd: energy in 4-6 Hz (PD resting tremor band)
          - band_power_low: energy in 1-3 Hz (chewing / postural drift)
          - band_power_high: energy in 6-12 Hz (tic / jitter / fast motion)
          - motion_power: total energy 0.5-12 Hz
          - dom_freq: argmax in 2-12 Hz (skip postural drift)
          - lock_in_ratio: fraction of last ~3s where dom_freq was in 3-7 Hz
          - pd_likelihood: 0-100% classification score combining all features

        Likelihood logic:
          • ratio = pd_power / max(low_power, eps): >1 means PD band dominates
            chewing/postural energy. Sensitive even at low absolute amplitudes.
          • lock_in: tremor sustains; chewing & talking don't.
          • penalty if motion is high but PD band is dominated by low band
            (= chewing). Idle (motion ≈ 0) clamps likelihood to 0.
        """
        empty = {
            "band_power_pd": 0.0, "band_power_low": 0.0, "band_power_high": 0.0,
            "motion_power": 0.0, "dom_freq": 0.0,
            "lock_in_ratio": 0.0, "pd_likelihood": 0.0,
        }
        if len(self.buf_x) < max(32, self.window_n // 4):
            return empty
        fx, px = self._axis_psd(list(self.buf_x))
        fy, py = self._axis_psd(list(self.buf_y))
        if fx is None or fy is None:
            return empty
        psd = px + py

        pd_mask = (fx >= self.freq_band[0]) & (fx <= self.freq_band[1])
        low_mask = (fx >= 1.0) & (fx < 3.0)
        high_mask = (fx > 7.0) & (fx <= 12.0)
        motion_mask = (fx >= 0.5) & (fx <= 12.0)
        domain_mask = (fx >= 2.0) & (fx <= 12.0)

        pd_power = float(np.sum(psd[pd_mask]))
        low_power = float(np.sum(psd[low_mask]))
        high_power = float(np.sum(psd[high_mask]))
        motion_power = float(np.sum(psd[motion_mask]))

        if np.any(domain_mask):
            dom_freq = float(fx[domain_mask][np.argmax(psd[domain_mask])])
        else:
            dom_freq = 0.0
        # Track dominant freq history for lock-in ratio
        self.dom_history.append(dom_freq)
        if len(self.dom_history) > 0:
            n_in_band = sum(1 for f in self.dom_history if 3.0 <= f <= 7.0)
            lock_in = n_in_band / float(len(self.dom_history))
        else:
            lock_in = 0.0

        # Likelihood scoring (0-100%)
        eps = 1e-12
        IDLE_THR = 1e-5  # below this, jaw is essentially still
        ratio = pd_power / max(low_power, eps)

        if motion_power < IDLE_THR:
            likelihood = 0.0
        else:
            # Base: spectral PD-vs-low ratio (sigmoid-ish)
            ratio_score = min(60.0, 30.0 * np.log1p(ratio))
            # Lock-in bonus
            lock_bonus = 40.0 * lock_in
            # Chewing / talking penalty: low band dominates while motion is high
            chewing_penalty = 0.0
            if low_power > 1.5 * pd_power and motion_power > 5e-5:
                chewing_penalty = 30.0
            likelihood = max(0.0, min(100.0,
                ratio_score + lock_bonus - chewing_penalty))

        return {
            "band_power_pd": pd_power,
            "band_power_low": low_power,
            "band_power_high": high_power,
            "motion_power": motion_power,
            "dom_freq": dom_freq,
            "lock_in_ratio": lock_in,
            "pd_likelihood": likelihood,
        }
