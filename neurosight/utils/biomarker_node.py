"""BiomarkerExtractor — central HostNode orchestrating per-visit metric extraction.

Inputs (linked):
  • gather_out  — synced (detections, landmarks) bundle from depthai_nodes.GatherData
  • depth_out   — stereo depth aligned to CAM_A (optional; None for replay mode)

Output:
  • metrics_out — dai.Buffer carrying a serialized JSON dict with current metrics:
      { t: float, blink_rate: float, smile_amp_mm: float|None,
        brow_lift_mm: float|None, asymmetry: float, gaze_dx: float, gaze_dy: float,
        face_distance_mm: float|None, frame_idx: int }
"""
import json
import time
from collections import deque
import numpy as np
import depthai as dai

from depthai_nodes import ImgDetectionsExtended, Keypoints

from .biomarkers import (
    both_eyes_ear,
    asymmetry_score,
    gaze_proxy,
    head_yaw_deg,
    BlinkCounter,
    AmplitudeTracker,
    TremorTracker,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    BROW_OUTER_L,
    BROW_OUTER_R,
    EYE_TOP_L,
    EYE_TOP_R,
    CHIN,
    LIP_UPPER,
    NOSE_BRIDGE,
    LEFT_EYE_EAR,
    RIGHT_EYE_EAR,
    EYE_REGION_INDICES,
    LOWER_FACE_FIXED_INDICES,
)

# Head-pose gate: when |yaw| exceeds this many degrees, suppress metrics that
# are direction-dependent (asymmetry mirror, gaze proxy). Other metrics keep
# updating because they are geometric ratios or temporal (blink, smile mm,
# tremor FFT).
HEAD_POSE_YAW_GATE_DEG = 25.0
from .depth_utils import (
    get_intrinsics,
    pixel_to_world_mm,
    euclid_mm,
    safe_depth_at,
)


class BiomarkerExtractor(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.metrics_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )
        self.t0 = None
        self.frame_idx = 0
        self.blink = BlinkCounter(ear_threshold=0.22, fps=30)
        self.smile_amp = AmplitudeTracker(window_s=30.0)
        self.brow_amp = AmplitudeTracker(window_s=30.0)
        self.asym_amp = AmplitudeTracker(window_s=30.0)
        self.gaze_x_amp = AmplitudeTracker(window_s=30.0)
        self.gaze_y_amp = AmplitudeTracker(window_s=30.0)
        # Tremor: rolling 10s FFT of jaw / lip position (normalized by face size)
        self.tremor_chin = TremorTracker(fps=30, window_s=10.0, freq_band=(4.0, 6.0))
        self.tremor_lip = TremorTracker(fps=30, window_s=10.0, freq_band=(4.0, 6.0))
        # Boxcar smoothing buffers for tremor landmark positions (3-frame window).
        # White-noise jitter from INT8 FaceMesh attenuates ~3× while a real 4-6 Hz
        # signal (~5-8 frames per period at 30 FPS) survives almost unchanged.
        self._chin_smooth = deque(maxlen=3)
        self._lip_smooth = deque(maxlen=3)
        self._K = None
        self._K_size = None

    def build(self, gather_out, depth_out, calib, fps):
        self.calib = calib
        self.fps = fps
        self.depth_enabled = depth_out is not None
        if self.depth_enabled:
            self.link_args(gather_out, depth_out)
        else:
            self.link_args(gather_out)
        return self

    def _ensure_intrinsics(self, w, h):
        if self._K is None or self._K_size != (w, h):
            self._K = np.array(
                self.calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h)
            )
            self._K_size = (w, h)

    def process(self, gather_msg, depth_msg=None):
        self.frame_idx += 1
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0

        det_msg: ImgDetectionsExtended = gather_msg.reference_data
        if not isinstance(det_msg, ImgDetectionsExtended):
            return

        landmarks_list = gather_msg.gathered
        if not landmarks_list or not isinstance(landmarks_list[0], Keypoints):
            return
        if not det_msg.detections:
            return
        raw_landmarks = landmarks_list[0].keypoints  # first detected face only
        det = det_msg.detections[0]

        src_w, src_h = det_msg.transformation.getSize()
        self._ensure_intrinsics(src_w, src_h)

        # ─── Rectify landmark coords from face-crop space → full-frame [0,1] ───
        # The face landmarker outputs landmarks normalized to its 192×192 crop.
        # Map them back to the original frame using the YuNet detection bbox.
        rr = det.rotated_rect
        bcx, bcy = rr.center.x, rr.center.y
        bw, bh = rr.size.width, rr.size.height

        class _Kp:
            __slots__ = ("x", "y", "z")
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

        landmarks = [
            _Kp(bcx + (kp.x - 0.5) * bw, bcy + (kp.y - 0.5) * bh,
                getattr(kp, "z", 0.0))
            for kp in raw_landmarks
        ]

        # ─── 0. Head pose (yaw) — gates direction-dependent metrics ───
        yaw_deg = head_yaw_deg(landmarks, src_w, src_h)
        off_axis = abs(yaw_deg) > HEAD_POSE_YAW_GATE_DEG

        # ─── 1. Blink rate (via EAR) ─── (yaw-robust, always update)
        ear = both_eyes_ear(landmarks, src_w, src_h)
        self.blink.update(ear, t)
        blink_rate = self.blink.rate_per_minute(t)

        # ─── 2. Asymmetry ─── (mirror is biased by yaw → gate)
        asym = asymmetry_score(landmarks, src_w, src_h)
        if not off_axis:
            self.asym_amp.update(asym, t)

        # ─── 3. Gaze proxy ─── (eye-center vs nose biased by yaw → gate)
        gx, gy = gaze_proxy(landmarks, src_w, src_h)
        if not off_axis:
            self.gaze_x_amp.update(float(gx), t)
            self.gaze_y_amp.update(float(gy), t)

        # ─── 3b. Tremor (jaw + upper lip), distance-invariant via face-diag norm ───
        face_diag_px = float(np.hypot(
            (landmarks[454].x - landmarks[234].x) * src_w,
            (landmarks[454].y - landmarks[234].y) * src_h,
        ))
        if face_diag_px > 1.0:
            nose_x = landmarks[NOSE_BRIDGE].x * src_w
            nose_y = landmarks[NOSE_BRIDGE].y * src_h
            chin_dx = (landmarks[CHIN].x * src_w - nose_x) / face_diag_px
            chin_dy = (landmarks[CHIN].y * src_h - nose_y) / face_diag_px
            lip_dx = (landmarks[LIP_UPPER].x * src_w - nose_x) / face_diag_px
            lip_dy = (landmarks[LIP_UPPER].y * src_h - nose_y) / face_diag_px
            # Boxcar 3-frame smoothing to attenuate ML jitter before FFT
            self._chin_smooth.append((chin_dx, chin_dy))
            self._lip_smooth.append((lip_dx, lip_dy))
            chin_avg = np.mean(self._chin_smooth, axis=0)
            lip_avg = np.mean(self._lip_smooth, axis=0)
            self.tremor_chin.update(float(chin_avg[0]), float(chin_avg[1]))
            self.tremor_lip.update(float(lip_avg[0]), float(lip_avg[1]))
        chin_t = self.tremor_chin.power_band()
        lip_t = self.tremor_lip.power_band()

        # ─── 4. Hypomimia (in mm via depth) ───
        # Sanity-clamp landmark depths to face_dist ± 100 mm so a single
        # background pixel sneaking into safe_depth_at can't blow up the
        # world-mm reconstruction (e.g. mouth corner reading a wall behind).
        smile_mm = None
        brow_mm = None
        face_dist = None
        depth_frame = None

        SMILE_MAX_MM = 100.0   # peak distance between mouth corners; >100 mm is a face
        BROW_MAX_MM = 30.0     # vertical brow→eye distance per side; >30 mm impossible
        FACE_DEPTH_TOL_MM = 80.0  # accept landmark depth within ±80 mm of nose depth
        FACE_DIST_MAX_MM = 1500.0  # face cannot be >1.5 m away in clinical use;
                                   # FaceMesh sometimes hallucinates landmarks on
                                   # the wall when the face leaves the crop

        def _validated_depth(d, anchor):
            if d is None or np.isnan(d) or d <= 0:
                return None
            if anchor is None or np.isnan(anchor) or anchor <= 0:
                return float(d)
            if abs(d - anchor) > FACE_DEPTH_TOL_MM:
                return float(anchor)  # snap outlier to face plane
            return float(d)

        if depth_msg is not None and isinstance(depth_msg, dai.ImgFrame):
            depth_frame = depth_msg.getCvFrame()  # uint16 mm
            # Depth output is at stereo resolution (e.g. 640x400) while landmark
            # pixels are in RGB resolution (e.g. 1024x768). Scale pixel coords
            # from RGB → depth space before lookup.
            dh, dw = depth_frame.shape[:2]
            sx = dw / float(src_w)
            sy = dh / float(src_h)

            def _d_at(u_rgb, v_rgb, window=3):
                return safe_depth_at(depth_frame, u_rgb * sx, v_rgb * sy, window=window)

            # use nose depth as the "face distance" anchor
            from .biomarkers import NOSE_TIP
            nu = landmarks[NOSE_TIP].x * src_w
            nv = landmarks[NOSE_TIP].y * src_h
            face_dist = _d_at(nu, nv, window=4)
            # Sanity-clip: clinical use is < 1.5 m; anything beyond is the
            # FaceMesh hallucinating on the background wall.
            if face_dist is not None and not np.isnan(face_dist):
                if face_dist > FACE_DIST_MAX_MM or face_dist <= 0:
                    face_dist = None
            anchor = face_dist if (face_dist is not None and not np.isnan(face_dist)) else None

            # mouth corners in 3D mm
            ml_u = landmarks[MOUTH_LEFT].x * src_w
            ml_v = landmarks[MOUTH_LEFT].y * src_h
            mr_u = landmarks[MOUTH_RIGHT].x * src_w
            mr_v = landmarks[MOUTH_RIGHT].y * src_h
            d_ml = _validated_depth(_d_at(ml_u, ml_v), anchor)
            d_mr = _validated_depth(_d_at(mr_u, mr_v), anchor)
            p_ml = pixel_to_world_mm(ml_u, ml_v, d_ml, self._K) if d_ml else None
            p_mr = pixel_to_world_mm(mr_u, mr_v, d_mr, self._K) if d_mr else None
            cand_smile = euclid_mm(p_ml, p_mr)
            if cand_smile is not None and cand_smile <= SMILE_MAX_MM:
                smile_mm = cand_smile
                self.smile_amp.update(smile_mm, t)

            # brow lift = vertical distance brow_outer ↔ eye_top, average L/R
            bl_u = landmarks[BROW_OUTER_L].x * src_w
            bl_v = landmarks[BROW_OUTER_L].y * src_h
            el_u = landmarks[EYE_TOP_L].x * src_w
            el_v = landmarks[EYE_TOP_L].y * src_h
            br_u = landmarks[BROW_OUTER_R].x * src_w
            br_v = landmarks[BROW_OUTER_R].y * src_h
            er_u = landmarks[EYE_TOP_R].x * src_w
            er_v = landmarks[EYE_TOP_R].y * src_h
            d_bl = _validated_depth(_d_at(bl_u, bl_v), anchor)
            d_el = _validated_depth(_d_at(el_u, el_v), anchor)
            d_br = _validated_depth(_d_at(br_u, br_v), anchor)
            d_er = _validated_depth(_d_at(er_u, er_v), anchor)
            p_bl = pixel_to_world_mm(bl_u, bl_v, d_bl, self._K) if d_bl else None
            p_el = pixel_to_world_mm(el_u, el_v, d_el, self._K) if d_el else None
            p_br = pixel_to_world_mm(br_u, br_v, d_br, self._K) if d_br else None
            p_er = pixel_to_world_mm(er_u, er_v, d_er, self._K) if d_er else None
            d_l = euclid_mm(p_bl, p_el)
            d_r = euclid_mm(p_br, p_er)
            valid_l = d_l is not None and d_l <= BROW_MAX_MM
            valid_r = d_r is not None and d_r <= BROW_MAX_MM
            if valid_l and valid_r:
                brow_mm = (d_l + d_r) / 2.0
                self.brow_amp.update(brow_mm, t)
            elif valid_l:
                brow_mm = d_l
                self.brow_amp.update(brow_mm, t)
            elif valid_r:
                brow_mm = d_r
                self.brow_amp.update(brow_mm, t)

        metrics = {
            "t": round(t, 3),
            "timestamp_ms": int(time.time() * 1000),
            "frame_idx": self.frame_idx,
            "ear": round(ear, 4),
            "blink_rate": None if blink_rate is None else round(blink_rate, 2),
            "blink_calibrated": int(self.blink.is_calibrated),
            "smile_mm": None if smile_mm is None else round(smile_mm, 2),
            "smile_amplitude_mm": round(self.smile_amp.amplitude(), 2),
            "brow_mm": None if brow_mm is None else round(brow_mm, 2),
            "brow_amplitude_mm": round(self.brow_amp.amplitude(), 2),
            "asymmetry": round(asym, 4),
            "asymmetry_std": round(self.asym_amp.std(), 4),
            "gaze_dx": round(float(gx), 4),
            "gaze_dy": round(float(gy), 4),
            "gaze_x_std": round(self.gaze_x_amp.std(), 4),
            "gaze_y_std": round(self.gaze_y_amp.std(), 4),
            "face_distance_mm": None if face_dist is None or np.isnan(face_dist) else round(float(face_dist), 1),
            "tremor_chin_power_4_6hz": round(chin_t["band_power_pd"], 6),
            "tremor_chin_dominant_hz": round(chin_t["dom_freq"], 2),
            "tremor_chin_motion_power": round(chin_t["motion_power"], 6),
            "tremor_chin_lock_in": round(chin_t["lock_in_ratio"], 3),
            "tremor_chin_pd_likelihood": round(chin_t["pd_likelihood"], 1),
            "tremor_lip_power_4_6hz": round(lip_t["band_power_pd"], 6),
            "tremor_lip_dominant_hz": round(lip_t["dom_freq"], 2),
            "tremor_lip_motion_power": round(lip_t["motion_power"], 6),
            "tremor_lip_lock_in": round(lip_t["lock_in_ratio"], 3),
            "tremor_lip_pd_likelihood": round(lip_t["pd_likelihood"], 1),
            "ear_threshold": round(self.blink.last_threshold, 4),
            "head_yaw_deg": round(yaw_deg, 1),
            "off_axis": int(off_axis),
        }

        # ─── Lower-face landmark embedding (for ML training CSV) ───
        # Rationale: PD facial masking dominates the lower face (Bologna 2013).
        # Fixed whitelist of ~100 landmark indices (lips, nose, jaw, chin,
        # lower cheeks) so CSV columns stay stable across visits.
        # Coords are full-frame normalized [0,1] (post-rectification);
        # z is MediaPipe depth (relative, not metric).
        lower_face = []
        for i in LOWER_FACE_FIXED_INDICES:
            if i >= len(landmarks):
                continue
            kp = landmarks[i]
            lower_face.append({
                "i": int(i),
                "x": round(float(kp.x), 5),
                "y": round(float(kp.y), 5),
                "z": round(float(getattr(kp, "z", 0.0)), 5),
            })
        metrics["landmarks_lower"] = lower_face

        buf = dai.Buffer()
        buf.setData(list(json.dumps(metrics).encode("utf-8")))
        buf.setTimestamp(det_msg.getTimestamp())
        buf.setSequenceNum(det_msg.getSequenceNum())
        self.metrics_out.send(buf)
