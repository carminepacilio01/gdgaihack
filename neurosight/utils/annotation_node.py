"""OverlayNode — paints a HUD with current biomarker values on the live feed.

Subscribes to BiomarkerExtractor.metrics_out (JSON-encoded dai.Buffer) and emits
ImgAnnotations to the visualizer.

Debug mode (--debug): also subscribes to the GatherData output and draws all
468 MediaPipe facial landmarks. Key biomarker landmarks (eyes / mouth / brows)
are highlighted in distinct colors so the viewer can see exactly which points
feed each metric.
"""
import json
import depthai as dai
from depthai_nodes import Keypoints
from depthai_nodes.utils import AnnotationHelper

from .biomarkers import (
    LEFT_EYE_EAR,
    RIGHT_EYE_EAR,
    MOUTH_LEFT,
    MOUTH_RIGHT,
    LIP_UPPER,
    LIP_LOWER,
    BROW_INNER_L,
    BROW_OUTER_L,
    BROW_INNER_R,
    BROW_OUTER_R,
    EYE_TOP_L,
    EYE_TOP_R,
    NOSE_BRIDGE,
    NOSE_TIP,
    CHIN,
)

# Biomarker landmark groups (indices into MediaPipe FaceMesh 468)
_EYE_INDICES = set(LEFT_EYE_EAR + RIGHT_EYE_EAR + [EYE_TOP_L, EYE_TOP_R])
_MOUTH_INDICES = {MOUTH_LEFT, MOUTH_RIGHT, LIP_UPPER, LIP_LOWER}
_BROW_INDICES = {BROW_INNER_L, BROW_OUTER_L, BROW_INNER_R, BROW_OUTER_R}
_TREMOR_INDICES = {CHIN, LIP_UPPER}
_AXIS_INDICES = {NOSE_BRIDGE, NOSE_TIP}

# ── Constellation mask (clinician-view "MISMATCH upper / lower" visual) ──
# Curated MediaPipe FaceMesh indices that trace the face shape sparsely.
# Upper half (forehead, brows, eye contours) — rendered cyan; theory says
# this region stays expressive in early PD ("signal stays").
# Lower half (cheeks, mouth, jaw, chin) — rendered deep-blue; this is where
# Parkinsonian facial masking onsets first ("masking onset").

_MASK_UPPER = [
    # Forehead / hairline
    10, 151, 9, 8, 168, 109, 67, 338, 297,
    # Brow ridge
    70, 63, 105, 66, 107, 55, 65,
    296, 334, 293, 300, 285, 295,
    # Eye outer/inner corners
    33, 133, 263, 362,
    # Cheekbone tops (still upper)
    234, 454,
]
_MASK_LOWER = [
    # Nose bridge → tip
    197, 195, 5, 4, 1, 19, 94,
    # Cheek lower
    50, 36, 205, 192, 138,
    280, 266, 425, 416, 367,
    # Mouth outer
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Mouth inner / lips center
    13, 14, 0, 12, 11, 15,
    # Jaw line
    172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365, 397, 288,
    # Chin
    175, 199, 200, 18,
]
_MASK_UPPER = sorted(set(_MASK_UPPER))
_MASK_LOWER = sorted(set(_MASK_LOWER))


def _color_for_blink(rate):
    if rate >= 15:
        return dai.Color(0.2, 0.9, 0.3, 1.0)  # green
    if rate >= 12:
        return dai.Color(0.9, 0.8, 0.2, 1.0)  # amber
    return dai.Color(0.95, 0.25, 0.25, 1.0)   # red


def _color_for_asym(score):
    if score < 0.05:
        return dai.Color(0.2, 0.9, 0.3, 1.0)
    if score < 0.10:
        return dai.Color(0.9, 0.8, 0.2, 1.0)
    return dai.Color(0.95, 0.25, 0.25, 1.0)


def _color_for_tremor_mu(milliunits):
    # Power threshold: 5.0 mU = TREMOR_NORMAL_POWER_MAX (5e-4)
    if milliunits < 5.0:
        return dai.Color(0.2, 0.9, 0.3, 1.0)
    if milliunits < 10.0:
        return dai.Color(0.9, 0.8, 0.2, 1.0)
    return dai.Color(0.95, 0.25, 0.25, 1.0)


def _color_for_likelihood(pct):
    """0-30 green (no signal), 30-60 amber (suspect), 60+ red (strong)."""
    if pct < 30.0:
        return dai.Color(0.2, 0.9, 0.3, 1.0)
    if pct < 60.0:
        return dai.Color(0.9, 0.8, 0.2, 1.0)
    return dai.Color(0.95, 0.25, 0.25, 1.0)


class OverlayNode(dai.node.HostNode):
    def __init__(self):
        super().__init__()
        self.overlay_out = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )
        self.debug = False
        self.show_hud = True
        self.clean_mask = False
        self.frame_idx = 0

    def build(self, metrics_out, gather_out=None, debug=False,
              show_hud=True, clean_mask=False):
        self.debug = bool(debug)
        self.show_hud = bool(show_hud)
        self.clean_mask = bool(clean_mask)
        if self.debug and gather_out is not None:
            self.link_args(metrics_out, gather_out)
        else:
            self.link_args(metrics_out)
        return self

    def process(self, metrics_buf, gather_msg=None):
        try:
            m = json.loads(bytes(metrics_buf.getData()).decode("utf-8"))
        except Exception:
            return
        self.frame_idx += 1

        helper = AnnotationHelper()
        white = dai.Color(1, 1, 1, 1)
        bg = dai.Color(0.0, 0.0, 0.0, 0.55)  # semi-transparent black plate

        # Compact left-edge HUD panel, fixed-width font feel via aligned padding.
        # Each row at y_step = 0.038, font size 16, padded to ~32 chars.
        y0 = 0.06
        y_step = 0.045
        x = 0.012
        size = 16

        def _row(label, value, color=white, idx=0):
            # Suppressed entirely when --no_hud is set (clinician portal view).
            if not self.show_hud:
                return
            helper.draw_text(
                text=f"{label:<12} {value}",
                position=(x, y0 + idx * y_step),
                color=color,
                background_color=bg,
                size=size,
            )

        blink_raw = m.get("blink_rate")
        blink_calibrated = bool(m.get("blink_calibrated", 1))
        blink = blink_raw if blink_raw is not None else 0
        smile = m.get("smile_amplitude_mm", 0) or 0
        brow = m.get("brow_amplitude_mm", 0) or 0
        asym = m.get("asymmetry", 0) or 0
        fd = m.get("face_distance_mm")
        tp = m.get("tremor_chin_power_4_6hz")
        tf = m.get("tremor_chin_dominant_hz")
        ear = m.get("ear", 0) or 0
        ear_thr = m.get("ear_threshold")
        yaw = m.get("head_yaw_deg", 0) or 0
        off_axis = bool(m.get("off_axis", 0))
        t_now = m.get("t", 0) or 0

        amber = dai.Color(0.9, 0.8, 0.2, 1.0)

        if blink_calibrated and blink_raw is not None:
            _row("Blink", f"{blink:5.1f} /min",
                 _color_for_blink(blink), idx=0)
        else:
            _row("Blink", "calibrating…", amber, idx=0)
        _row("Smile",      f"{smile:5.1f} mm",   white, idx=1)
        _row("Brow",       f"{brow:5.1f} mm",    white, idx=2)
        if off_axis:
            _row("Asymmetry",  f"{asym:.3f} (paused, off-axis)",
                 amber, idx=3)
        else:
            _row("Asymmetry",  f"{asym:.3f}",
                 _color_for_asym(asym), idx=3)
        _row("Face dist",  f"{fd:.0f} mm" if fd is not None else "—",
             white, idx=4)
        if tp is not None and tf is not None:
            likelihood = float(m.get("tremor_chin_pd_likelihood") or 0.0)
            lock = float(m.get("tremor_chin_lock_in") or 0.0) * 100.0
            tm = float(m.get("tremor_chin_motion_power") or 0.0) * 10000.0
            tag = ("PD" if likelihood >= 60 else
                   "?"  if likelihood >= 30 else
                   "—")
            _row("Jaw tremor",
                 f"{likelihood:5.1f}% @ {tf:.1f} Hz [{tag}]  (lock {lock:.0f}%, mot {tm:.0f})",
                 _color_for_likelihood(likelihood), idx=5)
        head_color = amber if off_axis else white
        head_tag = " (gate)" if off_axis else ""
        _row("Head yaw",   f"{yaw:+5.1f}°{head_tag}", head_color, idx=6)
        ear_txt = f"{ear:.3f}" + (f"  (thr {ear_thr:.3f})" if ear_thr else "")
        _row("EAR",        ear_txt, white, idx=7)
        _row("Elapsed",    f"{t_now:5.1f} s",   white, idx=8)

        # ─── Debug mode: draw all 468 landmarks, color-coded by biomarker group ───
        if self.debug and gather_msg is not None:
            det_msg = gather_msg.reference_data
            landmarks_list = gather_msg.gathered
            if (landmarks_list and isinstance(landmarks_list[0], Keypoints)
                    and det_msg is not None and det_msg.detections):
                raw_kps = landmarks_list[0].keypoints
                rr = det_msg.detections[0].rotated_rect
                bcx, bcy = rr.center.x, rr.center.y
                bw, bh = rr.size.width, rr.size.height
                kps = [
                    (float(bcx + (kp.x - 0.5) * bw),
                     float(bcy + (kp.y - 0.5) * bh))
                    for kp in raw_kps
                ]

                # Color palette
                grey = dai.Color(0.5, 0.55, 0.6, 0.9)
                # Pitch-deck "MISMATCH" mask: upper cyan, lower deep-blue
                upper_c = dai.Color(0.30, 0.83, 0.88, 0.95)   # #4dd3e0
                lower_c = dai.Color(0.27, 0.49, 0.95, 0.95)   # #457bf2
                # Salient highlights still distinguished
                eye_c = dai.Color(0.55, 0.92, 1.00, 1.0)
                mouth_c = dai.Color(0.55, 0.78, 1.00, 1.0)
                brow_c = dai.Color(0.55, 0.92, 1.00, 1.0)
                tremor_c = dai.Color(0.27, 0.49, 0.95, 1.0)
                axis_c = dai.Color(1.0, 1.0, 1.0, 1.0)

                special = (_EYE_INDICES | _MOUTH_INDICES | _BROW_INDICES
                           | _TREMOR_INDICES | _AXIS_INDICES)

                # Subtle, slow constellation pulse — three phase-shifted sines so
                # different point groups breathe at different times. Looks alive
                # without distracting motion.
                import math
                f = self.frame_idx
                pulse_a = 1.0 + 0.18 * math.sin(f * 0.045)
                pulse_b = 1.0 + 0.18 * math.sin(f * 0.060 + 1.7)
                pulse_c = 1.0 + 0.22 * math.sin(f * 0.080 + 3.1)

                if self.clean_mask:
                    # ── Constellation mask (innovation view) ──
                    # 1) upper-face curated points in cyan, multi-size
                    #    → split in two thickness classes for "depth" feeling
                    upper_idx = [i for i in _MASK_UPPER if i not in special]
                    upper_small = [kps[i] for i in upper_idx[::2] if i < len(kps)]
                    upper_big   = [kps[i] for i in upper_idx[1::2] if i < len(kps)]
                    if upper_small:
                        helper.draw_points(upper_small, color=upper_c,
                                           thickness=1.6 * pulse_a)
                    if upper_big:
                        helper.draw_points(upper_big, color=upper_c,
                                           thickness=2.6 * pulse_b)

                    # 2) lower-face curated points in deep-blue, multi-size
                    lower_idx = [i for i in _MASK_LOWER if i not in special]
                    lower_small = [kps[i] for i in lower_idx[::2] if i < len(kps)]
                    lower_big   = [kps[i] for i in lower_idx[1::2] if i < len(kps)]
                    if lower_small:
                        helper.draw_points(lower_small, color=lower_c,
                                           thickness=1.7 * pulse_b)
                    if lower_big:
                        helper.draw_points(lower_big, color=lower_c,
                                           thickness=2.8 * pulse_c)

                    # 3) salient clinical points on top, slightly larger,
                    #    pulsing distinctly so the eye is drawn to them
                    def _grp(idx_set):
                        return [kps[i] for i in idx_set if i < len(kps)]
                    for pts, color, thick in [
                        (_grp(_EYE_INDICES),    eye_c,    2.6 * pulse_a),
                        (_grp(_BROW_INDICES),   brow_c,   2.6 * pulse_a),
                        (_grp(_MOUTH_INDICES),  mouth_c,  3.0 * pulse_b),
                        (_grp(_TREMOR_INDICES), tremor_c, 3.4 * pulse_c),
                        (_grp(_AXIS_INDICES),   axis_c,   3.0),  # axis steady
                    ]:
                        if pts:
                            helper.draw_points(pts, color=color, thickness=thick)
                else:
                    # ── Dense debug view (programmer mode) ──
                    # Render all 463 background points + salient highlights as before
                    grey_pts = [pt for i, pt in enumerate(kps) if i not in special]
                    if grey_pts:
                        helper.draw_points(grey_pts, color=grey, thickness=3.0)

                    def _grp(idx_set):
                        return [kps[i] for i in idx_set if i < len(kps)]
                    for pts, color, thick in [
                        (_grp(_EYE_INDICES),    eye_c,    5.0 * pulse_a),
                        (_grp(_MOUTH_INDICES),  mouth_c,  5.0 * pulse_b),
                        (_grp(_BROW_INDICES),   brow_c,   5.0 * pulse_a),
                        (_grp(_TREMOR_INDICES), tremor_c, 6.0 * pulse_c),
                        (_grp(_AXIS_INDICES),   axis_c,   5.0),
                    ]:
                        if pts:
                            helper.draw_points(pts, color=color, thickness=thick)

                # Legend bottom-right with background plate (only when HUD is on)
                if self.show_hud:
                    helper.draw_text(
                        "DEBUG 468 landmarks",
                        position=(0.66, 0.86), color=white, background_color=bg, size=14,
                    )
                    helper.draw_text(
                        "blue=eyes  pink=mouth",
                        position=(0.66, 0.91), color=white, background_color=bg, size=12,
                    )
                    helper.draw_text(
                        "yellow=brows  green=tremor",
                        position=(0.66, 0.95), color=white, background_color=bg, size=12,
                    )

        ann = helper.build(
            timestamp=metrics_buf.getTimestamp(),
            sequence_num=metrics_buf.getSequenceNum(),
        )
        self.overlay_out.send(ann)
