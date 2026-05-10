"""Render a ScreeningReport as a clinician-friendly text output.

The agent's structured output (`ScreeningReport`) is good for machines.
Doctors want prose with numbers next to claims. This module produces
that view, embedding both the LLM's interpretation and the raw upstream
metrics from the input payload.
"""
from __future__ import annotations

import textwrap

from .input_schema import KnowledgePayload
from .schemas import Confidence, RiskLevel, ScreeningReport, SignName


SEP = "─" * 72


RISK_BADGES: dict[RiskLevel, tuple[str, str]] = {
    RiskLevel.LOW:        ("🟢", "LOW"),
    RiskLevel.BORDERLINE: ("🟡", "BORDERLINE"),
    RiskLevel.ELEVATED:   ("🔴", "ELEVATED"),
}

CONFIDENCE_LABELS: dict[Confidence, str] = {
    Confidence.LOW:      "low",
    Confidence.MODERATE: "moderate",
    Confidence.HIGH:     "high",
}

SIGN_LABELS: dict[SignName, str] = {
    SignName.HYPOMIMIA:              "Hypomimia (facial masking)",
    SignName.JAW_TREMOR:             "Jaw rest tremor",
    SignName.REDUCED_BLINK_RATE:     "Reduced blink rate",
    SignName.MOUTH_CORNER_ASYMMETRY: "Mouth-corner asymmetry",
    SignName.EYELID_REDUCED_MOTION:  "Reduced eyelid motion",
}

SIDE_LABELS = {
    "left":      "left",
    "right":     "right",
    "bilateral": "bilateral",
    None:        "—",
}


def _wrap(text: str, width: int = 68, indent: str = "  ") -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        lines.extend(indent + ln for ln in wrapped)
    return lines


def _section(title: str) -> list[str]:
    return [SEP, f"  {title}", SEP]


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return ""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " · ".join(parts)


def render_report(report: ScreeningReport, payload: KnowledgePayload) -> str:
    """Build a multi-line clinician-facing report."""
    out: list[str] = []

    # ── Header ─────────────────────────────────────────────────────────
    badge, risk_label = RISK_BADGES[report.overall_risk_level]
    out.append(SEP)
    out.append("  PARKINSON'S SCREENING REPORT — Face-only")
    out.append(SEP)
    out.append(f"  Patient    : {payload.patient_id}")
    out.append(f"  Session    : {payload.session_id}")
    if payload.captured_at:
        out.append(f"  Captured   : {payload.captured_at}")
    if payload.duration_s and payload.n_frames and payload.fps:
        out.append(
            f"  Capture    : {payload.duration_s:.1f}s · "
            f"{payload.n_frames} frames · {payload.fps:.1f} fps"
        )
    meta_bits = []
    if payload.metadata.age is not None:
        meta_bits.append(f"age {payload.metadata.age:g}")
    if payload.metadata.sex is not None:
        meta_bits.append(f"sex {payload.metadata.sex}")
    if payload.metadata.ground_truth_label is not None:
        gt = "PD" if payload.metadata.ground_truth_label == 1 else "control"
        meta_bits.append(f"known label: {gt}")
    if meta_bits:
        out.append(f"  Metadata   : {' · '.join(meta_bits)}")
    out.append("")
    out.append(f"  {badge}  OVERALL RISK:  {risk_label}")
    if report.asymmetry_detected:
        out.append("  ⚠️   Left/right asymmetry detected")
    out.append("")

    # ── Motor signs ────────────────────────────────────────────────────
    out += _section("MOTOR SIGNS ASSESSED")
    if not report.motor_signs:
        out.append("  No motor signs assessed.")
        out.append("")
    else:
        for sign in report.motor_signs:
            mark = "✅" if sign.detected else "➖"
            label = SIGN_LABELS.get(sign.name, sign.name.value)
            conf = CONFIDENCE_LABELS[sign.confidence]
            side = SIDE_LABELS.get(sign.side, sign.side or "—")
            sev = f"{sign.severity}/4" if sign.severity is not None else "—"
            out.append(f"  {mark} {label}")
            out.append(f"     Detected: {'yes' if sign.detected else 'no'} · "
                       f"side: {side} · severity: {sev} · confidence: {conf}")
            metrics = _format_metrics(sign.key_metrics)
            if metrics:
                out.append(f"     Supporting metrics: {metrics}")
            if sign.rationale:
                out.append("     Rationale:")
                out.extend(_wrap(sign.rationale, width=64, indent="       "))
            out.append("")

    # ── Findings ───────────────────────────────────────────────────────
    if report.flagged_findings:
        out += _section("FLAGGED FINDINGS")
        for f in report.flagged_findings:
            out.append(f"  • {f}")
        out.append("")

    # ── Follow-up ──────────────────────────────────────────────────────
    if report.recommended_followup:
        out += _section("RECOMMENDED FOLLOW-UP")
        for r in report.recommended_followup:
            out.append(f"  • {r}")
        out.append("")

    # ── Clinician notes ────────────────────────────────────────────────
    if report.clinician_notes:
        out += _section("CLINICIAN NOTES")
        out.extend(_wrap(report.clinician_notes, width=68, indent="  "))
        out.append("")

    # ── Quality issues ─────────────────────────────────────────────────
    if report.quality_issues:
        out += _section("DATA-QUALITY CAVEATS")
        for q in report.quality_issues:
            out.append(f"  ⚠️  {q}")
        out.append("")

    # ── Raw numbers from upstream ──────────────────────────────────────
    out += _section("ANALYZED DATA (from upstream model)")
    cf = payload.clinical_features

    if cf.regional_motion and cf.regional_motion.valid:
        rm = cf.regional_motion
        if rm.composite_expressivity_score is not None:
            out.append(
                f"  Composite expressivity score: "
                f"{rm.composite_expressivity_score:.4f}"
            )
            out.append("    (lower = more reduced expressivity)")
        out.append("    Per-region range of motion (with applied clinical weights):")
        for region, stats in (rm.per_region or {}).items():
            out.append(
                f"      · {region:14s}  "
                f"RoM={stats.range_of_motion:.4f}  "
                f"n_landmarks={stats.available_landmarks or '?'}  "
                f"weight={stats.weight}"
            )
        out.append("")
    elif cf.regional_motion:
        out.append(f"  Regional motion: invalid ({cf.regional_motion.reason})")
        out.append("")

    if cf.jaw_tremor and cf.jaw_tremor.valid:
        jt = cf.jaw_tremor
        in_band = "in parkinsonian band 4–6 Hz ✓" \
            if jt.in_parkinsonian_range_4_6hz else "outside parkinsonian band ✗"
        out.append("  Jaw tremor (FFT 3–7 Hz):")
        if jt.dominant_frequency_hz is not None:
            out.append(f"    Dominant frequency: {jt.dominant_frequency_hz:.2f} Hz "
                       f"({in_band})")
        if jt.in_band_fraction_of_total is not None:
            out.append(f"    In-band power fraction: "
                       f"{jt.in_band_fraction_of_total:.1%}")
        if jt.spectral_peakedness is not None:
            out.append(f"    Spectral peakedness: {jt.spectral_peakedness:.2f}")
        if jt.anchor_used:
            out.append(f"    Anchor used: {jt.anchor_used}")
        out.append("")
    elif cf.jaw_tremor:
        out.append(f"  Jaw tremor: invalid ({cf.jaw_tremor.reason})")
        out.append("")

    if cf.mouth_asymmetry and cf.mouth_asymmetry.valid:
        ma = cf.mouth_asymmetry
        out.append("  Mouth-corner asymmetry (landmarks 61 vs 291):")
        if ma.asymmetry_ratio is not None:
            out.append(f"    Asymmetry ratio: {ma.asymmetry_ratio:.1%}")
        if ma.less_mobile_side:
            out.append(
                f"    Less mobile side: "
                f"{SIDE_LABELS.get(ma.less_mobile_side, ma.less_mobile_side)}"
            )
        if ma.rom_left is not None and ma.rom_right is not None:
            out.append(
                f"    RoM left: {ma.rom_left:.4f}  ·  "
                f"RoM right: {ma.rom_right:.4f}"
            )
        out.append("")
    elif cf.mouth_asymmetry:
        out.append(f"  Mouth asymmetry: invalid ({cf.mouth_asymmetry.reason})")
        out.append("")

    if payload.model_inference and payload.model_inference.pd_probability is not None:
        mi = payload.model_inference
        out.append("  Upstream classifier output:")
        out.append(f"    Model: {mi.model_name} v{mi.version or '?'}")
        out.append(f"    PD probability: {mi.pd_probability:.1%}")
        if mi.n_windows_analyzed:
            out.append(f"    Windows analyzed: {mi.n_windows_analyzed}")
        out.append("")

    # ── Limitations ────────────────────────────────────────────────────
    if payload.quality and payload.quality.missing_modalities:
        out += _section("DATA LIMITATIONS")
        out.append("  The following modalities were NOT measured by OAK and")
        out.append("  therefore were not assessed by the agent:")
        for m in payload.quality.missing_modalities:
            out.append(f"    · {m}")
        out.append("")

    # ── Disclaimer ─────────────────────────────────────────────────────
    out.append(SEP)
    out.append(f"  ⚠️  {report.disclaimer}")
    out.append(SEP)

    return "\n".join(out)
