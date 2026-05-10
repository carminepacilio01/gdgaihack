"""Build a single-page clinician PDF report for a patient.

Each card shows: latest value, baseline (first visit), delta %, and a sparkline
trace of all visits. Footer carries the legal disclaimer.
"""
from datetime import datetime
from pathlib import Path
import io
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas


# source: "metrics" → v["metrics"][key]["mean"]; "voice" → v["voice"][key]
METRICS = [
    # facial / on-device
    ("metrics", "blink_rate",               "Blink rate",       "/min", (12, 22),  True),
    ("metrics", "smile_amplitude_mm",       "Smile amplitude",  "mm",   (35, 60),  True),
    ("metrics", "brow_amplitude_mm",        "Brow amplitude",   "mm",   (3, 12),   True),
    ("metrics", "asymmetry",                "Facial asymmetry", "",     (0, 0.05), False),
    ("metrics", "gaze_x_std",               "Gaze stability X", "",     (0, 0.04), False),
    ("metrics", "tremor_chin_power_4_6hz",  "Jaw tremor 4-6Hz", "",     (0, 5e-4), False),
    # voice / host-side, local Whisper+Praat
    ("voice",   "jitter_local_pct",  "Jitter (local)",      "%",   (0, 1.04),  False),
    ("voice",   "shimmer_local_pct", "Shimmer (local)",     "%",   (0, 3.81),  False),
    ("voice",   "hnr_db",            "Harmonics-to-Noise",  "dB",  (20, 35),   True),
    ("voice",   "speech_rate_wpm",   "Speech rate",         "wpm", (130, 180), True),
]

REPORT_DIR = Path("data/reports")


def _trend(visits, source, key):
    out = []
    for v in visits:
        if source == "metrics":
            out.append((v.get("metrics", {}).get(key) or {}).get("mean"))
        else:  # voice (flat dict)
            out.append((v.get("voice", {}) or {}).get(key))
    return out


def _delta_pct(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2 or vals[0] == 0:
        return None
    return (vals[-1] - vals[0]) / abs(vals[0]) * 100


def _draw_sparkline(c, x, y, w, h, values, color):
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return
    vmin, vmax = min(valid), max(valid)
    span = max(vmax - vmin, 1e-6)
    step = w / max(len(values) - 1, 1)
    pts = []
    for i, v in enumerate(values):
        if v is None:
            continue
        nx = x + i * step
        ny = y + (v - vmin) / span * h
        pts.append((nx, ny))
    c.setStrokeColor(color)
    c.setLineWidth(1.2)
    p = c.beginPath()
    p.moveTo(*pts[0])
    for pt in pts[1:]:
        p.lineTo(*pt)
    c.drawPath(p, stroke=1, fill=0)


def build_pdf_report(patient_id, visits):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{patient_id}_{datetime.now():%Y%m%d_%H%M}.pdf"

    c = canvas.Canvas(str(out_path), pagesize=A4)
    pw, ph = A4

    # Header bar
    c.setFillColor(HexColor("#0d1117"))
    c.rect(0, ph - 22 * mm, pw, 22 * mm, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(15 * mm, ph - 13 * mm, "NeuroVista — Longitudinal Screening Report")
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, ph - 18 * mm, "Facial biomarker tracking for neurodegenerative disease screening")

    # Patient block
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 12)
    y = ph - 32 * mm
    c.drawString(15 * mm, y, f"Patient: {patient_id}")
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, y - 5 * mm,
                 f"Visits recorded: {len(visits)}    First: {visits[0].get('visit_id')}    Latest: {visits[-1].get('visit_id')}")
    c.drawString(15 * mm, y - 10 * mm, f"Report generated: {datetime.now():%Y-%m-%d %H:%M}")

    # Metric cards: 2 cols × 5 rows (10 cards total)
    card_w = 90 * mm
    card_h = 32 * mm
    margin_x = 15 * mm
    margin_y = ph - 50 * mm
    gap_x = 5 * mm
    gap_y = 4 * mm

    def _fmt_value(v):
        if v is None:
            return "–"
        if abs(v) < 0.01:
            return f"{v:.2e}"
        return f"{v:.2f}"

    def _fmt_range(lo, hi, unit):
        def f(x):
            return f"{x:.0e}" if abs(x) < 0.01 and x != 0 else f"{x}"
        return f"normal: {f(lo)}–{f(hi)} {unit}".strip()

    for i, (source, key, label, unit, rng, lower_is_worse) in enumerate(METRICS):
        col = i % 2
        row = i // 2
        cx = margin_x + col * (card_w + gap_x)
        cy = margin_y - row * (card_h + gap_y) - card_h

        c.setStrokeColor(HexColor("#30363d"))
        c.setFillColor(HexColor("#f6f8fa"))
        c.roundRect(cx, cy, card_w, card_h, 4, fill=1, stroke=1)

        values = _trend(visits, source, key)
        latest = next((v for v in reversed(values) if v is not None), None)
        delta = _delta_pct(values)

        # source tag
        c.setFillColor(HexColor("#0969da") if source == "metrics" else HexColor("#bf3989"))
        c.setFont("Helvetica-Bold", 7)
        tag = "ON-DEVICE" if source == "metrics" else "VOICE (local)"
        c.drawString(cx + card_w - 28 * mm, cy + card_h - 5 * mm, tag)

        c.setFillColor(HexColor("#7d8590"))
        c.setFont("Helvetica", 8)
        c.drawString(cx + 4 * mm, cy + card_h - 5 * mm, label.upper())

        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(cx + 4 * mm, cy + card_h - 12 * mm, _fmt_value(latest))
        c.setFont("Helvetica", 9)
        c.drawString(cx + 4 * mm + 22 * mm, cy + card_h - 12 * mm, unit)

        if delta is not None:
            worsening = (delta < 0) if lower_is_worse else (delta > 0)
            color = HexColor("#cf222e") if worsening else HexColor("#1a7f37")
            arrow = "↓" if delta < 0 else "↑"
            c.setFillColor(color)
            c.setFont("Helvetica", 8)
            c.drawString(cx + 4 * mm, cy + card_h - 17 * mm,
                         f"{arrow} {abs(delta):.1f}% vs baseline")

        # Range hint
        c.setFillColor(HexColor("#7d8590"))
        c.setFont("Helvetica-Oblique", 7)
        c.drawString(cx + 4 * mm, cy + 3 * mm, _fmt_range(rng[0], rng[1], unit))

        # Sparkline
        _draw_sparkline(c, cx + 32 * mm, cy + 4 * mm, card_w - 36 * mm, 14 * mm,
                        values, HexColor("#58a6ff"))

    # Footer disclaimer
    c.setFillColor(HexColor("#fff8c5"))
    c.rect(0, 0, pw, 22 * mm, fill=1, stroke=0)
    c.setFillColor(HexColor("#7d4e00"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(15 * mm, 15 * mm,
                 "DISCLAIMER — Research / screening prototype. Not a medical device.")
    c.setFont("Helvetica", 8)
    c.drawString(15 * mm, 11 * mm,
                 "NeuroVista does NOT provide a medical diagnosis. Clinical interpretation requires a licensed neurologist.")
    c.drawString(15 * mm, 7 * mm,
                 "Refs: Karson 1983; Bologna et al. Brain 2013; Djaldetti 2006; Rusz et al. 2011 (jitter/shimmer); Goberman 2002 (HNR); Skodda 2011 (speech rate).")
    c.drawString(15 * mm, 3 * mm,
                 "Facial biomarkers computed on-device (OAK 4 D). Voice features computed locally (no audio uploaded). No patient data left the host.")

    c.showPage()
    c.save()
    return str(out_path)
