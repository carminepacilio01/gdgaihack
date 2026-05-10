"""Convert a Markdown file to a brand-styled PDF.

Pipeline: markdown → HTML body → wrap in NeuroVista-themed CSS template →
Chrome headless print-to-pdf.

Usage:
    python3 scripts/md_to_pdf.py <input.md> <output.pdf> [--title "Title"]
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown


CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  @page {
    size: A4;
    margin: 22mm 18mm 20mm 18mm;
  }
  :root {
    --bg: #0A1230;
    --panel: #11193D;
    --border: rgba(64, 112, 220, 0.30);
    --text: #EAF4FF;
    --muted: #7A8FB8;
    --mono: #5C7CB8;
    --accent: #22DDFF;
    --accent-deep: #3D5BD9;
    --green: #00E5A0;
    --amber: #FFC857;
    --red: #FF5470;
    --font-sans: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
    --font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 10.5pt;
    line-height: 1.55;
    margin: 0; padding: 0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  body {
    padding: 0;
    background-image:
      linear-gradient(rgba(90, 143, 255, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(90, 143, 255, 0.04) 1px, transparent 1px);
    background-size: 40px 40px;
  }
  .page-header {
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px; margin-bottom: 24px;
  }
  .logo-mark { width: 28px; height: 28px; flex-shrink: 0; }
  .logo-text {
    font-weight: 600; font-size: 14pt; color: var(--text);
    letter-spacing: -0.01em;
  }
  .logo-sep { color: var(--mono); font-size: 11pt; margin: 0 4px; }
  .logo-sub { color: var(--muted); font-size: 11pt; }
  .doc-tag {
    margin-left: auto; font-family: var(--font-mono);
    color: var(--accent); font-size: 9pt;
    letter-spacing: 0.10em; text-transform: uppercase;
  }
  h1 {
    font-size: 22pt; font-weight: 600; letter-spacing: -0.02em;
    margin: 28px 0 14px; color: var(--text);
    page-break-after: avoid;
  }
  h2 {
    font-size: 15pt; font-weight: 600; letter-spacing: -0.01em;
    margin: 30px 0 10px; color: var(--text);
    padding-top: 14px;
    border-top: 1px solid var(--border);
    page-break-after: avoid;
  }
  h2:first-of-type { padding-top: 0; border-top: none; }
  h3 {
    font-size: 12pt; font-weight: 600; color: var(--accent);
    margin: 24px 0 8px;
    page-break-after: avoid;
  }
  h4 {
    font-size: 11pt; font-weight: 600; color: var(--text);
    margin: 18px 0 6px;
    page-break-after: avoid;
  }
  p {
    margin: 8px 0;
    color: var(--text);
  }
  ul, ol { margin: 6px 0 10px 22px; padding: 0; }
  li { margin: 3px 0; color: var(--text); }
  strong { color: var(--text); font-weight: 600; }
  em { color: var(--accent); font-style: normal; }
  hr {
    border: none; border-top: 1px solid var(--border);
    margin: 22px 0;
  }
  blockquote {
    margin: 12px 0; padding: 10px 16px;
    border-left: 2px solid var(--accent);
    background: rgba(34, 221, 255, 0.06);
    color: var(--text);
    font-style: italic;
  }
  code {
    font-family: var(--font-mono);
    font-size: 9.5pt;
    color: var(--accent);
    background: rgba(34, 221, 255, 0.08);
    padding: 1px 5px; border-radius: 2px;
  }
  pre {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 16px;
    overflow-x: auto;
    page-break-inside: avoid;
  }
  pre code {
    color: var(--text);
    background: transparent;
    padding: 0;
    font-size: 9pt;
    line-height: 1.5;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0 18px;
    font-size: 9.5pt;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    page-break-inside: avoid;
  }
  thead th {
    background: rgba(34, 221, 255, 0.06);
    color: var(--accent);
    font-family: var(--font-mono);
    font-weight: 500;
    font-size: 8.5pt;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  tbody td {
    padding: 8px 12px;
    border-top: 1px solid var(--border);
    vertical-align: top;
    color: var(--text);
  }
  tbody tr:nth-child(2n) { background: rgba(34, 221, 255, 0.02); }
  a { color: var(--accent); text-decoration: none; }
  /* Page footer */
  .page-footer {
    position: fixed; bottom: 0; left: 0; right: 0;
    padding: 10px 18mm;
    color: var(--mono);
    font-family: var(--font-mono);
    font-size: 8pt;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    display: flex; justify-content: space-between;
    border-top: 1px solid var(--border);
    background: var(--bg);
  }
</style>
</head>
<body>
  <div class="page-header">
    <svg class="logo-mark" viewBox="0 0 60 40" xmlns="http://www.w3.org/2000/svg">
      <circle cx="35" cy="20" r="20" fill="#0F1A45"/>
      <circle cx="14" cy="22" r="8" fill="#22DDFF"/>
      <circle cx="28" cy="18" r="9" fill="#5A8FFF"/>
      <circle cx="44" cy="18" r="10" fill="#3D5BD9"/>
    </svg>
    <span class="logo-text">NeuroVista</span>
    <span class="logo-sep">·</span>
    <span class="logo-sub">Parkinson Screening</span>
    <span class="doc-tag">__TITLE__</span>
  </div>
__BODY__
  <div class="page-footer">
    <span>NeuroVista · longitudinal screening · on-device · privacy-first</span>
    <span>GDG AI Hack 2026 · Track B · Luxonis OAK 4 D</span>
  </div>
</body>
</html>
"""


def find_chrome():
    for p in CHROME_PATHS:
        if Path(p).exists():
            return p
    raise SystemExit("Chrome / Chromium not found. Install Chrome to generate PDFs.")


def md_to_html(md_path: Path, title: str) -> str:
    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=[
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.toc",
            "markdown.extensions.attr_list",
            "markdown.extensions.smarty",
        ],
    )
    return (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__BODY__", html_body))


def html_to_pdf(html: str, out_pdf: Path):
    chrome = find_chrome()
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "doc.html"
        html_path.write_text(html, encoding="utf-8")
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--no-margins",
            f"--print-to-pdf={out_pdf}",
            f"file://{html_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not out_pdf.exists():
            print("Chrome stdout:", result.stdout[-500:])
            print("Chrome stderr:", result.stderr[-500:])
            raise SystemExit(f"PDF was not produced at {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_md", type=Path)
    ap.add_argument("output_pdf", type=Path)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    title = args.title or args.input_md.stem.upper().replace("_", " ")
    html = md_to_html(args.input_md, title)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    html_to_pdf(html, args.output_pdf)
    print(f"Wrote {args.output_pdf}  ({args.output_pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
