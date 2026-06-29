#!/usr/bin/env python3
"""
docs/gen_pdf.py
───────────────
Converts docs/rapport_v3_v4.md → docs/rapport_pipeline.pdf

Math rendering: $...$ and $$...$$ blocks are rendered to PNG via
matplotlib mathtext and embedded as base64 inline images, so
WeasyPrint renders them correctly without LaTeX or JS.
"""
import base64
import io
import os
import re
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD   = os.path.join(BASE, "docs", "rapport_v3_v4.md")
HTML = os.path.join(BASE, "docs", "rapport_pipeline.html")
PDF  = os.path.join(BASE, "docs", "rapport_pipeline.pdf")
WEASY = "/home/youssef/radioconda/bin/weasyprint"

# ── Math rendering via matplotlib ─────────────────────────────────────────────
def render_math_to_b64(latex: str, display: bool = False) -> str:
    """Render a LaTeX math string to a base64-encoded PNG via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fontsize  = 13 if display else 11
    fig_w     = 7.0 if display else 4.0

    # matplotlib mathtext uses $ delimiters
    expr = f"${latex}$"

    fig = plt.figure(figsize=(fig_w, 0.6))
    fig.patch.set_alpha(0.0)
    fig.text(
        0.5 if display else 0.02,
        0.5,
        expr,
        ha="center" if display else "left",
        va="center",
        fontsize=fontsize,
        color="#1E1B4B",
        usetex=False,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160,
                transparent=True, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def replace_math(md_text: str) -> str:
    """Replace $$...$$ and $...$ with inline <img> tags."""

    # ── Block math: $$...$$ ───────────────────────────────────────────────────
    def block_sub(m):
        latex = m.group(1).strip()
        try:
            b64 = render_math_to_b64(latex, display=True)
            return (
                f'\n<div style="text-align:center;margin:14px 0;">'
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{latex}" style="max-height:60px;"></div>\n'
            )
        except Exception as e:
            print(f"  [math WARN] block render failed: {e}")
            return f"\n<pre>{latex}</pre>\n"

    md_text = re.sub(r"\$\$(.*?)\$\$", block_sub, md_text, flags=re.DOTALL)

    # ── Inline math: $...$ ────────────────────────────────────────────────────
    def inline_sub(m):
        latex = m.group(1).strip()
        try:
            b64 = render_math_to_b64(latex, display=False)
            return (
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{latex}" style="height:1.2em;vertical-align:middle;">'
            )
        except Exception as e:
            print(f"  [math WARN] inline render failed: {e}")
            return f"<code>{latex}</code>"

    md_text = re.sub(r"\$([^\$\n]+?)\$", inline_sub, md_text)
    return md_text


# ── Main ──────────────────────────────────────────────────────────────────────
print("Reading markdown...")
with open(MD, encoding="utf-8") as f:
    md_text = f.read()

print("Rendering math equations...")
md_text = replace_math(md_text)

print("Converting markdown to HTML...")
import markdown
body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

css = """
@page {
  size: A4;
  margin: 22mm 18mm 24mm 18mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: Arial, sans-serif;
    font-size: 8.5pt;
    color: #94A3B8;
  }
}
* { box-sizing: border-box; }
body {
  font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.72;
  color: #1E1B4B;
}
h1 {
  font-size: 17pt; font-weight: 700; color: #4338CA;
  border-bottom: 4px solid #4338CA;
  padding-bottom: 10px; margin: 0 0 26px;
  page-break-after: avoid;
}
h2 {
  font-size: 13pt; font-weight: 700; color: #4338CA;
  border-left: 5px solid #4338CA; padding-left: 12px;
  margin: 30px 0 10px; page-break-after: avoid;
}
h3 {
  font-size: 11pt; font-weight: 600; color: #6D28D9;
  margin: 20px 0 6px; page-break-after: avoid;
}
h4 {
  font-size: 10pt; font-weight: 600; color: #64748B;
  text-transform: uppercase; letter-spacing: .06em;
  margin: 14px 0 4px; page-break-after: avoid;
}
p { margin: 0 0 10px; }
table {
  width: 100%; border-collapse: collapse;
  font-size: 9.5pt; margin: 14px 0 20px;
  page-break-inside: avoid;
}
thead tr { background: #4338CA; color: #fff; }
thead th { padding: 8px 12px; font-weight: 600; text-align: left; }
tbody tr:nth-child(even) { background: #F8F9FF; }
tbody td { padding: 6px 12px; border-bottom: 1px solid #E2E8F0; vertical-align: top; }
pre {
  background: #F1F5F9; border-left: 4px solid #4338CA;
  border-radius: 6px; padding: 12px 16px;
  font-size: 8.5pt; line-height: 1.5;
  page-break-inside: avoid; margin: 12px 0;
  white-space: pre-wrap; word-break: break-word;
}
code {
  font-family: 'Courier New', monospace; font-size: 8.5pt;
  background: #EEF2FF; color: #3730A3;
  padding: 1px 5px; border-radius: 3px;
}
pre code { background: none; color: inherit; padding: 0; }
ul, ol { padding-left: 22px; margin: 6px 0 12px; }
li { margin-bottom: 4px; }
hr { border: none; border-top: 2px solid #E2E8F0; margin: 24px 0; }
strong { color: #312E81; }
img { max-width: 100%; page-break-inside: avoid; }
"""

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Pipeline Re-ID Véhicules</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""

with open(HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"HTML written: {HTML}")

print("Running WeasyPrint...")
r = subprocess.run([WEASY, HTML, PDF], capture_output=True, text=True)
if r.returncode == 0:
    size = os.path.getsize(PDF) / 1024
    print(f"\n✅ PDF generated: {PDF}  ({size:.1f} KB)")
else:
    print("❌ WeasyPrint error:\n", r.stderr[-2000:])
    sys.exit(1)
