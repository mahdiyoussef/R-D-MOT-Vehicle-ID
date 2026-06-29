#!/usr/bin/env python3
import base64
import io
import os
import re
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD   = os.path.join(BASE, "docs", "rapport_avancement.md")
HTML = os.path.join(BASE, "docs", "rapport_avancement.html")
PDF  = os.path.join(BASE, "docs", "rapport_avancement.pdf")
WEASY = "/home/youssef/radioconda/bin/weasyprint"

def render_math_to_b64(latex: str, display: bool = False) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fontsize = 13 if display else 11
    fig_w = 9.0 if display else 4.0
    expr = f"${latex}$"
    fig = plt.figure(figsize=(fig_w, 0.6))
    fig.patch.set_alpha(0.0)
    fig.text(0.5 if display else 0.02, 0.5, expr, ha="center" if display else "left", va="center", fontsize=fontsize, color="#1E1B4B", usetex=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160, transparent=True, pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def replace_math(md_text: str) -> str:
    def block_sub(m):
        latex = m.group(1).strip()
        try:
            b64 = render_math_to_b64(latex, display=True)
            return f'\n<div style="text-align:center;margin:14px 0;"><img src="data:image/png;base64,{b64}" alt="{latex}" style="max-height:60px;"></div>\n'
        except: return f"\n<pre>{latex}</pre>\n"
    md_text = re.sub(r"\$\$(.*?)\$\$", block_sub, md_text, flags=re.DOTALL)
    def inline_sub(m):
        latex = m.group(1).strip()
        try:
            b64 = render_math_to_b64(latex, display=False)
            return f'<img src="data:image/png;base64,{b64}" alt="{latex}" style="height:1.2em;vertical-align:middle;">'
        except: return f"<code>{latex}</code>"
    md_text = re.sub(r"\$([^\$\n]+?)\$", inline_sub, md_text)
    return md_text

with open(MD, encoding="utf-8") as f:
    md_text = replace_math(f.read())

import markdown
body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

css = """
@page { size: A4; margin: 22mm 18mm 24mm 18mm;
  @bottom-center { content: counter(page) " / " counter(pages);
    font-family: Arial, sans-serif; font-size: 8.5pt; color: #94A3B8; } }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt; line-height: 1.72; color: #1E1B4B; }
h1 { font-size: 18pt; font-weight: 700; color: #4338CA; border-bottom: 4px solid #4338CA; padding-bottom: 10px; margin: 0 0 26px; page-break-after: avoid; text-align: center; }
h2 { font-size: 14pt; font-weight: 700; color: #4338CA; border-left: 5px solid #4338CA; padding-left: 12px; margin: 28px 0 10px; page-break-after: avoid; }
p { margin: 0 0 12px; }
table { width: 100%; border-collapse: collapse; font-size: 10pt; margin: 14px 0 20px; page-break-inside: avoid; }
thead tr { background: #4338CA; color: #fff; }
thead th { padding: 8px 12px; font-weight: 600; text-align: left; }
tbody tr:nth-child(even) { background: #F8F9FF; }
tbody td { padding: 6px 12px; border-bottom: 1px solid #E2E8F0; vertical-align: top; }
ul, ol { padding-left: 22px; margin: 6px 0 12px; }
li { margin-bottom: 6px; }
hr { border: none; border-top: 2px solid #E2E8F0; margin: 24px 0; }
strong { color: #312E81; }
"""

html = f'<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8"><title>Rapport Avancement</title><style>{css}</style></head><body>{body}</body></html>'
with open(HTML, "w", encoding="utf-8") as f:
    f.write(html)

r = subprocess.run([WEASY, HTML, PDF], capture_output=True, text=True)
if r.returncode == 0:
    print(f"✅ PDF generated: {PDF} ({os.path.getsize(PDF)/1024:.1f} KB)")
else:
    print("ERR:", r.stderr[-800:])
    sys.exit(1)
