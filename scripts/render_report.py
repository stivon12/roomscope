"""Render docs/REPORT.md to docs/REPORT.pdf (A4) with headless Chrome: python scripts/render_report.py"""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """@page { size: A4; margin: 14mm; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.32; color: #111; }
h1 { font-size: 16pt; margin: 0 0 6px; } h2 { font-size: 12pt; margin: 12px 0 4px; border-bottom: 1px solid #ccc; }
table { border-collapse: collapse; width: 100%; margin: 5px 0 8px; font-size: 8.4pt; }
th, td { border: 1px solid #bbb; padding: 2px 4px; vertical-align: top; text-align: left; } th { background: #f0f0f0; }
code { font-size: 8.4pt; background: #f4f4f4; padding: 0 2px; } p, li { margin: 2px 0; } ul, ol { margin: 2px 0 4px 18px; padding: 0; }"""


def prepare(md: str) -> str:
    """Python-Markdown wants a blank line before a list and 4-space nesting; the source uses GitHub style."""
    out = []
    for line in md.splitlines():
        m = re.match(r"^( +)([-*]|\d+\.) ", line)
        if m:
            line = " " * (len(m.group(1)) * 2) + line.lstrip()
        if re.match(r"^\s*([-*]|\d+\.) ", line) and out and out[-1].strip() and not re.match(r"^\s*([-*]|\d+\.) ", out[-1]) \
                and not out[-1].startswith(" "):
            out.append("")
        out.append(line)
    return "\n".join(out)


html = markdown.markdown(prepare((ROOT / "docs/REPORT.md").read_text()), extensions=["tables"])
(ROOT / "docs/REPORT.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>roomscope technical report</title>"
                                       f"<style>{CSS}</style></head><body>{html}</body></html>")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={ROOT / 'docs/REPORT.pdf'}", str(ROOT / "docs/REPORT.html")], check=True,
               stderr=subprocess.DEVNULL)
print("wrote docs/REPORT.pdf")
