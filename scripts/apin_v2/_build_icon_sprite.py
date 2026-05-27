"""Build helper · Phase 8.G · extract the canonical icon sprite from
ui_template.html into a standalone file so the Console can inline-inject it
into pages that don't include ui_template.html.

Why a standalone file (not just constants in apin_server.py): keeps the
sprite editable as a normal .svg file, and the server's startup helper can
re-read it (with mtime invalidation) just like the static JS files. Single
source of truth: ui_template.html holds the master, and this script copies
the <svg style="display:none"> block to console_icons.svg verbatim.

Run manually when ui_template.html's icon block changes:
    python scripts/apin_v2/_build_icon_sprite.py
"""
import re
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "ui_template.html"
DST = HERE / "console_icons.svg"

text = SRC.read_text(encoding="utf-8")
m = re.search(
    r'<svg xmlns="http://www\.w3\.org/2000/svg" style="display:none">.*?</svg>',
    text, re.DOTALL)
if not m:
    raise SystemExit("Could not find the icon sprite <svg> block in ui_template.html")
sprite = m.group(0)
DST.write_text(sprite + "\n", encoding="utf-8")
n_symbols = sprite.count("<symbol id=")
print(f"Wrote {DST.relative_to(HERE.parent.parent)} — {n_symbols} symbols, "
      f"{len(sprite):,} bytes")
