#!/usr/bin/env python
"""WCAG contrast of the token pairs the web UI actually uses, in both themes.

The dark palette in web/src/index.css was chosen against its surfaces rather than
inverted from the light one, and this is what holds it to that: a token that
drifts below its floor fails here, before anyone squints at a sidebar label.

    uv run python scripts/check_contrast.py

Thresholds are WCAG 2.1: 4.5:1 for body text (AA), 3:1 for large text and UI
chrome. `ink-faint` is section labels and timestamps — chrome, so 3:1 — and the
light theme's 2.8:1 on white is the one inherited exception, recorded rather than
hidden: it was sampled from the mockup, and this script pins that it does not get
worse.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CSS = Path(__file__).resolve().parent.parent / "web" / "src" / "index.css"

# (foreground token, background token, minimum ratio)
PAIRS = [
    ("ink", "surface", 7.0),
    ("ink", "surface-sunken", 7.0),
    ("ink", "surface-raised", 7.0),
    ("ink-muted", "surface", 4.5),
    ("ink-muted", "surface-sunken", 4.5),
    ("ink-muted", "surface-raised", 4.5),
    ("ink-faint", "surface", 3.0),
    ("ink-faint", "surface-sunken", 3.0),
    ("ink-inverse", "ink", 7.0),
    ("violet-600", "surface", 3.0),
    ("violet-700", "surface", 3.0),
    ("graph-label", "surface", 4.5),
    ("graph-caption", "surface", 3.0),
]

# The mockup's own value; kept, not fixed, so the light theme still matches it.
KNOWN_EXCEPTIONS = {
    ("light", "ink-faint", "surface"): 2.75,
    ("light", "ink-faint", "surface-sunken"): 2.5,
}


def _luminance(rgb: tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = _luminance(a), _luminance(b)
    light, dark = max(la, lb), min(la, lb)
    return (light + 0.05) / (dark + 0.05)


def _block(css: str, selector: str) -> dict[str, tuple[int, int, int]]:
    start = css.index(selector)
    body = css[css.index("{", start) + 1 : css.index("}", start)]
    tokens = {}
    for name, r, g, b in re.findall(r"--c-([\w-]+):\s*(\d+)\s+(\d+)\s+(\d+)", body):
        tokens[name] = (int(r), int(g), int(b))
    return tokens


def main() -> int:
    css = CSS.read_text(encoding="utf-8")
    themes = {"light": _block(css, ":root {"), "dark": _block(css, ":root[data-theme='dark']")}
    failures = 0
    for theme, tokens in themes.items():
        print(f"--- {theme} ---")
        for fg, bg, floor in PAIRS:
            ratio = contrast(tokens[fg], tokens[bg])
            allowed = KNOWN_EXCEPTIONS.get((theme, fg, bg), floor)
            ok = ratio + 0.05 >= allowed
            failures += not ok
            note = "  (mockup value, pinned)" if (theme, fg, bg) in KNOWN_EXCEPTIONS else ""
            print(f"  {'OK  ' if ok else 'FAIL'} {fg:<12} on {bg:<15} {ratio:5.2f}:1  floor {allowed}{note}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
