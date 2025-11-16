"""
Simple procedural glyph generator.
Produces deterministic PNG thumbnails for codes like "NB001".
Returns the path to the PNG image (string) which can be passed to st.image().
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import math
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _code_seed(code: str) -> int:
    s = code.upper().strip()
    clean = "".join(ch for ch in s if ch.isalnum())
    h = 0
    for c in clean:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return int(h)


def generate_glyph(code: str, mode: str = "vector", size: int = 256, overwrite: bool = False) -> str:
    """
    Generate and save a PNG glyph for `code`. Returns the absolute path string.
    Modes: "vector", "acoustic", "light", "matrix" (vector is same as neutral)
    """
    code = code.upper()
    out_path = OUT_DIR / f"{code}.png"
    if out_path.exists() and not overwrite:
        return str(out_path)

    seed = _code_seed(code)
    rng = np.random.default_rng(seed)

    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = size // 2, size // 2
    stroke = max(2, size // 36)

    # background rings dependent on mode
    for i in range(8, 0, -1):
        t = i / 8.0
        if mode in ("acoustic",):
            col = (20, 60 + int(120 * t), 170, int(12 * t))
        elif mode in ("light",):
            col = (240, 240, 255, int(10 * t))
        elif mode in ("matrix",):
            col = (20, 200 - int(100 * t), 100, int(14 * t))
        else:
            col = (200 - int(80 * t), 200 - int(80 * t), 220, int(10 * t))
        r = int((size * 0.48) * t)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)

    # primitives drawn deterministically
    n = 3 + (seed % 5)
    for i in range(n):
        ang = 2.0 * math.pi * ((seed >> (i * 3)) % 360) / 360.0
        r = (size * 0.18) + i * (size * 0.08)
        x1 = cx + r * math.cos(ang)
        y1 = cy + r * math.sin(ang)
        x2 = cx + (r * 0.4) * math.cos(ang + 0.7)
        y2 = cy + (r * 0.4) * math.sin(ang + 0.7)
        color = (int(220 - i * 20), int(220 - i * 12), 240, 220)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=stroke)

    # small dots
    for k in range(6):
        a = 2.0 * math.pi * (k / 6.0 + (seed % 10) * 0.01)
        rr = size * (0.05 + 0.02 * (k % 3))
        dx = cx + rr * math.cos(a)
        dy = cy + rr * math.sin(a)
        draw.ellipse([dx - 3, dy - 3, dx + 3, dy + 3], fill=(255, 255, 255, 200))

    # subtle glow
    glow = img.filter(ImageFilter.GaussianBlur(radius=2))
    combined = Image.alpha_composite(glow, img)
    combined.save(out_path, format="PNG")
    return str(out_path)
