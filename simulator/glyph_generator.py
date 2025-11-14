"""
glyph_generator.py

Procedural glyph renderer for NB codes.

- Deterministic: seed derived from NB code (e.g. 'NB023' -> seed)
- Produces a stylized vector/raster PNG with optional modality:
    - 'neutral' : simple geometric glyph
    - 'acoustic': concentric wave rings + stroke motifs
    - 'light'   : glow / radial gradient + brighter strokes
    - 'matrix'  : grid + interference-like cross-hatch
- Uses Pillow and matplotlib for simple drawing. No external font requirements.
- Output: returns path to generated PNG in data/images/ (creates directory if needed)
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _code_seed(code: str) -> int:
    # deterministic numeric seed from code string
    # e.g. NB023 -> 23023 + sum of char codes to reduce collisions
    s = code.upper().strip()
    # drop non-alphanumerics
    clean = "".join(ch for ch in s if ch.isalnum())
    # simple hash -> int
    h = 0
    for c in clean:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return int(h)

def _circle_points(cx, cy, r, n, phase=0.0):
    return [(cx + r * math.cos(2*math.pi*i/n + phase),
             cy + r * math.sin(2*math.pi*i/n + phase)) for i in range(n)]

def generate_glyph(code: str, size: int = 256, mode: str = "neutral", overwrite: bool = False) -> str:
    """
    Generate a procedural glyph PNG for 'code' and return the file path (string).
    Modes: "neutral", "acoustic", "light", "matrix"
    """
    code = code.upper()
    fname = OUT_DIR / f"{code}.png"
    if fname.exists() and not overwrite:
        return str(fname)

    seed = _code_seed(code)
    rng = np.random.default_rng(seed)

    # Create base image (RGBA)
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img, "RGBA")

    cx, cy = size/2, size/2
    pad = int(size * 0.08)
    inner = pad
    outer = size - pad

    # Background faint radial gradient (PIL doesn't have gradient primitives — approximate with concentric circles)
    for i in range(12, 0, -1):
        t = i / 12.0
        # base color depends on mode
        if mode == "acoustic":
            base_col = (20, 30 + int(100*t), 120 + int(80*t), int(12 * t))
        elif mode == "light":
            base_col = (220 + int(20*t), 220 + int(20*t), 235, int(12 * t))
        elif mode == "matrix":
            base_col = (10, 200 - int(100*t), 70 + int(30*t), int(10 * t))
        else:
            base_col = (100, 100, 120, int(10 * t))
        r = outer * t
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=base_col)

    # Determine number of core strokes / primitives from seed
    n_primitives = 3 + (seed % 6)  # 3..8 primitives
    stroke_w = max(1, size // 28)

    # Build primitive shapes (lines, arcs, chevrons, dots), deterministic ordering
    primitives = []
    for i in range(n_primitives):
        typ = (seed >> (i*3)) & 0x7
        angle = (seed >> (i*5)) % 360
        r = (size * (0.12 + ((i + 1) / (n_primitives + 2)) * 0.65))
        primitives.append((typ % 4, angle, r))

    # Draw primitives
    for idx, (typ, angle, r) in enumerate(primitives):
        angle_rad = math.radians(angle)
        # compute end points
        x1 = cx + r * math.cos(angle_rad)
        y1 = cy + r * math.sin(angle_rad)
        x2 = cx - r * 0.6 * math.cos(angle_rad + 0.3)
        y2 = cy - r * 0.6 * math.sin(angle_rad + 0.3)

        # stroke color varies by mode and primitive index
        if mode == "acoustic":
            stroke = (255, 255, 255, 230 - idx*18)
        elif mode == "light":
            stroke = (255, 235, 180, 240 - idx*20)
        elif mode == "matrix":
            stroke = (40, 230 - idx*12, 120, 230)
        else:
            stroke = (230 - idx*15, 230 - idx*15, 230 - idx*15, 220)

        if typ == 0:
            # simple thick line
            draw.line([(x1, y1), (x2, y2)], fill=stroke, width=stroke_w + (idx % 3))
            # add small perpendicular tick
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            perp = 0.12 * r
            draw.line([(mx-perp, my-perp), (mx+perp, my+perp)], fill=stroke, width=max(1, stroke_w//2))
        elif typ == 1:
            # arc (partial circle)
            bbox = [cx - r, cy - r, cx + r, cy + r]
            start = (angle % 360)
            end = (start + 60 + (seed % 120)) % 360
            draw.arc(bbox, start=start, end=end, fill=stroke, width=stroke_w + (idx%2))
        elif typ == 2:
            # chevron: two lines forming a V
            ang = angle_rad
            p1 = (cx + r*0.9*math.cos(ang), cy + r*0.9*math.sin(ang))
            p2 = (cx + r*0.4*math.cos(ang+0.7), cy + r*0.4*math.sin(ang+0.7))
            p3 = (cx + r*0.4*math.cos(ang-0.7), cy + r*0.4*math.sin(ang-0.7))
            draw.line([p1,p2], fill=stroke, width=stroke_w)
            draw.line([p1,p3], fill=stroke, width=stroke_w)
        else:
            # cluster of dots
            for k in range(4 + (idx % 4)):
                a = angle_rad + (k - 1.5) * 0.25
                rr = r * (0.6 + 0.15 * k)
                dx = cx + rr * math.cos(a)
                dy = cy + rr * math.sin(a)
                dot_r = max(2, stroke_w // 2 + (k % 3))
                draw.ellipse([dx-dot_r, dy-dot_r, dx+dot_r, dy+dot_r], fill=stroke)

    # Mode-specific overlays
    if mode == "acoustic":
        # add concentric faint rings centered at cx,cy
        ring_color = (180, 210, 255, 80)
        for k in range(1, 6):
            rr = (size * 0.06) * k * (1.0 + ((seed >> k) & 3)*0.04)
            bbox = [cx-rr, cy-rr, cx+rr, cy+rr]
            draw.ellipse(bbox, outline=ring_color, width=max(1, stroke_w//2))

    if mode == "light":
        # add bright central glow
        glow = Image.new("RGBA", (size, size), (0,0,0,0))
        gd = ImageDraw.Draw(glow)
        for k in range(6,0,-1):
            alpha = int(40 * (k))
            r = size * (0.02 * k)
            gd.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 240, 200, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=size*0.02))
        img = Image.alpha_composite(img, glow)

    if mode == "matrix":
        # overlay a fine grid and a cross-hatch interference pattern
        grid = Image.new("RGBA", (size, size), (0,0,0,0))
        gd = ImageDraw.Draw(grid)
        step = max(8, size // 24)
        grid_col = (20, 60, 40, 100)
        for x in range(0, size, step):
            gd.line([(x,0),(x,size)], fill=grid_col, width=1)
        for y in range(0, size, step):
            gd.line([(0,y),(size,y)], fill=grid_col, width=1)
        grid = grid.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = Image.alpha_composite(img, grid)

    # subtle vignette for contrast
    vign = Image.new("L", (size, size), 0)
    vg = Image.new("RGBA", (size, size), (0,0,0,0))
    vign_draw = ImageDraw.Draw(vign)
    for i in range(size//2):
        alpha = int(120 * (i / (size/2)))
        bbox = [i, i, size - i, size - i]
        vign_draw.ellipse(bbox, fill=alpha)
    # invert vign and apply as alpha mask of a dark overlay
    vign = vign.convert("L")
    inv = Image.eval(vign, lambda px: 255-px)
    dark = Image.new("RGBA", (size, size), (0,0,0,80))
    dark.putalpha(inv)
    img = Image.alpha_composite(img, dark)

    # final optional blur / sharpen depending on mode
    if mode == "light":
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    if mode == "matrix":
        img = img.filter(ImageFilter.SHARPEN)

    # Save PNG
    img = img.convert("RGBA")
    img.save(fname := OUT_DIR / f"{code}.png", format="PNG")
    return str(fname)
