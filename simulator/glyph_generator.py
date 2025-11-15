# simulator/glyph_generator.py
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _code_seed(code: str) -> int:
    s = code.upper().strip()
    clean = "".join(ch for ch in s if ch.isalnum())
    h = 0
    for c in clean:
        h = (h * 131 + ord(c)) & 0xFFFFFFFF
    return int(h)

def generate_glyph(code: str, size: int = 256, mode: str = "neutral", overwrite: bool = False) -> str:
    code = code.upper()
    path = OUT_DIR / f"{code}.png"
    if path.exists() and not overwrite:
        return str(path)
    seed = _code_seed(code)
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = size/2, size/2
    outer = size/2
    for i in range(12, 0, -1):
        t = i / 12.0
        if mode == "acoustic":
            col = (40, 80 + int(80*t), 160 + int(40*t), int(10*t))
        elif mode == "light":
            col = (240, 240, 255, int(10*t))
        elif mode == "matrix":
            col = (30, 200 - int(50*t), 100 + int(20*t), int(10*t))
        else:
            col = (120, 120, 150, int(10*t))
        r = outer * t
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)
    n_primitives = 3 + (seed % 6)
    stroke = max(2, size // 28)
    for idx in range(n_primitives):
        typ = (seed >> (idx*3)) & 3
        angle = math.radians((seed >> (idx*5)) % 360)
        r = size * (0.15 + 0.65 * (idx + 1) / (n_primitives + 2))
        if mode == "acoustic":
            c = (255, 255, 255, 200 - idx*20)
        elif mode == "light":
            c = (255, 235, 200, 220 - idx*15)
        elif mode == "matrix":
            c = (40, 230 - idx*12, 120, 230)
        else:
            c = (230 - idx*15, 230 - idx*15, 230 - idx*15, 220)
        if typ == 0:
            x1, y1 = cx + r*math.cos(angle), cy + r*math.sin(angle)
            x2, y2 = cx - r*0.7*math.cos(angle+0.2), cy - r*0.7*math.sin(angle+0.2)
            draw.line([(x1,y1),(x2,y2)], fill=c, width=stroke)
        elif typ == 1:
            bbox = [cx-r, cy-r, cx+r, cy+r]
            start = (seed % 180)
            end = start + 90 + (seed % 60)
            draw.arc(bbox, start=start, end=end, fill=c, width=stroke)
        elif typ == 2:
            ang = angle
            p1 = (cx + r*0.9*math.cos(ang), cy + r*0.9*math.sin(ang))
            p2 = (cx + r*0.4*math.cos(ang+0.7), cy + r*0.4*math.sin(ang+0.7))
            p3 = (cx + r*0.4*math.cos(ang-0.7), cy + r*0.4*math.sin(ang-0.7))
            draw.line([p1,p2], fill=c, width=stroke)
            draw.line([p1,p3], fill=c, width=stroke)
        elif typ == 3:
            for k in range(4):
                a = angle + k * 0.6
                rr = r * 0.4 + k * 5
                dx = cx + rr*math.cos(a)
                dy = cy + rr*math.sin(a)
                d = max(2, stroke//2)
                draw.ellipse([dx-d, dy-d, dx+d, dy+d], fill=c)
    if mode == "acoustic":
        for k in range(1,5):
            rr = k * (size * 0.05)
            draw.ellipse([cx-rr, cy-rr, cx+rr, cy+rr], outline=(200,220,255,90), width=1)
    if mode == "light":
        glow = Image.new("RGBA", (size, size), (0,0,0,0))
        gdraw = ImageDraw.Draw(glow)
        for k in range(6,0,-1):
            r = size * 0.03 * k
            gdraw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255,240,200,40*k))
        glow = glow.filter(ImageFilter.GaussianBlur(8))
        img = Image.alpha_composite(img, glow)
    if mode == "matrix":
        grid = Image.new("RGBA", (size, size), (0,0,0,0))
        g = ImageDraw.Draw(grid)
        step = max(10, size//20)
        col = (40,170,100,120)
        for x in range(0,size,step):
            g.line([(x,0),(x,size)], fill=col, width=1)
        for y in range(0,size,step):
            g.line([(0,y),(size,y)], fill=col, width=1)
        img = Image.alpha_composite(img, grid)
    img.save(path := OUT_DIR / f"{code}.png", "PNG")
    return str(path)
