# simulator/glyphs_generate.py
# Usage: python simulator/glyphs_generate.py --start 3 --end 25
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import math, hashlib, numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def seed_from_code(code: str) -> int:
    h = hashlib.sha256(code.encode()).digest()
    return int.from_bytes(h[:4], "big")

def draw_glyph_image(code: str, size:int=512):
    seed = seed_from_code(code)
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = size//2, size//2

    # background subtle ring
    for i in range(3,0,-1):
        t = i/3.0
        r = int(size*0.48*t)
        col = (200, 200, 220, int(10*t))
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=col)

    # central motif (deterministic)
    n = 3 + (seed % 5)
    for i in range(n):
        ang = (seed >> (i*3)) & 0xFF
        theta = ang/255.0 * 2*math.pi
        r = (size * 0.18) + i * (size * 0.08)
        x1 = cx + r * math.cos(theta)
        y1 = cy + r * math.sin(theta)
        x2 = cx + (r * 0.4) * math.cos(theta + 0.7)
        y2 = cy + (r * 0.4) * math.sin(theta + 0.7)
        stroke = max(2, size // 48 - i)
        color = (int(220 - i*18), int(180 - i*10), 240, 220)
        draw.line([(x1,y1),(x2,y2)], fill=color, width=stroke)

    # small dots around center
    for k in range(6):
        a = 2.0 * math.pi * (k / 6.0 + (seed % 10) * 0.01)
        rr = size * (0.05 + 0.02 * (k % 3))
        dx = cx + rr * math.cos(a)
        dy = cy + rr * math.sin(a)
        draw.ellipse([dx-4, dy-4, dx+4, dy+4], fill=(255,255,255,200))

    # subtle blur and save
    glow = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    combined = Image.alpha_composite(glow, img)
    return combined

def make_binary_mask(img: Image.Image, threshold:int=40):
    # Convert RGBA -> grayscale -> binary mask
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p > threshold else 0)
    # Slight blur to soften edges for more realistic holographic patterns
    bw = bw.filter(ImageFilter.GaussianBlur(radius=1))
    return bw

def generate_range(start:int=3, end:int=25, size:int=512):
    outputs = []
    for i in range(start, end+1):
        code = f"NB{i:03d}"
        img = draw_glyph_image(code, size=size)
        mask = make_binary_mask(img, threshold=30)
        img_path = OUT_DIR / f"{code}.png"
        mask_path = OUT_DIR / f"{code}_mask.png"
        img.save(img_path, format="PNG")
        mask.save(mask_path, format="PNG")
        outputs.append((code, img_path, mask_path))
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=3)
    parser.add_argument("--end", type=int, default=25)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    out = generate_range(args.start, args.end, args.size)
    print("Generated:", out)
