import json, requests, io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import hashlib, math
import cairosvg

REGISTRY_PATH = Path("data/glyph_registry.json")
SVG_DIR = Path("data/glyphs/svg")
PNG_DIR = Path("data/glyphs/png")
MASK_DIR = Path("data/glyphs/masks")

for d in [SVG_DIR, PNG_DIR, MASK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Placeholder NB → Mahadevan mapping
# (You can update this as we decode signs)
# -----------------------------------------------------------
def nb_to_m(nb_num):
    return nb_num  # simple placeholder until full mapping provided


# -----------------------------------------------------------
# Try to download historical SVG
# -----------------------------------------------------------
def try_download_svg(m_num):
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/Indus_script_sign_{m_num:03d}.svg"
    resp = requests.get(url)

    if resp.status_code == 200 and resp.text.startswith("<svg"):
        out = SVG_DIR / f"sign_{m_num:03d}.svg"
        out.write_text(resp.text, encoding="utf-8")
        return out
    return None


# -----------------------------------------------------------
# Procedural glyph generator
# -----------------------------------------------------------
def procedural_glyph(nb_code, size=512):
    seed = int(hashlib.sha256(nb_code.encode()).hexdigest(), 16)
    img = Image.new("RGBA", (size, size), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    cx, cy = size//2, size//2

    n = 4 + (seed % 4)
    for i in range(n):
        ang = (seed >> (i*3)) & 0xFF
        th = ang / 255.0 * 2 * math.pi
        r1 = size * (0.15 + i*0.1)
        x1 = cx + r1 * math.cos(th)
        y1 = cy + r1 * math.sin(th)
        x2 = cx + (r1*0.4) * math.cos(th + 0.7)
        y2 = cy + (r1*0.4) * math.sin(th + 0.7)
        draw.line([x1,y1,x2,y2], fill=(0,0,0,255), width=8)

    # mask
    gray = img.convert("L")
    bw = gray.point(lambda p: 255 if p > 20 else 0)
    return img, bw


# -----------------------------------------------------------
# Build registry
# -----------------------------------------------------------
def build_registry(start=1, end=417):
    registry = {}

    for nb in range(start, end+1):
        nb_code = f"NB{nb:03d}"
        print(f"Processing {nb_code}")

        # Try historical SVG
        m_num = nb_to_m(nb)
        svg_path = try_download_svg(m_num)

        if svg_path:
            # Rasterize SVG into PNG + mask
            png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=512)
            png_path = PNG_DIR / f"{nb_code}.png"
            with open(png_path, "wb") as f:
                f.write(png_bytes)

            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            mask = img.convert("L").point(lambda p: 255 if p > 20 else 0)
            mask_path = MASK_DIR / f"{nb_code}_mask.png"
            mask.save(mask_path)

            registry[nb_code] = {
                "glyph_type": "historical",
                "svg_path": str(svg_path),
                "png_path": str(png_path),
                "mask_path": str(mask_path)
            }

        else:
            # Fallback to procedural generator
            img, mask = procedural_glyph(nb_code)
            png_path = PNG_DIR / f"{nb_code}.png"
            mask_path = MASK_DIR / f"{nb_code}_mask.png"

            img.save(png_path)
            mask.save(mask_path)

            registry[nb_code] = {
                "glyph_type": "procedural",
                "svg_path": None,
                "png_path": str(png_path),
                "mask_path": str(mask_path)
            }

    # Save registry
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    print("Registry complete.")


if __name__ == "__main__":
    build_registry()
