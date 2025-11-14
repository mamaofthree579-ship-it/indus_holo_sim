"""
indus_signs.py

Provides INDUS_SIGNS registry for the Mahadevan NB sign list (NB001..NB417).

Behavior:
- If a CSV/JSON file with real metadata exists at 'data/nb_signs.csv' or 'data/nb_signs.json',
  this module will prefer that and load richer metadata (name, description, freq, image_url, etc.)
- Otherwise it generates a fallback registry with entries NB001..NB417 and safe defaults.
- Intended to be easy to update: drop a CSV/JSON or a 'data/images/' folder with images and the
  registry will pick them up automatically.

Fields in each sign entry (recommended):
{
  "code": "NB001",
  "name": "Jar (placeholder)",
  "maha_id": null,          # Mahadevan M-code if available
  "parpola_id": null,       # Parpola/Parpola-style mapping if available
  "occurrences": None,      # corpus count
  "category": None,         # rough category: fauna, vessel, geometric, human, deity...
  "default_freq": 20.0,     # placeholder frequency in Hz (tunable)
  "sigma": 0.06,            # spatial spread
  "harmonics": [(1.0,1.0,0.0)],  # default harmonic template
  "image_url": None,
  "notes": ""
}
"""

import os
import json
import csv
from pathlib import Path

NB_MIN = 1
NB_MAX = 417

# Default heuristic parameters
DEFAULT_FREQ = 20.0
DEFAULT_SIGMA = 0.06
DEFAULT_HARMONICS = [(1.0, 1.0, 0.0)]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def _zero_pad(nb):
    return f"NB{nb:03d}"

def _make_placeholder_entry(nb):
    code = _zero_pad(nb)
    return {
        "code": code,
        "name": code,               # placeholder name == code
        "maha_id": None,
        "parpola_id": None,
        "occurrences": None,
        "category": None,
        "default_freq": float(DEFAULT_FREQ),
        "sigma": float(DEFAULT_SIGMA),
        "harmonics": [tuple(map(float, h)) for h in DEFAULT_HARMONICS],
        "image_url": None,
        "notes": "Placeholder entry generated automatically. Replace with verified metadata."
    }

def _load_csv(path):
    """
    Expect CSV with columns:
    code,name,maha_id,parpola_id,occurrences,category,default_freq,sigma,harmonics,image_url,notes

    harmonics should be encoded as a JSON string, e.g. '[[1,1.0,0.0],[2,0.5,0.1]]'
    """
    entries = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("code") or row.get("nb") or row.get("NB") 
            if not code:
                continue
            try:
                harmonics = json.loads(row.get("harmonics") or "[]")
                harmonics = [tuple(map(float, h)) for h in harmonics] if harmonics else [tuple(map(float, h)) for h in DEFAULT_HARMONICS]
            except Exception:
                harmonics = [tuple(map(float, h)) for h in DEFAULT_HARMONICS]

            entry = {
                "code": code,
                "name": row.get("name") or code,
                "maha_id": row.get("maha_id") or None,
                "parpola_id": row.get("parpola_id") or None,
                "occurrences": int(row["occurrences"]) if row.get("occurrences") else None,
                "category": row.get("category") or None,
                "default_freq": float(row["default_freq"]) if row.get("default_freq") else float(DEFAULT_FREQ),
                "sigma": float(row["sigma"]) if row.get("sigma") else float(DEFAULT_SIGMA),
                "harmonics": harmonics,
                "image_url": row.get("image_url") or None,
                "notes": row.get("notes") or ""
            }
            entries[code] = entry
    return entries

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expecting dict keyed by code
    entries = {}
    for code, val in data.items():
        entry = {
            "code": code,
            "name": val.get("name") or code,
            "maha_id": val.get("maha_id"),
            "parpola_id": val.get("parpola_id"),
            "occurrences": val.get("occurrences"),
            "category": val.get("category"),
            "default_freq": float(val.get("default_freq", DEFAULT_FREQ)),
            "sigma": float(val.get("sigma", DEFAULT_SIGMA)),
            "harmonics": [tuple(map(float, h)) for h in val.get("harmonics", DEFAULT_HARMONICS)],
            "image_url": val.get("image_url"),
            "notes": val.get("notes","")
        }
        entries[code] = entry
    return entries

def build_registry():
    """
    Load registry from data/nb_signs.csv or data/nb_signs.json if available.
    Otherwise, generate placeholders NB001..NB417.
    """
    # check CSV
    csv_path = DATA_DIR / "nb_signs.csv"
    json_path = DATA_DIR / "nb_signs.json"

    if csv_path.exists():
        try:
            entries = _load_csv(csv_path)
            # ensure all NB codes are present; fill missing with placeholders
            for i in range(NB_MIN, NB_MAX+1):
                code = _zero_pad(i)
                if code not in entries:
                    entries[code] = _make_placeholder_entry(i)
            return entries
        except Exception as e:
            print("Warning: failed to load CSV registry:", e)

    if json_path.exists():
        try:
            entries = _load_json(json_path)
            for i in range(NB_MIN, NB_MAX+1):
                code = _zero_pad(i)
                if code not in entries:
                    entries[code] = _make_placeholder_entry(i)
            return entries
        except Exception as e:
            print("Warning: failed to load JSON registry:", e)

    # fallback: generate placeholder registry
    entries = {}
    for i in range(NB_MIN, NB_MAX+1):
        entries[_zero_pad(i)] = _make_placeholder_entry(i)

    return entries

# Build the global registry on import
INDUS_SIGNS = build_registry()
NB_LIST = list(sorted(INDUS_SIGNS.keys()))

# Convenience: small helper to attach images from a directory (optional)
def attach_images_from_dir(img_dir):
    """
    If you place images named NB001.png, NB002.png, ... in data/images/,
    this function will attach image_url fields pointing to the local path.

    Example:
        from simulator.indus_signs import attach_images_from_dir
        attach_images_from_dir('data/images')
    """
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return 0
    attached = 0
    for code in NB_LIST:
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
            p = img_dir / f"{code}{ext}"
            if p.exists():
                INDUS_SIGNS[code]["image_url"] = str(p.resolve())
                attached += 1
                break
    return attached

from simulator.glyph_generator import generate_glyph

def ensure_glyphs_for_registry(mode="neutral"):
    """
    For any INDUS_SIGNS entry missing image_url, generate a procedural glyph.
    mode: 'neutral'|'acoustic'|'light'|'matrix'
    """
    for code, entry in INDUS_SIGNS.items():
        if not entry.get("image_url"):
            path = generate_glyph(code, mode=mode)
            entry["image_url"] = path

# If running directly, print a quick summary
if __name__ == "__main__":
    print(f"INDUS_SIGNS loaded: {len(INDUS_SIGNS)} entries (NB{NB_MIN:03d}..NB{NB_MAX:03d})")
    sample = NB_LIST[:8]
    for k in sample:
        print("-", k, INDUS_SIGNS[k]["name"], INDUS_SIGNS[k]["default_freq"])
