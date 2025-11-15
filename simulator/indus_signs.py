# simulator/indus_signs.py
import json
from pathlib import Path

NB_MIN = 1
NB_MAX = 417

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = DATA_DIR / "nb_signs.json"

def _zero_pad(n):
    return f"NB{n:03d}"

def _make_placeholder(n):
    code = _zero_pad(n)
    return {
        "code": code,
        "name": code,
        "maha_id": None,
        "occurrences": None,
        "category": None,
        "default_freq": 20.0,
        "sigma": 0.06,
        "harmonics": [[1.0, 1.0, 0.0]],
        "image_url": None,
        "notes": "Placeholder - replace with authoritative metadata or run scraper."
    }

def build_registry():
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # ensure all NB codes exist
            for i in range(NB_MIN, NB_MAX+1):
                c = _zero_pad(i)
                if c not in data:
                    data[c] = _make_placeholder(i)
            return data
        except Exception:
            pass
    # fallback: placeholders
    registry = {}
    for i in range(NB_MIN, NB_MAX+1):
        registry[_zero_pad(i)] = _make_placeholder(i)
    return registry

INDUS_SIGNS = build_registry()
NB_LIST = list(sorted(INDUS_SIGNS.keys()))
