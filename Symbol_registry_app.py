import streamlit as st
import json
from pathlib import Path
from PIL import Image
import base64

REGISTRY = Path("data/glyph_registry.json")

st.title("Indus Glyph Viewer (Historical + Procedural)")

if not REGISTRY.exists():
    st.error("Registry not built. Run build_glyph_registry.py")
    st.stop()

registry = json.loads(REGISTRY.read_text())

nb_list = sorted(registry.keys())
nb = st.selectbox("Choose Sign", nb_list)

entry = registry[nb]

# Load PNG
img = Image.open(entry["png_path"])
st.image(img, caption=f"{nb} — {entry['glyph_type']}", width=300)

# Download button
with open(entry["png_path"], "rb") as f:
    bts = f.read()

st.download_button(
    "Download PNG",
    data=bts,
    file_name=f"{nb}.png",
    mime="image/png"
)
