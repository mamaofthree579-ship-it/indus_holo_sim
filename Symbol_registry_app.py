import base64
import streamlit as st
import json
from pathlib import Path
from PIL import Image

REGISTRY_PATH = Path("data/glyph_registry.json")

st.header("Indus Glyph Viewer / Downloader")

# Load registry
if REGISTRY_PATH.exists():
    registry = json.loads(REGISTRY_PATH.read_text())
else:
    st.warning("No glyph registry found. Run build_glyph_registry.py")
    st.stop()

all_nb = sorted(registry.keys())
nb_choice = st.selectbox("Choose symbol", all_nb)

entry = registry[nb_choice]
st.write("Glyph type:", entry["glyph_type"])

# Load PNG for display
png_path = entry["png_path"]
img = Image.open(png_path)

st.image(img, caption=nb_choice, width=300)

# Download button
with open(png_path, "rb") as f:
    png_bytes = f.read()

b64 = base64.b64encode(png_bytes).decode()

st.download_button(
    label="Download PNG",
    data=png_bytes,
    file_name=f"{nb_choice}.png",
    mime="image/png"
)
