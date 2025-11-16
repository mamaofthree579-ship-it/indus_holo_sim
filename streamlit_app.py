import os
import json
import streamlit as st

# Import local simulator modules
from simulator.symbol import Symbol
from simulator.simulator import HoloSimulator, create_grid
from simulator.glyph_generator import generate_glyph


# -------------------------------------------------------------
# 1. Load or create NB registry
# -------------------------------------------------------------
REGISTRY_PATH = "data/nb_signs.json"

def load_registry():
    """Load the NB registry file; generate if missing."""
    if not os.path.exists(REGISTRY_PATH):
        st.warning("⚠️ nb_signs.json not found — generating empty registry…")

        os.makedirs("data", exist_ok=True)

        # Auto-generate blank registry (NB001–NB417)
        blank = {}
        for i in range(1, 418):
            code = f"NB{i:03d}"
            blank[code] = {
                "mahadevan_id": None,
                "wells_id": None,
                "fuls_id": None,
                "structural_class": None,
                "frequency_total": None,
                "frequency_rank": None,
                "positional_profile": None,
                "regional_distribution": None,
                "artifact_distribution": None,
                "variants": [],
                "name": None,
                "interpretation_martin2025": None,
                "glyph": {
                    "image_path": None,
                    "vector_path": None,
                    "procedural_params": {
                        "stroke": None,
                        "symmetry": None,
                        "density": None
                    }
                },
                "holography": {
                    "default_freq": 25.0,
                    "sigma": 0.06,
                    "harmonics": [[1, 1.0, 0.0]],
                    "class_profile": "basic"
                },
                "sources": {
                    "mahadevan": None,
                    "fuls": None,
                    "harappa_structure": None,
                    "martin": None
                },
                "confidence": {
                    "mahadevan_mapping": "unknown",
                    "structural_class": "unknown",
                    "interpretation": "none"
                }
            }

        with open(REGISTRY_PATH, "w") as f:
            json.dump(blank, f, indent=2)

        return blank

    # Existing file — load normally
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


# -------------------------------------------------------------
# Load registry into memory
# -------------------------------------------------------------
registry = load_registry()


# -------------------------------------------------------------
# 2. Streamlit UI
# -------------------------------------------------------------
st.title("Indus Script Holographic-Frequency Simulator")
st.subheader("Procedural Glyphs • Resonance Field • Multi-Harmonic Phase Interference")


# Sidebar selection
symbol_id = st.sidebar.selectbox(
    "Select NB Sign",
    list(registry.keys())
)

symbol_data = registry[symbol_id]


# -------------------------------------------------------------
# 3. Create Symbol Object
# -------------------------------------------------------------
freq = symbol_data.get("holography", {}).get("default_freq", 25.0)
sigma = symbol_data.get("holography", {}).get("sigma", 0.06)
harm = symbol_data.get("holography", {}).get("harmonics", [[1, 1.0, 0.0]])

symbol = Symbol(
    name=symbol_id,
    frequency=freq,
    sigma=sigma,
    harmonics=harm
)


# -------------------------------------------------------------
# 4. User Harmonic Controls
# -------------------------------------------------------------
st.sidebar.markdown("### Harmonic Controls")

num_harm = st.sidebar.number_input(
    "Number of Harmonics",
    min_value=1,
    max_value=10,
    value=len(harm),
    step=1
)

user_harmonics = []
for i in range(num_harm):
    st.sidebar.markdown(f"##### Harmonic {i+1}")
    order = st.sidebar.number_input(f"Order {i+1}", 1, 10, value=(harm[i][0] if i < len(harm) else 1))
    multiplier = float(st.sidebar.number_input(f"Multiplier {i+1}", min_value=0.1, max_value=10.0, value=1.0))
    phase = float(st.sidebar.number_input(f"Phase {i+1}", min_value=-3.14, max_value=3.14, value=0.0))
    user_harmonics.append([order, multiplier, phase])

symbol.harmonics = user_harmonics


# -------------------------------------------------------------
# 5. Generate the procedural glyph
# -------------------------------------------------------------
st.markdown("### Generated Glyph")

glyph_img = generate_glyph(symbol_id, mode="vector")

st.image(glyph_img, caption=f"Glyph for {symbol_id}", use_column_width=True)


# -------------------------------------------------------------
# 6. Create holographic field simulation
# -------------------------------------------------------------
sim = HoloSimulator()

grid = create_grid(size=200)  # adjustable

field = sim.compute_field(symbol, grid)

st.markdown("### Holographic Resonance Field")
st.pyplot(field)


# -------------------------------------------------------------
# 7. Show metadata (debug / research view)
# -------------------------------------------------------------
with st.expander("Metadata (from nb_signs.json)"):
    st.json(symbol_data)
