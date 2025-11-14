import matplotlib.pyplot as plt
from simulator.grid import create_grid
from simulator.symbol import Symbol
from simulator.simulator import HoloSimulator

# Create grid
XX, YY = create_grid(256)

# Build simulator
sim = HoloSimulator(XX, YY)

# Add symbols
sim.add_symbol(Symbol("economic_marker", 0.25, 0.7, base_freq=10.0))
sim.add_symbol(Symbol("social_glyph", 0.6, 0.5, base_freq=25.0))
sim.add_symbol(Symbol("cosmic_motif", 0.4, 0.25, base_freq=60.0))

sim.add_symbol(Symbol(
    "deity",
    0.75,
    0.2,
    base_freq=12.0,
    amplitude=1.4,
    harmonics=[
        (1.0, 1.0, 0.0),
        (2.0, 0.7, 0.5),
        (3.0, 0.4, 1.1),
        (5.0, 0.2, 2.0),
    ]
))

# Run simulation
intensity = sim.simulate()

# Visualize
plt.figure(figsize=(6,6))
plt.imshow(intensity, origin='lower', extent=(0,1,0,1))
plt.title("Holographic Intensity Map")
plt.xlabel("X")
plt.ylabel("Y")

for s in sim.symbols:
    plt.plot(s.x, s.y, 'wo', markersize=5)
    plt.text(s.x+0.01, s.y+0.01, s.name, fontsize=8)

plt.show()
