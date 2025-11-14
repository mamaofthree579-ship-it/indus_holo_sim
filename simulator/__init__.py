"""
Indus Holographic-Frequency Simulator Package

This package provides:
- Symbol: wave-emitting holographic node with harmonics and propagation delay
- HoloSimulator: time-domain wave interference engine
- create_grid: helper to create the spatial mesh

Usage:

    from simulator import Symbol, HoloSimulator, create_grid
"""

from .symbol import Symbol
from .simulator import HoloSimulator
from .grid import create_grid

__all__ = [
    "Symbol",
    "HoloSimulator",
    "create_grid"
]
