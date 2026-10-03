"""tests/support.py

Shared fixtures for the test suite.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import functools
import math

from truss_bridge import AISC_360_16, YIELD_EULER, BridgeDesign, Edge, Material, design_bridge
from truss_bridge.solver import lattice_units

STEEL = Material("steel", 200e9, 250e6, 7850.0, AISC_360_16, 1.67)
PLASTIC = Material("plastic", 3e9, 50e6, 1400.0, YIELD_EULER, 3.0)


def inventory_for(length: float, width: float, height: float, copies: int = 150, slack: float = 0.0) -> list[Edge]:
    """Edges at every lattice-multiple length and 45-degree diagonal, rounded up to 0.1 mm."""
    unit = lattice_units(length, width, height)[0]
    multiples = range(1, round(max(width, height) / unit) + 1)
    lengths = [k * unit for k in multiples] + [k * unit * math.sqrt(2) for k in multiples]
    stock = [value + slack for value in lengths for _ in range(copies)]
    return [Edge(index, math.ceil(value * 1e4) / 1e4) for index, value in enumerate(stock)]


@functools.cache
def cached_design(length: float, width: float, height: float) -> BridgeDesign:
    """A design for the given dimensions, computed once per test session."""
    return design_bridge(inventory_for(length, width, height), length, width, height, max_trim=0.25)
