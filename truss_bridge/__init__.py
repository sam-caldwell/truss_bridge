"""truss_bridge/__init__.py

Design, verify, load-test and draw rigid box-truss bridges built from an inventory of edges.

Typical use::

    from truss_bridge import (basic_load_cases, design_bridge, plot_design, read_inventory,
                              run_load_tests, verify_design)

    edges = read_inventory("example.csv")  # parts with their materials and sections
    design = design_bridge(edges, length=12, width=3, height=3, max_trim=0.25)
    assert not verify_design(design)
    cases = basic_load_cases(design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)
    print(run_load_tests(design, cases).summary())  # each member uses its part's material
    plot_design(design, "bridge.png")

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from .analysis import (
    LoadCase,
    LoadTestReport,
    analyze_load_case,
    axle_load,
    basic_load_cases,
    deck_pressure_load,
    lateral_pressure_load,
    moving_axle_loads,
    run_load_tests,
)
from .assignment import assign_edges_to_members
from .constraints import ANGLE_RULES, DEFAULT_ANGLES
from .inventory import example_inventory, read_inventory, write_inventory
from .model import (
    AISC_360_16,
    STRENGTH_METHODS,
    YIELD_EULER,
    BridgeDesign,
    CrossSection,
    Edge,
    Material,
    Member,
    MemberAssignment,
    NoDesignFoundError,
    SearchBudgetExceededError,
)
from .plotting import plot_design
from .sizing import SizingError, SizingResult, size_members, square_hollow_catalog
from .solver import design_bridge
from .verification import verify_design

__all__ = [
    "AISC_360_16",
    "ANGLE_RULES",
    "DEFAULT_ANGLES",
    "STRENGTH_METHODS",
    "YIELD_EULER",
    "BridgeDesign",
    "CrossSection",
    "Edge",
    "LoadCase",
    "LoadTestReport",
    "Material",
    "Member",
    "MemberAssignment",
    "NoDesignFoundError",
    "SearchBudgetExceededError",
    "SizingError",
    "SizingResult",
    "analyze_load_case",
    "assign_edges_to_members",
    "axle_load",
    "basic_load_cases",
    "deck_pressure_load",
    "design_bridge",
    "example_inventory",
    "lateral_pressure_load",
    "moving_axle_loads",
    "plot_design",
    "read_inventory",
    "run_load_tests",
    "size_members",
    "square_hollow_catalog",
    "verify_design",
    "write_inventory",
]
