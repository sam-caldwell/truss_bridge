"""truss_bridge/solver.py

Top-level design entry point.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import math
from fractions import Fraction

from .constraints import ANGLE_RULES, DEFAULT_ANGLES
from .constructive_search import search_constructive
from .exact_search import search_exact
from .lattice import CandidateLattice, build_candidate_lattice
from .model import BridgeDesign, Edge, NoDesignFoundError, SearchBudgetExceededError

MAX_DIMENSION_DENOMINATOR = 1000


def lattice_units(length: float, width: float, height: float, refinement: int = 1) -> list[float]:
    """Candidate lattice spacings: the greatest common divisor of the dimensions, divided by 1..refinement.

    Args:
        length: Bridge span.
        width: Bridge width.
        height: Bridge height.
        refinement: Number of successively finer spacings to return.

    Returns:
        Spacings from coarsest to finest.

    Raises:
        ValueError: If a dimension is not a fraction with denominator at most 1000.
    """
    dimensions = (length, width, height)
    fractions = [Fraction(value).limit_denominator(MAX_DIMENSION_DENOMINATOR) for value in dimensions]
    if any(abs(float(fraction) - value) > 1e-9 for fraction, value in zip(fractions, dimensions, strict=True)):
        raise ValueError(f"dimensions must be rational with denominator <= {MAX_DIMENSION_DENOMINATOR}")
    common_denominator = math.lcm(*(fraction.denominator for fraction in fractions))
    greatest_common_divisor = Fraction(
        math.gcd(*(int(fraction * common_denominator) for fraction in fractions)), common_denominator
    )
    return [float(greatest_common_divisor / divisor) for divisor in range(1, refinement + 1)]


def _validate_inputs(edges: list[Edge], length: float, width: float, height: float, angle_rule: str) -> None:
    if min(length, width, height) <= 0:
        raise ValueError("length, width and height must be positive")
    if angle_rule not in ANGLE_RULES:
        raise ValueError(f"unknown angle rule {angle_rule!r}; expected one of {ANGLE_RULES}")
    indices = [edge.index for edge in edges]
    if len(set(indices)) != len(indices):
        raise ValueError("edge indices must be unique")
    if any(edge.length <= 0 for edge in edges):
        raise ValueError("edge lengths must be positive")


def _search_node_set(
    lattice: CandidateLattice,
    edges: list[Edge] | None,
    max_trim: float,
    angles_degrees: tuple[float, ...],
    angle_rule: str,
    max_cut_rounds: int,
    exact_time_budget: float,
) -> BridgeDesign | None:
    """Search one node set: exactly for "chords" (with constructive fallback), constructively for "faces"."""
    if lattice.node_set.startswith("chords"):
        try:
            return search_exact(
                lattice,
                edges,
                max_trim,
                angles_degrees,
                angle_rule,
                max_cut_rounds=max_cut_rounds,
                time_budget=exact_time_budget,
            )
        except SearchBudgetExceededError:
            pass  # optimality could not be proven in time; settle for a valid design
    return search_constructive(lattice, edges, max_trim, angles_degrees, angle_rule)


def design_bridge(
    edges: list[Edge],
    length: float,
    width: float,
    height: float,
    *,
    max_trim: float = math.inf,
    angles_degrees: tuple[float, ...] = DEFAULT_ANGLES,
    angle_rule: str = "planar",
    refinement: int = 1,
    max_cut_rounds: int = 300,
    exact_time_budget: float = 30.0,
) -> BridgeDesign:
    """Design a rigid box-truss bridge from an inventory of edges.

    Panel lengths (multiples of the lattice unit that divide the span) are tried from
    longest to shortest. For each, the "chords" node set (section corners only) is searched
    exactly, which gives the least total trim or proves no design exists there. If the exact
    search cannot finish within its budget (long spans), the constructive search is used on
    the same node set instead. When the width and height are both multiples of the panel
    length and not both equal to it, the "faces" node set (every section-perimeter point) is
    then searched constructively. The first design found is returned; its ``search_method``
    says whether its trim is proven minimal.

    Args:
        edges: Inventory; indices must be unique.
        length: Span along x.
        width: Width along y.
        height: Height along z.
        max_trim: Largest length that may be cut off one edge.
        angles_degrees: Permitted joint angles.
        angle_rule: ``"planar"`` (angles checked within elevation, plan and section
            planes) or ``"pairwise"`` (true 3D angle between every pair of members).
        refinement: Number of successively finer lattice spacings to try.
        max_cut_rounds: Limit on rigidity cut iterations in each exact search.
        exact_time_budget: Wall-clock seconds allowed for each exact search before falling
            back to the constructive search.

    Returns:
        A rigid design satisfying the angle rule.

    Raises:
        ValueError: On invalid dimensions, angle rule or inventory.
        NoDesignFoundError: If no design is found. The message says whether a geometry
            exists but the inventory cannot build it.
    """
    _validate_inputs(edges, length, width, height, angle_rule)
    tried: list[str] = []
    buildable_geometry: str | None = None
    search_options = (max_trim, angles_degrees, angle_rule, max_cut_rounds, exact_time_budget)
    for unit in lattice_units(length, width, height, refinement):
        length_units, width_units, height_units = (round(value / unit) for value in (length, width, height))
        panel_lengths = sorted(
            (
                panel
                for panel in range(1, length_units + 1)
                if length_units % panel == 0 and panel <= max(width_units, height_units)
            ),
            reverse=True,
        )
        for panel_units in panel_lengths:
            node_sets = ["chords"]
            faces_fit = width_units % panel_units == 0 and height_units % panel_units == 0
            if faces_fit and (width_units, height_units) != (panel_units, panel_units):
                node_sets.append("faces")
            for node_set in node_sets:
                lattice = build_candidate_lattice(node_set, unit, length_units, width_units, height_units, panel_units)
                tried.append(f"{node_set}@panel {panel_units * unit:g}")
                design = _search_node_set(lattice, edges, *search_options)
                if design is not None:
                    return design
                if buildable_geometry is None and _search_node_set(lattice, None, *search_options) is not None:
                    buildable_geometry = tried[-1]
    if buildable_geometry:
        raise NoDesignFoundError(
            f"a rigid angle-compliant geometry exists ({buildable_geometry}) but the inventory cannot "
            "supply its members within max_trim"
        )
    raise NoDesignFoundError("no rigid angle-compliant geometry found on node sets " + ", ".join(tried))
