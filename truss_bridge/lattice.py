"""truss_bridge/lattice.py

Candidate node sets and the candidate members joining them.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .geometry import IntVector, classify_member, is_lattice_direction, lies_strictly_inside_segment, subtract

NODE_SETS = ("chords", "faces")


@dataclass
class CandidateLattice:
    """Every node and every member the search may use for one node set.

    Attributes:
        node_set: Label of the node set, including the panel length when not one unit.
        unit: Physical length of one lattice unit.
        nodes: Integer lattice coordinates of every node.
        candidates: Candidate members as ``(start_node, end_node)`` index pairs.
        roles: Structural role of each candidate, parallel to ``candidates``.
        width_units: Bridge width in lattice units.
        height_units: Bridge height in lattice units.
    """

    node_set: str
    unit: float
    nodes: list[IntVector]
    candidates: list[tuple[int, int]]
    roles: list[str]
    width_units: int
    height_units: int


def section_points(node_set: str, width_units: int, height_units: int, spacing: int = 1) -> list[tuple[int, int]]:
    """Cross-section ``(y, z)`` positions where nodes are placed at every panel point.

    Args:
        node_set: ``"chords"`` for the four corners of the box section only, or
            ``"faces"`` for every point on the section perimeter at ``spacing``.
        width_units: Section width in lattice units.
        height_units: Section height in lattice units.
        spacing: Distance between perimeter points, in lattice units.

    Returns:
        Sorted list of ``(y, z)`` points.

    Raises:
        ValueError: If ``node_set`` is not recognized.
    """
    if node_set == "chords":
        return sorted({(0, 0), (width_units, 0), (0, height_units), (width_units, height_units)})
    if node_set == "faces":
        return sorted(
            {
                (y, z)
                for y in range(0, width_units + 1, spacing)
                for z in range(0, height_units + 1, spacing)
                if y in (0, width_units) or z in (0, height_units)
            }
        )
    raise ValueError(f"unknown node set {node_set!r}; expected one of {NODE_SETS}")


def build_candidate_lattice(
    node_set: str, unit: float, length_units: int, width_units: int, height_units: int, panel_units: int = 1
) -> CandidateLattice:
    """Place nodes at every panel point and list every admissible member between them.

    A candidate member joins two nodes along an axis or an in-plane 45-degree diagonal and
    does not pass through any other node. Members are limited to spanning at most
    ``max(width_units, height_units)`` along x, the longest a 45-degree diagonal can reach.

    Args:
        node_set: ``"chords"`` or ``"faces"``; see :func:`section_points`.
        unit: Physical length of one lattice unit.
        length_units: Span in lattice units.
        width_units: Width in lattice units.
        height_units: Height in lattice units.
        panel_units: Distance between panel points along x, in lattice units.

    Returns:
        The candidate lattice.
    """
    section = section_points(node_set, width_units, height_units, panel_units)
    nodes = [(x, y, z) for x in range(0, length_units + 1, panel_units) for y, z in section]
    reach = max(width_units, height_units)
    nodes_at_station: dict[int, list[int]] = defaultdict(list)
    for node_index, node in enumerate(nodes):
        nodes_at_station[node[0]].append(node_index)
    stations = sorted(nodes_at_station)

    candidates: list[tuple[int, int]] = []
    roles: list[str] = []
    for start_index, start in enumerate(nodes):
        for station in stations:
            if not start[0] <= station <= start[0] + reach:
                continue
            for end_index in nodes_at_station[station]:
                if end_index <= start_index:
                    continue
                end = nodes[end_index]
                if not is_lattice_direction(subtract(end, start)):
                    continue
                low_x, high_x = min(start[0], end[0]), max(start[0], end[0])
                passes_through_node = any(
                    lies_strictly_inside_segment(nodes[other], start, end)
                    for other_station in stations
                    if low_x <= other_station <= high_x
                    for other in nodes_at_station[other_station]
                )
                if passes_through_node:
                    continue
                candidates.append((start_index, end_index))
                roles.append(classify_member(start, end, width_units=width_units, height_units=height_units))

    label = node_set if panel_units == 1 else f"{node_set}/panel {panel_units * unit:g}"
    return CandidateLattice(label, unit, nodes, candidates, roles, width_units, height_units)
