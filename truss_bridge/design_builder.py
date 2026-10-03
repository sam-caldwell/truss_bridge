"""truss_bridge/design_builder.py

Turning a chosen set of candidate members into a BridgeDesign.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import numpy as np

from .assignment import assign_edges_to_members
from .lattice import CandidateLattice
from .model import BridgeDesign, Edge, Member, MemberAssignment

GEOMETRY_ONLY_EDGE_INDEX = -1
"""Edge index used when a design is built without an inventory (geometry probe)."""


def members_from_candidates(
    lattice: CandidateLattice, chosen_candidates: list[int]
) -> tuple[list[tuple[float, float, float]], list[Member], list[int]]:
    """Physical nodes and members for the chosen candidates.

    Nodes no chosen member touches are dropped and the rest renumbered.

    Returns:
        ``(node_coordinates, members, candidate_of_member)``, where members are ordered by
        role then position and ``candidate_of_member[i]`` is the candidate index of member ``i``.
    """
    used_nodes = sorted({node for candidate in chosen_candidates for node in lattice.candidates[candidate]})
    renumber = {old: new for new, old in enumerate(used_nodes)}
    node_coordinates = [tuple(float(c) * lattice.unit for c in lattice.nodes[node]) for node in used_nodes]
    ordered = sorted(chosen_candidates, key=lambda candidate: (lattice.roles[candidate], lattice.candidates[candidate]))
    members = []
    for candidate in ordered:
        start, end = lattice.candidates[candidate]
        length = float(np.linalg.norm(np.subtract(lattice.nodes[end], lattice.nodes[start]))) * lattice.unit
        members.append(Member(renumber[start], renumber[end], lattice.roles[candidate], length))
    return node_coordinates, members, ordered


def geometry_only_design(lattice: CandidateLattice, chosen_candidates: list[int]) -> BridgeDesign:
    """A design whose members are paired with exact-length placeholder edges.

    Used to report whether a geometry exists independently of the inventory.
    """
    nodes, members, _ = members_from_candidates(lattice, chosen_candidates)
    assignments = [MemberAssignment(m, Edge(GEOMETRY_ONLY_EDGE_INDEX, m.required_length)) for m in members]
    return BridgeDesign(lattice.node_set, lattice.unit, nodes, assignments, search_method="geometry only")


def design_with_matched_edges(
    lattice: CandidateLattice,
    chosen_candidates: list[int],
    edges: list[Edge],
    max_trim: float,
    search_method: str = "constructive",
) -> BridgeDesign | None:
    """A design whose edges are chosen by :func:`assign_edges_to_members`, or None if the stock cannot cover it."""
    nodes, members, _ = members_from_candidates(lattice, chosen_candidates)
    assignments = assign_edges_to_members(members, edges, max_trim)
    if assignments is None:
        return None
    used = {assignment.edge.index for assignment in assignments}
    return BridgeDesign(
        lattice.node_set,
        lattice.unit,
        nodes,
        assignments,
        [edge for edge in edges if edge.index not in used],
        search_method,
    )
