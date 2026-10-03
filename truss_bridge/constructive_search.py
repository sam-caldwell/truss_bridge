"""truss_bridge/constructive_search.py

Constructive design search for dense node sets.

Used where the exact search is too slow: the "faces" node set, needed when the bridge
width and height differ.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, milp

from .constraints import find_conflicting_pairs
from .design_builder import design_with_matched_edges, geometry_only_design
from .geometry import subtract
from .lattice import CandidateLattice
from .linear_rows import LinearRows
from .model import BridgeDesign, Edge
from .rigidity import rigidity_matrix

ROLE_PRIORITY = {
    "bottom_chord": 0,
    "top_chord": 0,
    "stringer": 1,
    "vertical": 1,
    "floor_beam": 1,
    "top_strut": 1,
    "side_diagonal": 2,
    "bottom_lateral": 2,
    "top_lateral": 2,
    "sway_brace": 3,
    "lateral": 4,
    "cross_strut": 4,
    "interior_post": 4,
    "interior_diagonal": 4,
}
"""Order in which roles are offered to the constructive search (lower first)."""

SECTION_ROLES = ("sway_brace", "cross_strut", "interior_post")
_SECTION_CUT_ROUNDS = 200


def choose_section_bracing(lattice: CandidateLattice, conflicts_of: dict[int, set[int]]) -> set[int] | None:
    """Choose a cross-section bracing pattern that holds every face node out of plane.

    A node in the middle of a face can only be held against moving out of that face by a
    member that leaves the face. A small exact MILP picks the fewest non-conflicting section
    members (sway braces, cross struts, interior posts) at the first station such that the
    out-of-plane motions of all face-interior nodes are fully restrained; rank deficiency is
    removed by the same mechanism cuts as the exact search. The pattern is then repeated at
    every station.

    Args:
        lattice: Candidate nodes and members.
        conflicts_of: For each candidate, the set of candidates it conflicts with.

    Returns:
        Candidate indices of the pattern at all stations (empty if no face has interior
        nodes), or None if no valid pattern exists.
    """
    nodes, candidates = lattice.nodes, lattice.candidates
    width, height = lattice.width_units, lattice.height_units
    in_section_plane = [
        candidate
        for candidate, (start, end) in enumerate(candidates)
        if nodes[start][0] == nodes[end][0] and lattice.roles[candidate] in SECTION_ROLES
    ]
    first_station = nodes[0][0]
    station_candidates = [
        candidate for candidate in in_section_plane if nodes[candidates[candidate][0]][0] == first_station
    ]

    out_of_plane_axis: dict[tuple[int, int, int], int] = {}
    for node in nodes:
        if node[0] != first_station:
            continue
        _, y, z = node
        if y in (0, width) and 0 < z < height:
            out_of_plane_axis[node] = 1  # side-face node: must be held in y
        elif z in (0, height) and 0 < y < width:
            out_of_plane_axis[node] = 2  # deck or top node: must be held in z
    if not out_of_plane_axis:
        return set()

    variable_of = {candidate: position for position, candidate in enumerate(station_candidates)}
    rows = LinearRows()
    for node, axis in out_of_plane_axis.items():
        holding = {}
        for candidate in station_candidates:
            start, end = candidates[candidate]
            if node in (nodes[start], nodes[end]) and subtract(nodes[end], nodes[start])[axis] != 0:
                holding[variable_of[candidate]] = 1
        if not holding:
            return None
        rows.add(holding, lower=1)
    for candidate in station_candidates:
        for other in conflicts_of[candidate]:
            if other in variable_of and other > candidate:
                rows.at_most_one(variable_of[candidate], variable_of[other])

    # Restraint matrix: with every face braced in its own plane, the unknowns are one
    # out-of-plane velocity per face-interior node; each member contributes one row.
    unknown_of = {node: position for position, node in enumerate(out_of_plane_axis)}
    restraint = np.zeros((len(station_candidates), len(unknown_of)))
    for position, candidate in enumerate(station_candidates):
        start, end = (nodes[index] for index in candidates[candidate])
        direction = np.subtract(end, start, dtype=float)
        direction /= np.linalg.norm(direction)
        if start in unknown_of:
            restraint[position, unknown_of[start]] -= direction[out_of_plane_axis[start]]
        if end in unknown_of:
            restraint[position, unknown_of[end]] += direction[out_of_plane_axis[end]]
    rows.add({position: 1 for position in range(len(station_candidates))}, lower=len(unknown_of))

    for _ in range(_SECTION_CUT_ROUNDS):
        result = milp(
            np.ones(len(station_candidates)),
            constraints=rows.to_constraint(len(station_candidates)),
            integrality=np.ones(len(station_candidates)),
            bounds=Bounds(0, 1),
        )
        if result.x is None:
            return None
        selected = [position for position in range(len(station_candidates)) if result.x[position] > 0.5]
        _, singular_values, right_vectors = np.linalg.svd(restraint[selected], full_matrices=True)
        rank = int((singular_values > 1e-9).sum())
        if rank == len(unknown_of):
            break
        for free_motion in right_vectors[rank:]:
            restraining = {
                position: 1
                for position in range(len(station_candidates))
                if abs(restraint[position] @ free_motion) > 1e-7
            }
            rows.add(restraining, lower=1)
    else:
        return None

    pattern = {
        (
            subtract(nodes[candidates[station_candidates[position]][0]], (first_station, 0, 0)),
            subtract(nodes[candidates[station_candidates[position]][1]], (first_station, 0, 0)),
        )
        for position in selected
    }
    chosen = set()
    for candidate in in_section_plane:
        start, end = candidates[candidate]
        station = nodes[start][0]
        if (subtract(nodes[start], (station, 0, 0)), subtract(nodes[end], (station, 0, 0))) in pattern:
            chosen.add(candidate)
    return chosen


def search_constructive(
    lattice: CandidateLattice,
    edges: list[Edge] | None,
    max_trim: float,
    angles_degrees: tuple[float, ...],
    angle_rule: str,
    attempts: int = 8,
    seed: int = 0,
) -> BridgeDesign | None:
    """Build a minimally rigid design by adding members that raise the rigidity rank.

    Members are offered in order: the section bracing pattern, then by
    :data:`ROLE_PRIORITY`. A member is kept if it conflicts with no kept member and is
    linearly independent of the kept members' rows of the rigidity matrix, so a result
    with ``3 * node_count - 6`` members is minimally rigid. Later attempts randomize the
    order within each priority level. Inventory is matched by Glover's algorithm.

    Args:
        lattice: Candidate nodes and members.
        edges: Inventory, or None to search geometry only.
        max_trim: Largest length that may be cut off one edge.
        angles_degrees: Permitted joint angles.
        angle_rule: ``"planar"`` or ``"pairwise"``.
        attempts: Number of member orderings to try.
        seed: Random seed for the reorderings.

    Returns:
        A design, or None if no attempt succeeded. None does not prove that no design exists.
    """
    nodes, candidates = lattice.nodes, lattice.candidates
    node_count = len(nodes)
    node_coordinates = np.asarray(nodes, float) * lattice.unit
    conflicts_of: dict[int, set[int]] = defaultdict(set)
    for first, second in find_conflicting_pairs(lattice, angles_degrees, angle_rule):
        conflicts_of[first].add(second)
        conflicts_of[second].add(first)
    section_bracing = choose_section_bracing(lattice, conflicts_of)
    if section_bracing is None:
        return None
    matrix = rigidity_matrix(node_coordinates, candidates)
    target_rank = 3 * node_count - 6
    random = np.random.default_rng(seed)

    for attempt in range(attempts):
        shuffle_key = random.random(len(candidates)) if attempt else np.zeros(len(candidates))
        order = sorted(
            range(len(candidates)),
            key=lambda candidate: (
                0 if candidate in section_bracing else 1 + ROLE_PRIORITY[lattice.roles[candidate]],
                shuffle_key[candidate],
                nodes[candidates[candidate][0]],
            ),
        )
        independent_rows = np.zeros((0, 3 * node_count))
        chosen: list[int] = []
        chosen_set: set[int] = set()
        for candidate in order:
            if conflicts_of[candidate] & chosen_set:
                continue
            row = matrix[candidate]
            residual = row - independent_rows.T @ (independent_rows @ row) if len(independent_rows) else row.copy()
            residual_norm = np.linalg.norm(residual)
            if residual_norm < 1e-9:
                continue  # redundant: adds no stiffness the kept members do not already give
            independent_rows = np.vstack([independent_rows, residual / residual_norm])
            chosen.append(candidate)
            chosen_set.add(candidate)
            if len(chosen) == target_rank:
                break
        if len(chosen) == target_rank:
            if edges is None:
                return geometry_only_design(lattice, chosen)
            return design_with_matched_edges(lattice, chosen, edges, max_trim)
    return None
