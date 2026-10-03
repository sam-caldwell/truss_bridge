"""truss_bridge/exact_search.py

Exact design search by mixed-integer programming with lazy rigidity cuts.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, milp

from .constraints import candidate_directions, find_conflicting_pairs
from .design_builder import geometry_only_design, members_from_candidates
from .lattice import CandidateLattice
from .linear_rows import LinearRows
from .model import BridgeDesign, Edge, MemberAssignment, SearchBudgetExceededError
from .rigidity import nontrivial_mechanisms

MILP_INFEASIBLE = 2
"""``scipy.optimize.milp`` status code for a proven-infeasible problem."""

MEMBER_COUNT_TIE_BREAK = 1e-4
"""Cost per member, so that among equal-trim designs the one with fewer members wins."""

_LENGTH_DIGITS = 9
_STRAIN_TOLERANCE = 1e-7


def _planes_through_lattice_directions(directions: list[np.ndarray]) -> list[np.ndarray]:
    """Unit normals of every plane spanned by two distinct member directions."""

    def canonical(direction: np.ndarray) -> tuple[float, ...]:
        signs = np.sign(direction)
        return tuple(signs * signs[np.nonzero(signs)[0][0]])

    lines = [np.array(line) for line in {canonical(direction) for direction in directions}]
    normals: list[np.ndarray] = []
    for first, second in itertools.combinations(lines, 2):
        normal = np.cross(first, second)
        if np.linalg.norm(normal) == 0:
            continue
        normal = normal / np.linalg.norm(normal)
        if not any(abs(abs(normal @ existing) - 1) < 1e-9 for existing in normals):
            normals.append(normal)
    return normals


def search_exact(
    lattice: CandidateLattice,
    edges: list[Edge] | None,
    max_trim: float,
    angles_degrees: tuple[float, ...],
    angle_rule: str,
    max_cut_rounds: int = 300,
    time_budget: float = 30.0,
) -> BridgeDesign | None:
    """Find the least-trim rigid design on a node set, or prove none exists.

    Binary variables choose candidate members and, for each inventory edge, the member
    length it will be cut to. Constraints:

    * conflicting candidates (forbidden angle, or crossing) are mutually exclusive;
    * the members at every node must not all lie in one plane;
    * chord members are mandatory;
    * at least ``3 * node_count - 6`` members (Maxwell's count for a rigid 3D frame);
    * each edge is used at most once, and every chosen member gets one fitting edge.

    Rigidity is then enforced lazily: if the solution has a mechanism ``u``, every rigid
    design must contain some member that ``u`` stretches, which is added as a cut and the
    problem re-solved (Asimow & Roth, 1978).

    Args:
        lattice: Candidate nodes and members.
        edges: Inventory, or None to search geometry only.
        max_trim: Largest length that may be cut off one edge.
        angles_degrees: Permitted joint angles.
        angle_rule: ``"planar"`` or ``"pairwise"``.
        max_cut_rounds: Maximum number of solve / cut iterations.
        time_budget: Total wall-clock seconds allowed for all solves.

    Returns:
        The optimal design, or None if the problem is proven infeasible.

    Raises:
        SearchBudgetExceededError: If the time budget or ``max_cut_rounds`` runs out before
            a rigid design is found or infeasibility is proven. The number of non-rigid
            candidate designs grows exponentially with the number of panels, so long spans
            can exceed any fixed budget.
    """
    nodes, candidates = lattice.nodes, lattice.candidates
    candidate_count, node_count = len(candidates), len(nodes)
    node_coordinates = np.asarray(nodes, float) * lattice.unit
    directions = candidate_directions(lattice)
    member_lengths = [float(np.linalg.norm(direction)) * lattice.unit for direction in directions]

    rows = LinearRows()
    for first, second in find_conflicting_pairs(lattice, angles_degrees, angle_rule):
        rows.at_most_one(first, second)

    members_at_node: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for candidate, (start, end) in enumerate(candidates):
        members_at_node[start].append((candidate, directions[candidate]))
        members_at_node[end].append((candidate, -directions[candidate]))
    for plane_normal in _planes_through_lattice_directions(directions):
        for incident in members_at_node.values():
            leaving_plane = {candidate: 1 for candidate, direction in incident if abs(plane_normal @ direction) > 1e-9}
            rows.add(leaving_plane, lower=1)

    mandatory = [candidate for candidate, role in enumerate(lattice.roles) if role in ("bottom_chord", "top_chord")]

    # Inventory: one variable per (edge, length class) pair where the edge fits that length.
    candidates_by_length: dict[float, list[int]] = defaultdict(list)
    for candidate, length in enumerate(member_lengths):
        candidates_by_length[round(length, _LENGTH_DIGITS)].append(candidate)
    length_classes = sorted(candidates_by_length)
    fitting_pairs = (
        []
        if edges is None
        else [
            (edge_position, class_position)
            for edge_position, edge in enumerate(edges)
            for class_position, length in enumerate(length_classes)
            if length - 1e-9 <= edge.length <= length + max_trim + 1e-9
        ]
    )
    variable_count = candidate_count + len(fitting_pairs)
    pair_variables_by_edge: dict[int, list[int]] = defaultdict(list)
    pair_variables_by_class: dict[int, list[int]] = defaultdict(list)
    for pair_position, (edge_position, class_position) in enumerate(fitting_pairs):
        pair_variables_by_edge[edge_position].append(candidate_count + pair_position)
        pair_variables_by_class[class_position].append(candidate_count + pair_position)
    for variables in pair_variables_by_edge.values():
        rows.add({variable: 1 for variable in variables}, upper=1)
    if edges is not None:
        for class_position, length in enumerate(length_classes):
            balance = {variable: 1.0 for variable in pair_variables_by_class[class_position]}
            for candidate in candidates_by_length[length]:
                balance[candidate] = -1.0
            rows.add(balance, lower=0, upper=0)
    # A least-cost rigid design is minimally rigid: any extra member only adds cost, and a
    # rigid frame always contains a minimally rigid subset that keeps the chords.
    minimal_count = 3 * node_count - 6
    rows.add({candidate: 1 for candidate in range(candidate_count)}, lower=minimal_count, upper=minimal_count)

    cost = np.zeros(variable_count)
    cost[:candidate_count] = MEMBER_COUNT_TIE_BREAK
    for pair_position, (edge_position, class_position) in enumerate(fitting_pairs):
        cost[candidate_count + pair_position] = edges[edge_position].length - length_classes[class_position]
    lower_bounds = np.zeros(variable_count)
    lower_bounds[mandatory] = 1

    deadline = time.monotonic() + time_budget
    for _ in range(max_cut_rounds):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result = milp(
            cost,
            constraints=rows.to_constraint(variable_count),
            integrality=np.ones(variable_count),
            bounds=Bounds(lower_bounds, np.ones(variable_count)),
            options={"time_limit": remaining},
        )
        if result.status == MILP_INFEASIBLE:
            return None
        if result.x is None:
            raise SearchBudgetExceededError(f"MILP stopped without a solution ({result.message})")
        chosen = [candidate for candidate in range(candidate_count) if result.x[candidate] > 0.5]
        mechanisms = nontrivial_mechanisms(node_coordinates, [candidates[candidate] for candidate in chosen])
        if mechanisms.shape[1] == 0:
            if edges is None:
                return geometry_only_design(lattice, chosen)
            return _design_from_solution(lattice, chosen, edges, fitting_pairs, length_classes, result.x)
        for mechanism in mechanisms.T:
            velocities = mechanism.reshape(-1, 3)
            velocities = velocities / np.abs(velocities).max()
            stretched = {
                candidate: 1
                for candidate, (start, end) in enumerate(candidates)
                if abs((velocities[start] - velocities[end]) @ directions[candidate])
                > _STRAIN_TOLERANCE * np.linalg.norm(directions[candidate])
            }
            rows.add(stretched, lower=1)
    raise SearchBudgetExceededError(f"exact search budget exhausted ({max_cut_rounds} rounds or {time_budget:g} s)")


def _design_from_solution(
    lattice: CandidateLattice,
    chosen: list[int],
    edges: list[Edge],
    fitting_pairs: list[tuple[int, int]],
    length_classes: list[float],
    solution: np.ndarray,
) -> BridgeDesign:
    """Read the chosen members and their edges out of a MILP solution vector."""
    candidate_count = len(lattice.candidates)
    edges_for_class: dict[int, list[Edge]] = defaultdict(list)
    for pair_position, (edge_position, class_position) in enumerate(fitting_pairs):
        if solution[candidate_count + pair_position] > 0.5:
            edges_for_class[class_position].append(edges[edge_position])
    class_of_length = {length: position for position, length in enumerate(length_classes)}
    nodes, members, _ = members_from_candidates(lattice, chosen)
    assignments = [
        MemberAssignment(member, edges_for_class[class_of_length[round(member.required_length, _LENGTH_DIGITS)]].pop())
        for member in members
    ]
    used = {assignment.edge.index for assignment in assignments}
    return BridgeDesign(
        lattice.node_set, lattice.unit, nodes, assignments, [edge for edge in edges if edge.index not in used]
    )
