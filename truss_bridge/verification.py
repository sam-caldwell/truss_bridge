"""truss_bridge/verification.py

Independent checks that a design meets every requirement.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from .constraints import DEFAULT_ANGLES, joint_angle_permitted, permitted_cosines
from .geometry import segments_overlap
from .model import BridgeDesign
from .rigidity import rigidity_matrix


def verify_design(
    design: BridgeDesign, angles_degrees: tuple[float, ...] = DEFAULT_ANGLES, angle_rule: str = "planar"
) -> list[str]:
    """Check a design without trusting the search that produced it.

    Checks every joint angle, that no two members cross, that every node is used, that the
    rigidity matrix has rank ``3 * node_count - 6``, that no edge is shorter than its
    member, and that no edge is used twice.

    Args:
        design: The design to check.
        angles_degrees: Permitted joint angles.
        angle_rule: ``"planar"`` or ``"pairwise"``.

    Returns:
        Descriptions of every violation; an empty list means the design is valid.
    """
    coordinates = np.asarray(design.nodes, float)
    allowed_cosines = permitted_cosines(angles_degrees)
    endpoints = [(member.start_node, member.end_node) for member in design.members]
    violations: list[str] = []

    members_at_node: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for member_index, (start, end) in enumerate(endpoints):
        members_at_node[start].append((member_index, coordinates[end] - coordinates[start]))
        members_at_node[end].append((member_index, coordinates[start] - coordinates[end]))
    for node, incident in members_at_node.items():
        for (first, first_direction), (second, second_direction) in itertools.combinations(incident, 2):
            if not joint_angle_permitted(first_direction, second_direction, allowed_cosines, angle_rule):
                norms = np.linalg.norm(first_direction) * np.linalg.norm(second_direction)
                cosine = first_direction @ second_direction / norms
                angle = math.degrees(math.acos(np.clip(cosine, -1, 1)))
                violations.append(f"node {node}: members {first} and {second} meet at {angle:.2f} degrees")

    lattice_points = [tuple(int(round(c / design.lattice_unit)) for c in node) for node in design.nodes]
    for (first, (a_start, a_end)), (second, (b_start, b_end)) in itertools.combinations(enumerate(endpoints), 2):
        a_segment = (lattice_points[a_start], lattice_points[a_end])
        if segments_overlap(*a_segment, lattice_points[b_start], lattice_points[b_end]):
            violations.append(f"members {first} and {second} cross")

    node_count = len(coordinates)
    if len(members_at_node) != node_count:
        violations.append(f"{node_count - len(members_at_node)} node(s) have no members")
    rank = np.linalg.matrix_rank(rigidity_matrix(coordinates, endpoints))
    if rank != 3 * node_count - 6:
        violations.append(f"rigidity matrix rank {rank}, need {3 * node_count - 6}")

    for assignment in design.assignments:
        if assignment.trim < -1e-9:
            violations.append(f"edge {assignment.edge.index} is shorter than its member")
    edge_indices = [assignment.edge.index for assignment in design.assignments]
    if len(set(edge_indices)) != len(edge_indices):
        violations.append("an edge is used more than once")
    return violations
