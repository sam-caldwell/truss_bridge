"""truss_bridge/constraints.py

Joint-angle rules and pairwise conflicts between candidate members.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from .geometry import segments_overlap
from .lattice import CandidateLattice

DEFAULT_ANGLES: tuple[float, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
ANGLE_RULES = ("planar", "pairwise")
_COSINE_TOLERANCE = 1e-9


def permitted_cosines(angles_degrees: tuple[float, ...]) -> list[float]:
    """Distinct cosines of the permitted joint angles."""
    return sorted({round(math.cos(math.radians(angle)), 12) for angle in angles_degrees})


def joint_angle_permitted(
    direction_a: np.ndarray, direction_b: np.ndarray, allowed_cosines: list[float], rule: str = "planar"
) -> bool:
    """Whether two members leaving the same node meet at a permitted angle.

    Args:
        direction_a: Vector from the shared node along the first member.
        direction_b: Vector from the shared node along the second member.
        allowed_cosines: Output of :func:`permitted_cosines`.
        rule: ``"planar"`` checks the angle only when both members lie in a common
            coordinate plane (as seen in an elevation, plan or section view);
            ``"pairwise"`` checks the true 3D angle between every pair.

    Returns:
        True if the pair is allowed.

    Raises:
        ValueError: If ``rule`` is not recognized.
    """
    if rule == "planar":
        plane_normal = np.cross(direction_a, direction_b)
        if np.count_nonzero(np.abs(plane_normal) > 1e-12) > 1:
            return True  # not in a common coordinate plane: unconstrained
    elif rule != "pairwise":
        raise ValueError(f"unknown angle rule {rule!r}; expected one of {ANGLE_RULES}")
    cosine = float(direction_a @ direction_b) / (np.linalg.norm(direction_a) * np.linalg.norm(direction_b))
    return any(abs(cosine - allowed) < _COSINE_TOLERANCE for allowed in allowed_cosines)


def candidate_directions(lattice: CandidateLattice) -> list[np.ndarray]:
    """Lattice-unit direction vector (start to end) of every candidate member."""
    return [np.subtract(lattice.nodes[end], lattice.nodes[start]).astype(float) for start, end in lattice.candidates]


def find_conflicting_pairs(
    lattice: CandidateLattice, angles_degrees: tuple[float, ...], rule: str
) -> list[tuple[int, int]]:
    """Pairs of candidates that cannot both be used.

    Two candidates conflict when they share a node and meet at a forbidden angle, or when
    they cross or overlap anywhere other than a shared end node.

    Returns:
        Candidate index pairs ``(first, second)`` with ``first < second``.
    """
    nodes, candidates = lattice.nodes, lattice.candidates
    directions = candidate_directions(lattice)
    allowed_cosines = permitted_cosines(angles_degrees)
    conflicts: list[tuple[int, int]] = []

    members_at_node: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for candidate_index, (start, end) in enumerate(candidates):
        members_at_node[start].append((candidate_index, directions[candidate_index]))
        members_at_node[end].append((candidate_index, -directions[candidate_index]))
    for incident in members_at_node.values():
        for (first, first_direction), (second, second_direction) in itertools.combinations(incident, 2):
            if not joint_angle_permitted(first_direction, second_direction, allowed_cosines, rule):
                conflicts.append((min(first, second), max(first, second)))

    candidates_spanning_station: dict[int, list[int]] = defaultdict(list)
    for candidate_index, (start, end) in enumerate(candidates):
        low_x, high_x = sorted((nodes[start][0], nodes[end][0]))
        for station in range(low_x, high_x + 1):
            candidates_spanning_station[station].append(candidate_index)
    examined: set[tuple[int, int]] = set()
    for spanning in candidates_spanning_station.values():
        for first, second in itertools.combinations(spanning, 2):
            if (first, second) in examined:
                continue
            examined.add((first, second))
            (first_start, first_end), (second_start, second_end) = candidates[first], candidates[second]
            if segments_overlap(nodes[first_start], nodes[first_end], nodes[second_start], nodes[second_end]):
                conflicts.append((first, second))
    return conflicts
