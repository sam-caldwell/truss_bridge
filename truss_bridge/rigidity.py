"""truss_bridge/rigidity.py

Infinitesimal rigidity of pin-jointed frameworks (Asimow & Roth, 1978).

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import numpy as np

_RANK_TOLERANCE = 1e-9
_MECHANISM_TOLERANCE = 1e-7


def rigidity_matrix(node_coordinates: np.ndarray, member_endpoints: list[tuple[int, int]]) -> np.ndarray:
    """Build the rigidity matrix of a framework.

    Row ``r`` holds the unit direction of member ``r`` in the columns of its start node and
    the negated direction in the columns of its end node, so ``R @ velocities`` gives the
    rate of length change of every member.

    Args:
        node_coordinates: Array of shape ``(node_count, 3)``.
        member_endpoints: ``(start_node, end_node)`` for every member.

    Returns:
        Dense array of shape ``(member_count, 3 * node_count)``.
    """
    matrix = np.zeros((len(member_endpoints), node_coordinates.size))
    for row, (start, end) in enumerate(member_endpoints):
        unit_direction = node_coordinates[start] - node_coordinates[end]
        unit_direction = unit_direction / np.linalg.norm(unit_direction)
        matrix[row, 3 * start : 3 * start + 3] = unit_direction
        matrix[row, 3 * end : 3 * end + 3] = -unit_direction
    return matrix


def nontrivial_mechanisms(node_coordinates: np.ndarray, member_endpoints: list[tuple[int, int]]) -> np.ndarray:
    """Infinitesimal motions that change no member length and are not rigid-body motions.

    Args:
        node_coordinates: Array of shape ``(node_count, 3)``.
        member_endpoints: ``(start_node, end_node)`` for every member.

    Returns:
        Orthonormal columns of shape ``(3 * node_count, mechanism_count)``; zero columns
        means the framework is infinitesimally rigid.
    """
    node_count = len(node_coordinates)
    matrix = rigidity_matrix(node_coordinates, member_endpoints)
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    rank = int((singular_values > _RANK_TOLERANCE * max(1.0, singular_values.max(initial=0.0))).sum())
    null_space = right_vectors[rank:].T

    centered = node_coordinates - node_coordinates.mean(axis=0)
    rigid_body_motions = np.zeros((3 * node_count, 6))
    for axis in range(3):
        rigid_body_motions[axis::3, axis] = 1.0  # translation
        axis_vector = np.zeros(3)
        axis_vector[axis] = 1.0
        rigid_body_motions[:, 3 + axis] = np.cross(axis_vector, centered).ravel()  # rotation
    rigid_basis, _ = np.linalg.qr(rigid_body_motions)
    remaining = null_space - rigid_basis @ (rigid_basis.T @ null_space)
    if remaining.shape[1] == 0:
        return remaining
    left_vectors, remaining_values, _ = np.linalg.svd(remaining, full_matrices=False)
    return left_vectors[:, remaining_values > _MECHANISM_TOLERANCE]


def is_rigid(node_coordinates: np.ndarray, member_endpoints: list[tuple[int, int]]) -> bool:
    """Whether the rigidity matrix has the full rank ``3 * node_count - 6``."""
    rank = np.linalg.matrix_rank(rigidity_matrix(node_coordinates, member_endpoints))
    return rank == 3 * len(node_coordinates) - 6
