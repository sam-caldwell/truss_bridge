"""tests/test_rigidity.py

Tests for the infinitesimal rigidity checks.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import itertools
import unittest

import numpy as np

from truss_bridge.rigidity import is_rigid, nontrivial_mechanisms

TETRAHEDRON = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
CUBE = np.array(list(itertools.product([0, 1], repeat=3)), float)
CUBE_EDGES = [(a, b) for a, b in itertools.combinations(range(8), 2) if np.abs(CUBE[a] - CUBE[b]).sum() == 1]


class RigidityTests(unittest.TestCase):
    def test_tetrahedron_is_rigid(self):
        edges = list(itertools.combinations(range(4), 2))
        self.assertTrue(is_rigid(TETRAHEDRON, edges))
        self.assertEqual(nontrivial_mechanisms(TETRAHEDRON, edges).shape[1], 0)

    def test_tetrahedron_missing_an_edge_has_one_mechanism(self):
        edges = list(itertools.combinations(range(4), 2))[1:]
        self.assertFalse(is_rigid(TETRAHEDRON, edges))
        self.assertEqual(nontrivial_mechanisms(TETRAHEDRON, edges).shape[1], 1)

    def test_bare_cube_frame_has_six_mechanisms(self):
        # 8 nodes need 3*8 - 6 = 18 bars; the 12 edges leave 6 independent mechanisms.
        self.assertEqual(nontrivial_mechanisms(CUBE, CUBE_EDGES).shape[1], 6)

    def test_cube_with_one_diagonal_per_face_is_rigid(self):
        face_diagonals = [(0, 3), (4, 7), (0, 5), (2, 7), (0, 6), (1, 7)]
        self.assertTrue(is_rigid(CUBE, CUBE_EDGES + face_diagonals))


if __name__ == "__main__":
    unittest.main()
