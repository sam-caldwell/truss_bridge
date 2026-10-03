"""tests/test_geometry.py

Tests for exact lattice geometry.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import unittest

from truss_bridge.geometry import classify_member, is_lattice_direction, lies_strictly_inside_segment, segments_overlap


class LatticeDirectionTests(unittest.TestCase):
    def test_axes_and_planar_diagonals_are_allowed(self):
        for direction in [(1, 0, 0), (0, -3, 0), (0, 0, 2), (2, 2, 0), (1, 0, -1), (0, 4, 4)]:
            self.assertTrue(is_lattice_direction(direction), direction)

    def test_other_directions_are_rejected(self):
        for direction in [(0, 0, 0), (1, 1, 1), (2, 1, 0), (1, 2, 3), (3, 0, 1)]:
            self.assertFalse(is_lattice_direction(direction), direction)


class SegmentTests(unittest.TestCase):
    def test_point_strictly_inside(self):
        self.assertTrue(lies_strictly_inside_segment((1, 1, 0), (0, 0, 0), (2, 2, 0)))
        self.assertFalse(lies_strictly_inside_segment((2, 2, 0), (0, 0, 0), (2, 2, 0)))
        self.assertFalse(lies_strictly_inside_segment((1, 0, 0), (0, 0, 0), (2, 2, 0)))

    def test_crossing_diagonals_overlap(self):
        self.assertTrue(segments_overlap((0, 0, 0), (2, 2, 0), (0, 2, 0), (2, 0, 0)))

    def test_members_sharing_only_an_endpoint_do_not_overlap(self):
        self.assertFalse(segments_overlap((0, 0, 0), (2, 0, 0), (2, 0, 0), (2, 2, 0)))

    def test_collinear_overlap_is_detected(self):
        self.assertTrue(segments_overlap((0, 0, 0), (3, 0, 0), (2, 0, 0), (5, 0, 0)))

    def test_collinear_end_to_end_members_do_not_overlap(self):
        self.assertFalse(segments_overlap((0, 0, 0), (2, 0, 0), (2, 0, 0), (4, 0, 0)))

    def test_skew_segments_do_not_overlap(self):
        self.assertFalse(segments_overlap((0, 0, 0), (2, 0, 0), (1, -1, 1), (1, 1, 1)))

    def test_t_junction_mid_span_overlaps(self):
        self.assertTrue(segments_overlap((0, 0, 0), (2, 0, 0), (1, 0, 0), (1, 2, 0)))


class RoleTests(unittest.TestCase):
    def classify(self, start, end):
        return classify_member(start, end, width_units=2, height_units=4)

    def test_roles(self):
        expected = {
            ((0, 0, 0), (1, 0, 0)): "bottom_chord",
            ((0, 2, 4), (1, 2, 4)): "top_chord",
            ((0, 0, 2), (1, 0, 2)): "stringer",
            ((0, 0, 0), (0, 2, 0)): "floor_beam",
            ((0, 0, 4), (0, 2, 4)): "top_strut",
            ((0, 0, 2), (0, 2, 2)): "cross_strut",
            ((0, 0, 0), (0, 0, 2)): "vertical",
            ((0, 0, 0), (2, 0, 2)): "side_diagonal",
            ((0, 0, 0), (2, 2, 0)): "bottom_lateral",
            ((0, 0, 4), (2, 2, 4)): "top_lateral",
            ((0, 0, 0), (0, 2, 2)): "sway_brace",
        }
        for (start, end), role in expected.items():
            self.assertEqual(self.classify(start, end), role, (start, end))


if __name__ == "__main__":
    unittest.main()
