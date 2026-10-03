"""tests/test_assignment.py

Tests for matching inventory edges to members, checked against brute force.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import itertools
import random
import unittest

from truss_bridge import Edge, Member, assign_edges_to_members


def brute_force_least_trim(members, edges, max_trim):
    """Least total trim over every injective assignment, or None if none fits."""
    best = None
    for chosen in itertools.permutations(edges, len(members)):
        fits = all(
            m.required_length <= e.length + 1e-9 <= m.required_length + max_trim + 2e-9
            for m, e in zip(members, chosen, strict=True)
        )
        if fits:
            trim = sum(e.length - m.required_length for m, e in zip(members, chosen, strict=True))
            best = trim if best is None else min(best, trim)
    return best


class AssignmentTests(unittest.TestCase):
    def test_matches_brute_force_on_random_cases(self):
        generator = random.Random(1)
        for _ in range(400):
            members = [Member(0, 1, "x", round(generator.uniform(1, 3), 2)) for _ in range(4)]
            edges = [Edge(index, round(generator.uniform(1, 3.5), 2)) for index in range(6)]
            max_trim = generator.choice([0.2, 0.5, 1.0])
            expected = brute_force_least_trim(members, edges, max_trim)
            result = assign_edges_to_members(members, edges, max_trim)
            self.assertEqual(result is None, expected is None, (members, edges, max_trim))
            if result is not None:
                self.assertAlmostEqual(sum(a.trim for a in result), expected, places=9)

    def test_each_edge_used_once_and_fits(self):
        members = [Member(0, 1, "x", 2.0) for _ in range(3)]
        edges = [Edge(7, 2.1), Edge(8, 2.0), Edge(9, 2.05), Edge(10, 5.0)]
        result = assign_edges_to_members(members, edges, max_trim=0.2)
        self.assertEqual(sorted(a.edge.index for a in result), [7, 8, 9])
        self.assertTrue(all(0 <= a.trim <= 0.2 for a in result))

    def test_returns_none_when_stock_is_short(self):
        members = [Member(0, 1, "x", 2.0) for _ in range(2)]
        self.assertIsNone(assign_edges_to_members(members, [Edge(0, 2.0)], max_trim=1.0))

    def test_edges_are_never_stretched(self):
        members = [Member(0, 1, "x", 2.0)]
        self.assertIsNone(assign_edges_to_members(members, [Edge(0, 1.99)], max_trim=1.0))


if __name__ == "__main__":
    unittest.main()
