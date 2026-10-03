"""tests/test_solver.py

End-to-end tests of design search and independent verification.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import dataclasses
import math
import unittest

from truss_bridge import BridgeDesign, Edge, Member, MemberAssignment, NoDesignFoundError, design_bridge, verify_design
from truss_bridge.solver import lattice_units

from .support import cached_design, inventory_for

BUILDABLE = [(12, 3, 3), (24, 3, 3), (12, 4, 2), (16, 2, 4), (4, 2, 1), (12, 6, 3), (7.5, 1.5, 1.5)]


class DesignTests(unittest.TestCase):
    def test_buildable_dimensions_give_valid_designs(self):
        for dimensions in BUILDABLE:
            with self.subTest(dimensions=dimensions):
                design = cached_design(*dimensions)
                self.assertEqual(verify_design(design), [])
                self.assertEqual(len(design.members), 3 * len(design.nodes) - 6)
                span, width, height = (max(n[a] for n in design.nodes) for a in range(3))
                self.assertEqual((span, width, height), tuple(float(d) for d in dimensions))

    def test_chords_are_continuous_along_both_sides(self):
        design = cached_design(12, 3, 3)
        chord_length = sum(m.required_length for m in design.members if m.role in ("bottom_chord", "top_chord"))
        self.assertAlmostEqual(chord_length, 4 * 12)

    def test_unused_edges_complement_used_edges(self):
        edges = inventory_for(12, 3, 3)
        design = design_bridge(edges, 12, 3, 3, max_trim=0.25)
        used = {a.edge.index for a in design.assignments}
        self.assertEqual(used | {e.index for e in design.unused_edges}, {e.index for e in edges})
        self.assertFalse(used & {e.index for e in design.unused_edges})

    def test_exact_search_prefers_zero_trim_stock(self):
        edges = inventory_for(12, 3, 3) + inventory_for(12, 3, 3, slack=0.1)
        edges = [Edge(index, edge.length) for index, edge in enumerate(edges)]
        design = design_bridge(edges, 12, 3, 3, max_trim=0.25)
        self.assertLess(design.total_trim, 0.01)

    def test_four_to_three_section_has_no_design(self):
        with self.assertRaisesRegex(NoDesignFoundError, "no rigid angle-compliant geometry"):
            design_bridge(inventory_for(12, 4, 3), 12, 4, 3)

    def test_pairwise_angle_rule_has_no_design(self):
        with self.assertRaisesRegex(NoDesignFoundError, "no rigid angle-compliant geometry"):
            design_bridge(inventory_for(12, 3, 3), 12, 3, 3, angle_rule="pairwise")

    def test_short_inventory_is_reported_as_such(self):
        with self.assertRaisesRegex(NoDesignFoundError, "inventory cannot supply"):
            design_bridge(inventory_for(12, 3, 3)[:20], 12, 3, 3)

    def test_max_trim_is_respected(self):
        long_stock = inventory_for(12, 3, 3, slack=0.3)
        with self.assertRaisesRegex(NoDesignFoundError, "inventory cannot supply"):
            design_bridge(long_stock, 12, 3, 3, max_trim=0.25)
        design = design_bridge(long_stock, 12, 3, 3, max_trim=0.35)
        self.assertTrue(all(0.3 - 1e-6 <= a.trim <= 0.35 for a in design.assignments))

    def test_invalid_inputs_raise(self):
        edges = inventory_for(12, 3, 3)
        with self.assertRaises(ValueError):
            design_bridge(edges, 0, 3, 3)
        with self.assertRaises(ValueError):
            design_bridge(edges, 12, 3, 3, angle_rule="sideways")
        with self.assertRaises(ValueError):
            design_bridge([Edge(1, 3.0), Edge(1, 3.0)], 12, 3, 3)
        with self.assertRaises(ValueError):
            design_bridge([Edge(1, -3.0)], 12, 3, 3)
        with self.assertRaises(ValueError):
            design_bridge(edges, math.pi, 3, 3)

    def test_lattice_unit_is_greatest_common_divisor(self):
        self.assertEqual(lattice_units(12, 3, 3), [3.0])
        self.assertEqual(lattice_units(7.5, 1.5, 1.5, refinement=2), [1.5, 0.75])
        self.assertEqual(lattice_units(16, 2, 4), [2.0])


class VerificationTests(unittest.TestCase):
    def test_detects_missing_member(self):
        design = cached_design(12, 3, 3)
        broken = dataclasses.replace(design, assignments=design.assignments[1:])
        self.assertTrue(any("rank" in violation for violation in verify_design(broken)))

    def test_detects_reused_and_short_edges(self):
        design = cached_design(12, 3, 3)
        first, second = design.assignments[:2]
        short_edge = Edge(first.edge.index, first.member.required_length - 0.01)
        broken = dataclasses.replace(
            design,
            assignments=[MemberAssignment(first.member, short_edge), MemberAssignment(second.member, short_edge)]
            + design.assignments[2:],
        )
        violations = verify_design(broken)
        self.assertTrue(any("shorter" in violation for violation in violations))
        self.assertTrue(any("more than once" in violation for violation in violations))

    def test_detects_forbidden_angle_and_crossing(self):
        nodes = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        members = [
            Member(0, 1, "x", 2.0),
            Member(0, 2, "x", math.sqrt(5)),  # meets member 0 at about 26.57 degrees
            Member(1, 3, "x", math.sqrt(5)),  # crosses member 1
        ]
        design = BridgeDesign("test", 1.0, nodes, [MemberAssignment(m, Edge(i, 9.0)) for i, m in enumerate(members)])
        violations = verify_design(design)
        self.assertTrue(any("26.57 degrees" in violation for violation in violations))
        self.assertTrue(any("cross" in violation for violation in violations))


if __name__ == "__main__":
    unittest.main()
