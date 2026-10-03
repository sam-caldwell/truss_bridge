"""tests/test_performance.py

Load and scale tests: large spans, large inventories, many load cases.

Time limits are about four times the durations measured on the development machine, so
they catch order-of-magnitude regressions without being flaky. Set the environment
variable ``TRUSS_BRIDGE_SKIP_PERFORMANCE=1`` to skip this module (about a minute).

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import os
import random
import time
import unittest

from truss_bridge import (
    CrossSection,
    Edge,
    Member,
    SizingError,
    assign_edges_to_members,
    basic_load_cases,
    design_bridge,
    run_load_tests,
    size_members,
    verify_design,
)

from .support import STEEL, inventory_for

SKIP = os.environ.get("TRUSS_BRIDGE_SKIP_PERFORMANCE") == "1"
SECTION = CrossSection.square_hollow(0.15, 0.008)


def timed(function):
    """Run ``function`` and return ``(result, seconds)``."""
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


@unittest.skipIf(SKIP, "TRUSS_BRIDGE_SKIP_PERFORMANCE=1")
class ScaleTests(unittest.TestCase):
    def assert_valid_and_loadable(self, design):
        self.assertEqual(verify_design(design), [])
        cases = basic_load_cases(design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)
        report, seconds = timed(lambda: run_load_tests(design, cases, sections=SECTION, materials=STEEL))
        for result in report.results:
            self.assertLess(result.equilibrium_error, 1e-6 * max(1.0, abs(result.applied_force).max()))
        return seconds

    def test_twenty_panel_span_is_solved_exactly(self):
        design, seconds = timed(lambda: design_bridge(inventory_for(60, 3, 3, copies=400), 60, 3, 3, max_trim=0.25))
        self.assertEqual(design.search_method, "exact")
        self.assertEqual(len(design.members), 246)
        self.assertLess(seconds, 30)
        self.assertLess(self.assert_valid_and_loadable(design), 2)

    def test_eighty_panel_span_falls_back_to_constructive_search(self):
        edges = inventory_for(240, 3, 3, copies=2000)
        design, seconds = timed(lambda: design_bridge(edges, 240, 3, 3, max_trim=0.25, exact_time_budget=5))
        self.assertEqual(design.search_method, "constructive")
        self.assertEqual(len(design.members), 3 * len(design.nodes) - 6)
        self.assertLess(seconds, 30)
        self.assertLess(self.assert_valid_and_loadable(design), 12)

    def test_dense_face_lattice_long_span(self):
        design, seconds = timed(lambda: design_bridge(inventory_for(64, 2, 4, copies=2000), 64, 2, 4, max_trim=0.25))
        self.assertEqual(len(design.nodes), 198)
        self.assertLess(seconds, 8)
        self.assertLess(self.assert_valid_and_loadable(design), 4)

    def test_large_inventory_assignment(self):
        generator = random.Random(0)
        lengths = [1.0, 2.0, 3.0, 1.4142, 2.8284]
        members = [Member(0, 1, "x", generator.choice(lengths)) for _ in range(20_000)]
        edges = [Edge(index, m.required_length + generator.uniform(0, 0.3)) for index, m in enumerate(members * 5)]
        assignments, seconds = timed(lambda: assign_edges_to_members(members, edges, 0.3))
        self.assertEqual(len(assignments), 20_000)
        self.assertEqual(len({a.edge.index for a in assignments}), 20_000)
        self.assertLess(seconds, 1)


@unittest.skipIf(SKIP, "TRUSS_BRIDGE_SKIP_PERFORMANCE=1")
class SizingScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = design_bridge(inventory_for(32, 2, 4, copies=2000), 32, 2, 4, max_trim=0.25)
        cls.cases = basic_load_cases(cls.design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)

    def test_many_load_cases_share_one_factorization(self):
        design = design_bridge(inventory_for(240, 3, 3, copies=2000), 240, 3, 3, max_trim=0.25, exact_time_budget=5)
        cases = basic_load_cases(design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)
        report, seconds = timed(lambda: run_load_tests(design, cases, sections=SECTION, materials=STEEL))
        self.assertEqual(len(report.results), 84)
        self.assertLess(seconds, 2)

    def test_role_sizing_of_a_300_member_bridge(self):
        result, seconds = timed(lambda: size_members(self.design, self.cases, materials=STEEL))
        self.assertTrue(result.report.passed)
        self.assertLess(seconds, 5)

    def test_member_sizing_of_a_300_member_bridge(self):
        result, seconds = timed(lambda: size_members(self.design, self.cases, materials=STEEL, group_by="member"))
        self.assertTrue(result.report.passed)
        self.assertEqual(len(result.groups), 300)
        self.assertLess(seconds, 90)

    def test_unreachable_requirements_fail_fast(self):
        _, seconds = timed(
            lambda: self.assertRaises(
                SizingError, size_members, self.design, self.cases, materials=STEEL, deflection_limit_ratio=2000
            )
        )
        self.assertLess(seconds, 10)


if __name__ == "__main__":
    unittest.main()
