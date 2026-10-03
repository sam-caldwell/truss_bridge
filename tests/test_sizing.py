"""tests/test_sizing.py

Tests for automatic member sizing and the section checks it relies on.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import math
import os
import tempfile
import unittest

import numpy as np

from truss_bridge import (
    CrossSection,
    SizingError,
    basic_load_cases,
    plot_design,
    run_load_tests,
    size_members,
    square_hollow_catalog,
)
from truss_bridge.analysis import (
    SLENDER_WALL,
    SLENDERNESS_LIMIT,
    analyze_load_case,
    check_member,
    has_slender_walls,
    slender_wall_limit,
)

from .support import STEEL, cached_design

CATALOG = square_hollow_catalog()


def load_cases(design):
    return basic_load_cases(design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)


class SectionCheckTests(unittest.TestCase):
    def test_catalog_is_sorted_and_filtered(self):
        areas = [section.area for section in CATALOG]
        self.assertEqual(areas, sorted(areas))
        self.assertTrue(all(section.wall_slenderness >= 0 for section in CATALOG))
        names = {section.name for section in CATALOG}
        self.assertIn("SHS 100x100x5 mm", names)
        self.assertNotIn("SHS 40x40x16 mm", names)  # wall thicker than a quarter of the width

    def test_wall_slenderness_follows_b_minus_three_t(self):
        section = CrossSection.square_hollow(0.3, 0.003)
        self.assertAlmostEqual(section.wall_slenderness, (300 - 9) / 3)
        self.assertAlmostEqual(slender_wall_limit(STEEL), 1.40 * math.sqrt(200e9 / 250e6))
        self.assertTrue(has_slender_walls(section, STEEL))
        self.assertFalse(has_slender_walls(CrossSection.square_hollow(0.1, 0.005), STEEL))

    def test_slender_wall_fails_only_in_compression(self):
        section = CrossSection.square_hollow(0.3, 0.003)
        _, utilization, limit_state = check_member(2.0, -1e3, section, STEEL)
        self.assertEqual((utilization, limit_state), (math.inf, SLENDER_WALL))
        _, utilization, limit_state = check_member(2.0, 1e3, section, STEEL)
        self.assertLess(utilization, 1)
        self.assertEqual(limit_state, "tension yielding")

    def test_compression_slenderness_limit(self):
        section = CrossSection.square_hollow(0.04, 0.003)
        long_member = 201 * section.radius_of_gyration
        _, utilization, limit_state = check_member(
            long_member, -1.0, section, STEEL, max_compression_slenderness=200
        )
        self.assertEqual((utilization, limit_state), (math.inf, SLENDERNESS_LIMIT))
        _, utilization, _ = check_member(long_member, -1.0, section, STEEL)
        self.assertLess(utilization, 1)

    def test_per_member_sections_match_uniform_section(self):
        design = cached_design(12, 3, 3)
        section = CrossSection.square_hollow(0.1, 0.005)
        case = load_cases(design)[1]
        uniform = analyze_load_case(design, case, sections=section, materials=STEEL)
        listed = analyze_load_case(design, case, sections=[section] * len(design.members), materials=STEEL)
        np.testing.assert_array_equal(uniform.response.axial_forces, listed.response.axial_forces)
        with self.assertRaises(ValueError):
            analyze_load_case(design, case, sections=[section], materials=STEEL)


class CatalogStepTests(unittest.TestCase):
    def test_next_heavier_skips_equal_areas(self):
        from truss_bridge.sizing import _next_heavier

        equal = [CrossSection("a", 1.0, 2.0), CrossSection("b", 1.0, 1.0), CrossSection("c", 2.0, 1.0)]
        self.assertEqual(_next_heavier(equal, 0), 2)
        self.assertEqual(_next_heavier(equal, 1), 2)
        self.assertIsNone(_next_heavier(equal, 2))

    def test_catalog_contains_equal_area_sections(self):
        # 70x70x4 and 50x50x6 both have A = 4t(b - t) = 1056 mm^2, so the equal-area case is real.
        areas = {round(s.area * 1e6, 6) for s in CATALOG if s.name in ("SHS 70x70x4 mm", "SHS 50x50x6 mm")}
        self.assertEqual(areas, {1056.0})


class SizingTests(unittest.TestCase):
    def test_role_sizing_passes_and_no_group_can_be_lighter(self):
        design = cached_design(12, 3, 3)
        cases = load_cases(design)
        result = size_members(design, cases, CATALOG, materials=STEEL)
        self.assertTrue(result.report.passed)
        for group, members in result.groups.items():
            chosen = result.group_sections[group]
            for lighter in (section for section in CATALOG if section.area < chosen.area):
                sections = list(result.member_sections)
                for index in members:
                    sections[index] = lighter
                trial = run_load_tests(
                    design, cases, sections=sections, materials=STEEL, max_compression_slenderness=200
                )
                with self.subTest(group=group, lighter=lighter.name):
                    self.assertFalse(trial.passed)

    def test_every_member_in_a_role_group_shares_one_section(self):
        design = cached_design(16, 2, 4)
        result = size_members(design, load_cases(design), CATALOG, materials=STEEL)
        for group, members in result.groups.items():
            self.assertEqual({result.member_sections[i].name for i in members}, {result.group_sections[group].name})
            self.assertEqual({design.members[i].role for i in members}, {group})

    def test_per_member_sizing_is_no_heavier_than_per_role(self):
        for dimensions in [(12, 3, 3), (16, 2, 4)]:
            design = cached_design(*dimensions)
            cases = load_cases(design)
            by_role = size_members(design, cases, CATALOG, materials=STEEL)
            by_member = size_members(design, cases, CATALOG, materials=STEEL, group_by="member")
            with self.subTest(dimensions=dimensions):
                self.assertTrue(by_member.report.passed)
                self.assertEqual(len(by_member.groups), len(design.members))
                self.assertLessEqual(by_member.total_mass, by_role.total_mass * 1.001)

    def test_deflection_limit_is_met_and_costs_mass(self):
        design = cached_design(12, 3, 3)
        cases = load_cases(design)
        free = size_members(design, cases, CATALOG, materials=STEEL)
        limited = size_members(design, cases, CATALOG, materials=STEEL, deflection_limit_ratio=5000)
        self.assertTrue(limited.report.passed)
        self.assertLessEqual(limited.report.max_vertical_deflection, 12 / 5000)
        self.assertGreater(free.report.max_vertical_deflection, 12 / 5000)
        self.assertGreater(limited.total_mass, free.total_mass)
        self.assertGreater(limited.deflection_steps, 0)

    def test_unreachable_deflection_limit_is_reported(self):
        design = cached_design(16, 2, 4)
        with self.assertRaisesRegex(SizingError, "cannot be met"):
            size_members(design, load_cases(design), CATALOG, materials=STEEL, deflection_limit_ratio=10000)

    def test_too_small_catalog_is_reported(self):
        design = cached_design(12, 3, 3)
        with self.assertRaisesRegex(SizingError, "no catalog section is strong enough"):
            size_members(design, load_cases(design), square_hollow_catalog((40,), (3,)), materials=STEEL)

    def test_invalid_arguments(self):
        design = cached_design(12, 3, 3)
        with self.assertRaises(ValueError):
            size_members(design, load_cases(design), CATALOG, materials=STEEL, group_by="color")
        with self.assertRaises(ValueError):
            size_members(design, load_cases(design), [], materials=STEEL)

    def test_summary_and_drawing(self):
        design = cached_design(12, 3, 3)
        result = size_members(design, load_cases(design), CATALOG, materials=STEEL)
        self.assertIn("PASS", result.summary())
        utilization = np.max([[c.utilization for c in r.member_checks] for r in result.report.results], axis=0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "utilization.png")
            plot_design(design, path, member_utilization=utilization, member_sections=result.member_sections)
            self.assertGreater(os.path.getsize(path), 10_000)
            with self.assertRaises(ValueError):
                plot_design(design, path, member_forces=utilization, member_utilization=utilization)


if __name__ == "__main__":
    unittest.main()
