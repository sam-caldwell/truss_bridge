"""tests/test_inventory.py

Tests for inventory CSV files, per-part materials and the non-steel strength method.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import dataclasses
import math
import os
import tempfile
import unittest

import numpy as np

from truss_bridge import (
    CrossSection,
    Material,
    MemberAssignment,
    basic_load_cases,
    example_inventory,
    read_inventory,
    run_load_tests,
    size_members,
    square_hollow_catalog,
    write_inventory,
)
from truss_bridge.analysis import check_member, nominal_compressive_strength, self_weight_forces

from .support import PLASTIC, STEEL, cached_design

SECTION = CrossSection.square_hollow(0.1, 0.005)
HEADER = "index,length,material,elastic_modulus,yield_strength,density,strength_method,safety_factor"
PVC = "pvc,3e9,5e7,1400,yield_euler,3"
REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def with_parts(design, materials, section=SECTION):
    """The design with every edge carrying the given material (one per member) and section."""
    assignments = [
        MemberAssignment(a.member, dataclasses.replace(a.edge, material=material, section=section))
        for a, material in zip(design.assignments, materials, strict=True)
    ]
    return dataclasses.replace(design, assignments=assignments)


class InventoryFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, text):
        path = os.path.join(self.directory.name, "inventory.csv")
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def test_round_trip_keeps_materials_sections_and_unknowns(self):
        edges = example_inventory(12, 3, 3)
        edges[5] = dataclasses.replace(edges[5], material=None, section=None)
        path = write_inventory(os.path.join(self.directory.name, "out.csv"), edges)
        self.assertEqual(read_inventory(path), edges)
        self.assertEqual({edge.material.name for edge in edges if edge.material}, {"steel", "gfrp"})

    def test_repository_example_matches_generator(self):
        self.assertEqual(read_inventory(os.path.join(REPOSITORY, "example.csv")), example_inventory(12, 3, 3))

    def test_length_only_files_still_read(self):
        for text in ("0,3.0\n1,4.25\n", "index,length\n0,3.0\n1,4.25\n", "length,index\n3.0,0\n4.25,1\n"):
            with self.subTest(text=text):
                edges = read_inventory(self.write(text))
                self.assertEqual([(edge.index, edge.length) for edge in edges], [(0, 3.0), (1, 4.25)])
                self.assertTrue(all(edge.material is None and edge.section is None for edge in edges))

    def test_headerless_rows_use_documented_column_order(self):
        (edge,) = read_inventory(self.write("7,3.0,pvc,3e9,5e7,1400,yield_euler,3,0.1,0.05\n"))
        self.assertEqual(edge.material, Material("pvc", 3e9, 5e7, 1400, "yield_euler", 3))
        self.assertEqual(edge.section, CrossSection.square_hollow(0.1, 0.05))  # solid square bar

    def test_rows_naming_one_material_share_one_object(self):
        edges = read_inventory(self.write(f"{HEADER}\n0,3,{PVC}\n1,3,{PVC}\n"))
        self.assertIs(edges[0].material, edges[1].material)

    def test_malformed_files_are_rejected_with_line_numbers(self):
        cases = {
            "different properties": f"{HEADER}\n0,3,{PVC}\n1,3,{PVC.replace('3e9', '4e9')}\n",
            "missing density": f"{HEADER}\n0,3,pvc,3e9,5e7,,yield_euler,3\n",
            "unknown strength_method": f"{HEADER}\n0,3,pvc,3e9,5e7,1400,eurocode,3\n",
            "safety_factor must be at least 1": f"{HEADER}\n0,3,pvc,3e9,5e7,1400,yield_euler,0.5\n",
            "given together": "index,length,section_width,section_wall\n0,3,0.1,\n",
            "header needs": "index,length,colour\n0,3,red\n",
            "must be numbers": "index,length\n0,long\n",
        }
        for message, text in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                read_inventory(self.write(text))

    def test_write_rejects_sections_without_dimensions(self):
        edges = [dataclasses.replace(example_inventory(12, 3, 3)[0], section=CrossSection("custom", 1e-3, 1e-6))]
        with self.assertRaisesRegex(ValueError, "square hollow"):
            write_inventory(os.path.join(self.directory.name, "out.csv"), edges)


class StrengthMethodTests(unittest.TestCase):
    def test_yield_euler_uses_yield_then_euler_without_reduction(self):
        squash = PLASTIC.yield_strength * SECTION.area
        self.assertAlmostEqual(nominal_compressive_strength(1e-3, SECTION, PLASTIC), squash)
        length = 10.0
        euler = math.pi**2 * PLASTIC.elastic_modulus * SECTION.least_second_moment / length**2
        self.assertLess(euler, squash)
        self.assertAlmostEqual(nominal_compressive_strength(length, SECTION, PLASTIC) / euler, 1.0, places=12)

    def test_material_safety_factor_divides_strength(self):
        stronger = dataclasses.replace(PLASTIC, safety_factor=1.5)
        for force in (1e3, -1e3):
            with self.subTest(force=force):
                plastic_available, _, _ = check_member(2.0, force, SECTION, PLASTIC)
                stronger_available, _, _ = check_member(2.0, force, SECTION, stronger)
                self.assertAlmostEqual(stronger_available / plastic_available, 2.0)

    def test_invalid_materials_are_rejected(self):
        for change in ({"density": 0}, {"strength_method": "guess"}, {"safety_factor": 0.9}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(PLASTIC, **change)


class MixedMaterialTests(unittest.TestCase):
    def setUp(self):
        base = cached_design(12, 3, 3)
        self.materials = [PLASTIC if index % 3 == 0 else STEEL for index in range(len(base.members))]
        self.design = with_parts(base, self.materials)
        self.cases = basic_load_cases(self.design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)

    def test_inventory_parts_match_explicit_arguments(self):
        from_parts = run_load_tests(self.design, self.cases)
        explicit = run_load_tests(self.design, self.cases, sections=SECTION, materials=self.materials)
        for a, b in zip(from_parts.results, explicit.results, strict=True):
            np.testing.assert_array_equal(a.response.axial_forces, b.response.axial_forces)
        self.assertEqual(from_parts.materials, self.materials)
        for check in from_parts.results[1].member_checks:
            self.assertEqual(check.material_name, self.materials[check.member_index].name)
        self.assertIn("plastic x18 (yield/Euler, safety factor 3)", from_parts.summary())

    def test_self_weight_and_mass_use_each_members_density(self):
        expected = sum(
            material.density * SECTION.area * member.required_length
            for material, member in zip(self.materials, self.design.members, strict=True)
        )
        self.assertAlmostEqual(run_load_tests(self.design, self.cases[:1]).total_mass, expected, places=6)
        self.assertAlmostEqual(-self_weight_forces(self.design)[:, 2].sum(), expected * 9.80665, places=6)

    def test_softer_members_deflect_more(self):
        steel_only = run_load_tests(self.design, self.cases, materials=STEEL)
        mixed = run_load_tests(self.design, self.cases)
        self.assertGreater(mixed.max_vertical_deflection, steel_only.max_vertical_deflection)

    def test_missing_material_or_section_is_reported(self):
        design = cached_design(12, 3, 3)  # inventory of lengths only
        with self.assertRaisesRegex(ValueError, "have no material"):
            run_load_tests(design, self.cases, sections=SECTION)
        with self.assertRaisesRegex(ValueError, "have no section"):
            run_load_tests(design, self.cases, materials=STEEL)

    def test_sizing_checks_each_member_with_its_own_material(self):
        catalog = square_hollow_catalog()
        mixed = size_members(self.design, self.cases, catalog)
        steel = size_members(self.design, self.cases, catalog, materials=STEEL)
        self.assertTrue(mixed.report.passed)
        self.assertEqual(mixed.materials, self.materials)
        for group in mixed.groups:
            with self.subTest(group=group):
                self.assertGreaterEqual(mixed.group_sections[group].area, steel.group_sections[group].area)


if __name__ == "__main__":
    unittest.main()
