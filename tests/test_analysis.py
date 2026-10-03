"""tests/test_analysis.py

Tests of the structural analysis against hand solutions and conservation laws.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

import dataclasses
import math
import unittest

import numpy as np

from truss_bridge import (
    CrossSection,
    LoadCase,
    analyze_load_case,
    axle_load,
    basic_load_cases,
    deck_pressure_load,
    lateral_pressure_load,
    moving_axle_loads,
    run_load_tests,
)
from truss_bridge.analysis import (
    ALLOWABLE_STRENGTH_SAFETY_FACTOR,
    bearing_restraints,
    nominal_compressive_strength,
    nominal_tensile_strength,
    self_weight_forces,
    solve_truss,
)

from .support import STEEL, cached_design

ELASTIC_MODULUS = 200e9
AREA = 1e-3


def tripod(radius=1.0, height=2.0):
    """Three legs from a pinned base circle to an apex: hand-solvable by statics."""
    base = [(radius * math.cos(a), radius * math.sin(a), 0.0) for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
    coordinates = np.array(base + [(0.0, 0.0, height)])
    members = [(0, 3), (1, 3), (2, 3)]
    restraints = {0: (True, True, True), 1: (True, True, True), 2: (True, True, True)}
    return coordinates, members, restraints


class TrussSolverTests(unittest.TestCase):
    def test_single_bar_matches_hooke(self):
        coordinates = np.array([[0.0, 0, 0], [2.5, 0, 0]])
        forces = np.array([[0.0, 0, 0], [10e3, 0, 0]])
        restraints = {0: (True, True, True), 1: (False, True, True)}
        response = solve_truss(coordinates, [(0, 1)], np.array([ELASTIC_MODULUS * AREA]), restraints, forces)
        self.assertAlmostEqual(response.axial_forces[0], 10e3, places=6)
        self.assertAlmostEqual(response.displacements[1, 0], 10e3 * 2.5 / (ELASTIC_MODULUS * AREA), places=15)

    def test_tripod_leg_forces_and_deflection_match_hand_solution(self):
        radius, height, load = 1.0, 2.0, 30e3
        coordinates, members, restraints = tripod(radius, height)
        forces = np.zeros((4, 3))
        forces[3, 2] = -load
        stiffness = np.full(3, ELASTIC_MODULUS * AREA)
        response = solve_truss(coordinates, members, stiffness, restraints, forces)
        leg = math.hypot(radius, height)
        expected_force = -load * leg / (3 * height)  # vertical equilibrium, compression
        expected_drop = load * leg**3 / (3 * height**2 * ELASTIC_MODULUS * AREA)  # P*d = sum N^2 L / EA
        np.testing.assert_allclose(response.axial_forces, expected_force, rtol=1e-10)
        self.assertAlmostEqual(-response.displacements[3, 2], expected_drop, delta=expected_drop * 1e-10)
        np.testing.assert_allclose(sum(response.reactions.values()), [0, 0, load], atol=1e-6)

    def test_mechanism_is_rejected(self):
        coordinates, members, restraints = tripod()
        forces = np.zeros((4, 3))
        forces[3, 2] = -1e3
        with self.assertRaisesRegex(ValueError, "mechanism"):
            solve_truss(coordinates, members[:2], np.full(2, ELASTIC_MODULUS * AREA), restraints, forces)


class StrengthTests(unittest.TestCase):
    section = CrossSection.square_hollow(0.1, 0.005)

    def test_square_hollow_section_properties(self):
        self.assertAlmostEqual(self.section.area, 0.1**2 - 0.09**2, places=15)
        self.assertAlmostEqual(self.section.least_second_moment, (0.1**4 - 0.09**4) / 12, places=18)
        with self.assertRaises(ValueError):
            CrossSection.square_hollow(0.1, 0.06)

    def test_stocky_member_reaches_yield(self):
        squash = nominal_tensile_strength(self.section, STEEL)
        self.assertAlmostEqual(nominal_compressive_strength(1e-3, self.section, STEEL) / squash, 1.0, places=4)

    def test_slender_member_uses_reduced_euler_load(self):
        length = 20.0
        euler_stress = math.pi**2 * STEEL.elastic_modulus / (length / self.section.radius_of_gyration) ** 2
        expected = 0.877 * euler_stress * self.section.area
        self.assertAlmostEqual(nominal_compressive_strength(length, self.section, STEEL), expected, places=3)

    def test_column_curve_is_continuous_at_transition(self):
        limit = 4.71 * math.sqrt(STEEL.elastic_modulus / STEEL.yield_strength)
        r = self.section.radius_of_gyration
        below = nominal_compressive_strength(limit * r * (1 - 1e-9), self.section, STEEL)
        above = nominal_compressive_strength(limit * r * (1 + 1e-9), self.section, STEEL)
        self.assertAlmostEqual(below / above, 1.0, delta=1e-3)

    def test_compressive_strength_decreases_with_length(self):
        strengths = [nominal_compressive_strength(L, self.section, STEEL) for L in np.linspace(0.5, 15, 30)]
        self.assertTrue(all(a > b for a, b in zip(strengths, strengths[1:], strict=False)))


class BridgeLoadTests(unittest.TestCase):
    section = CrossSection.square_hollow(0.1, 0.005)

    def test_load_case_totals(self):
        for dimensions in [(12, 3, 3), (16, 2, 4)]:
            design = cached_design(*dimensions)
            span, width, height = dimensions
            with self.subTest(dimensions=dimensions):
                deck = deck_pressure_load(design, 4e3)
                np.testing.assert_allclose(sum(deck.nodal_forces.values()), [0, 0, -4e3 * span * width], rtol=1e-12)
                wind = lateral_pressure_load(design, 1e3)
                np.testing.assert_allclose(sum(wind.nodal_forces.values()), [0, 1e3 * span * height, 0], rtol=1e-12)
                for case in moving_axle_loads(design, 50e3):
                    np.testing.assert_allclose(sum(case.nodal_forces.values()), [0, 0, -50e3], rtol=1e-12)

    def test_axle_between_stations_uses_lever_rule(self):
        design = cached_design(12, 3, 3)
        case = axle_load(design, 60e3, 4.0)  # stations at 3 and 6: one third / two thirds
        by_station = {}
        for node, force in case.nodal_forces.items():
            by_station[design.nodes[node][0]] = by_station.get(design.nodes[node][0], 0) + force[2]
        self.assertAlmostEqual(by_station[3.0], -40e3)
        self.assertAlmostEqual(by_station[6.0], -20e3)
        with self.assertRaises(ValueError):
            axle_load(design, 1.0, 12.5)

    def test_self_weight_total(self):
        design = cached_design(12, 3, 3)
        total_length = sum(m.required_length for m in design.members)
        expected = STEEL.density * self.section.area * total_length * 9.80665
        self.assertAlmostEqual(-self_weight_forces(design, self.section, STEEL)[:, 2].sum(), expected, places=6)

    def test_bearings_restrain_exactly_eight_degrees_of_freedom(self):
        restraints = bearing_restraints(cached_design(12, 3, 3))
        self.assertEqual(len(restraints), 4)
        self.assertEqual(sum(sum(fixed) for fixed in restraints.values()), 8)

    def test_equilibrium_and_energy_balance(self):
        for dimensions in [(12, 3, 3), (16, 2, 4)]:
            design = cached_design(*dimensions)
            for case in basic_load_cases(design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3):
                with self.subTest(dimensions=dimensions, case=case.name):
                    result = analyze_load_case(design, case, sections=self.section, materials=STEEL)
                    scale = np.linalg.norm(result.applied_force)
                    self.assertLess(result.equilibrium_error, 1e-8 * scale)
                    # Clapeyron: external work equals strain energy (supports do no work).
                    applied = np.zeros((len(design.nodes), 3))
                    for node, force in case.nodal_forces.items():
                        applied[node] += force
                    applied += self_weight_forces(design, self.section, STEEL)
                    external_work = 0.5 * np.sum(applied * result.response.displacements)
                    axial_stiffness = STEEL.elastic_modulus * self.section.area
                    strain_energy = sum(
                        force**2 * member.required_length / (2 * axial_stiffness)
                        for force, member in zip(result.response.axial_forces, design.members, strict=True)
                    )
                    self.assertAlmostEqual(external_work / strain_energy, 1.0, places=9)

    def test_stiffer_material_halves_deflection_without_changing_forces(self):
        design = cached_design(12, 3, 3)
        case = LoadCase("deck").combined_with(deck_pressure_load(design, 4e3))
        stiff = dataclasses.replace(STEEL, elastic_modulus=2 * STEEL.elastic_modulus)
        base = analyze_load_case(design, case, sections=self.section, materials=STEEL)
        doubled = analyze_load_case(design, case, sections=self.section, materials=stiff)
        np.testing.assert_allclose(doubled.response.axial_forces, base.response.axial_forces, rtol=1e-9, atol=1e-6)
        self.assertAlmostEqual(doubled.max_vertical_deflection / base.max_vertical_deflection, 0.5, places=9)

    def test_response_is_linear_in_load(self):
        design = cached_design(16, 2, 4)
        single = LoadCase("deck", deck_pressure_load(design, 2e3).nodal_forces, include_self_weight=False)
        double = LoadCase("deck", deck_pressure_load(design, 4e3).nodal_forces, include_self_weight=False)
        a = analyze_load_case(design, single, sections=self.section, materials=STEEL).response.axial_forces
        b = analyze_load_case(design, double, sections=self.section, materials=STEEL).response.axial_forces
        np.testing.assert_allclose(b, 2 * a, rtol=1e-9, atol=1e-6)

    def test_member_checks_use_the_right_limit_state(self):
        design = cached_design(12, 3, 3)
        case = LoadCase("deck").combined_with(deck_pressure_load(design, 4e3))
        result = analyze_load_case(design, case, sections=self.section, materials=STEEL)
        tension_available = nominal_tensile_strength(self.section, STEEL) / ALLOWABLE_STRENGTH_SAFETY_FACTOR
        for check in result.member_checks:
            member = design.members[check.member_index]
            if check.axial_force >= 0:
                self.assertEqual(check.limit_state, "tension yielding")
                self.assertAlmostEqual(check.available_strength, tension_available)
            else:
                self.assertEqual(check.limit_state, "flexural buckling")
                expected = nominal_compressive_strength(member.required_length, self.section, STEEL)
                self.assertAlmostEqual(check.available_strength, expected / ALLOWABLE_STRENGTH_SAFETY_FACTOR)
            self.assertAlmostEqual(check.utilization, abs(check.axial_force) / check.available_strength)

    def test_report_pass_and_fail(self):
        design = cached_design(12, 3, 3)
        cases = basic_load_cases(design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)
        light = run_load_tests(design, cases, sections=self.section, materials=STEEL)
        self.assertTrue(light.passed)
        heavy_cases = basic_load_cases(design, deck_pressure=4e5, axle_force=50e3, lateral_pressure=1e3)
        heavy = run_load_tests(design, heavy_cases, sections=self.section, materials=STEEL)
        self.assertFalse(heavy.passed)
        self.assertIn("FAIL", heavy.summary())
        strict = run_load_tests(design, cases, sections=self.section, materials=STEEL, deflection_limit_ratio=1e6)
        self.assertFalse(strict.passed)


if __name__ == "__main__":
    unittest.main()
