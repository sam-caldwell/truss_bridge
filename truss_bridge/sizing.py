"""truss_bridge/sizing.py

Automatic member sizing: the lightest catalog section per member group that passes every load test.

Sizing proceeds in three phases, each followed by a full re-analysis:

1. **Strength.** Every group starts at the lightest catalog section. Each pass analyzes all
   load cases and moves every group up to the lightest section that carries its worst
   force in every case. Sections only ever move up, so the loop terminates. It converges in
   a few passes because the bridge is internally statically determinate (minimally rigid):
   member forces depend on member stiffness only through the two redundant bearing
   reactions.
2. **Deflection** (when a limit is given). By the unit-load method the governing
   deflection is ``sum over groups of c_g / A_g`` with ``c_g = sum(N n L / E)``, where ``N``
   are member forces in the governing case and ``n`` those from a unit load at the most
   deflected node. Holding forces fixed, the least-mass areas meeting the limit satisfy the
   optimality criterion ``A_g = max(A_strength, sqrt(c_g / (lambda w_g)))`` with
   ``w_g = sum(rho L)`` the group's mass per unit area (Haftka & Gurdal, 1992), with
   ``lambda`` found by bisection. Each group is moved up to the lightest
   catalog section at least that large, strength is re-checked, and the process repeats
   with updated forces (which include the extra self-weight). If an update changes nothing,
   the single group whose next larger section removes the most deflection per kilogram is
   moved up one step instead, so every iteration makes progress.
3. **Downsizing.** Lighter sections are screened with the current forces and the linear
   deflection estimate. First, the screened downsizes that save the most mass per unit of
   added deflection are applied together and verified by one re-analysis. Then each
   group, heaviest first, is tried at every lighter section that could plausibly pass, and
   a change is kept only if a full re-analysis of every load case still passes. Screens use
   the analysis from the start of each sweep; verification always uses a fresh analysis.
   Phase 3 repeats until a whole sweep changes nothing, so no single group can be made lighter on its own. The
   result is a local optimum, not a proven global minimum.

Each member keeps its own material (by default, that of its inventory edge), so a group may
mix materials; it still gets one section size.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .analysis import (
    LoadCase,
    LoadTestReport,
    MaterialsArgument,
    bearing_restraints,
    check_member,
    member_materials,
    run_load_tests,
    solve_truss,
)
from .model import BridgeDesign, CrossSection, Material

EXAMPLE_SHS_WIDTHS_MM = (40, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 200, 250, 300)
EXAMPLE_SHS_WALLS_MM = (3, 4, 5, 6, 8, 10, 12.5, 16)
"""Illustrative square hollow section dimensions; replace with the sizes actually available."""

RECOMMENDED_MAX_COMPRESSION_SLENDERNESS = 200.0
"""AISC 360-16 Section E2 user note: compression members should preferably not exceed L/r = 200."""

GROUPINGS = ("role", "member")
_SCREEN_MARGIN = 1.10
"""Downsizing skips sections whose estimated utilization exceeds this (forces barely move)."""
_STALL_STEPS = 5
"""Deflection sizing gives up after this many steps without a 0.1% improvement."""


class SizingError(Exception):
    """Raised when no catalog section can satisfy a group's requirements."""


def square_hollow_catalog(
    widths_mm: tuple[float, ...] = EXAMPLE_SHS_WIDTHS_MM, walls_mm: tuple[float, ...] = EXAMPLE_SHS_WALLS_MM
) -> list[CrossSection]:
    """Idealized square hollow sections (sharp corners, nominal wall) for every width and wall.

    Combinations with a wall thicker than a quarter of the width are skipped.

    Args:
        widths_mm: Outside widths, in mm.
        walls_mm: Wall thicknesses, in mm.

    Returns:
        Sections sorted from lightest to heaviest (ties: larger radius of gyration first).
    """
    sections = [
        CrossSection.square_hollow(width / 1e3, wall / 1e3)
        for width in widths_mm
        for wall in walls_mm
        if wall <= width / 4
    ]
    return sorted(sections, key=lambda section: (section.area, -section.radius_of_gyration))


@dataclass
class SizingResult:
    """Outcome of :func:`size_members`.

    Attributes:
        design: The bridge sized.
        materials: Material of each member, in ``design.members`` order.
        groups: Member indices in each group, keyed by group name.
        group_sections: Section chosen for each group.
        report: Load test of the sized bridge.
        strength_passes: Number of strength re-analysis passes in phase 1.
        deflection_steps: Number of single-group upsizing steps taken for deflection.
        downsizing_changes: Number of groups made lighter in phase 3.
    """

    design: BridgeDesign
    materials: list[Material]
    groups: dict[str, list[int]]
    group_sections: dict[str, CrossSection]
    report: LoadTestReport
    strength_passes: int
    deflection_steps: int
    downsizing_changes: int

    @property
    def member_sections(self) -> list[CrossSection]:
        """Section of every member, in ``design.members`` order."""
        return self.report.sections

    @property
    def total_mass(self) -> float:
        """Total member mass, in kg."""
        return self.report.total_mass

    def group_utilization(self, group: str) -> float:
        """Highest utilization of any member of ``group`` over all load cases."""
        members = set(self.groups[group])
        return max(
            check.utilization
            for result in self.report.results
            for check in result.member_checks
            if check.member_index in members
        )

    def summary(self, max_rows: int = 40) -> str:
        """Table of groups with their section, governing utilization and mass."""
        lines = [
            f"Member sizing: {len(self.groups)} groups, total member mass {self.total_mass:,.0f} kg, "
            f"{'PASS' if self.report.passed else 'FAIL'} "
            f"({self.strength_passes} strength passes, {self.deflection_steps} deflection steps, "
            f"{self.downsizing_changes} downsizing changes)",
            f"  {'group':<22} {'members':>7}  {'section':<24} {'max util.':>9} {'mass (kg)':>10}",
        ]
        rows = sorted(self.groups, key=lambda group: -self._group_mass(group))
        for group in rows[:max_rows]:
            lines.append(
                f"  {group:<22} {len(self.groups[group]):>7}  {self.group_sections[group].name:<24} "
                f"{self.group_utilization(group):9.3f} {self._group_mass(group):10.1f}"
            )
        if len(rows) > max_rows:
            lines.append(f"  ... {len(rows) - max_rows} more groups")
        if self.report.deflection_limit is not None:
            lines.append(
                f"  deflection {self.report.max_vertical_deflection * 1e3:.2f} mm "
                f"(limit {self.report.deflection_limit * 1e3:.2f} mm)"
            )
        return "\n".join(lines)

    def _group_mass(self, group: str) -> float:
        return self.group_sections[group].area * _mass_per_area(self.design, self.groups[group], self.materials)


def group_members(design: BridgeDesign, group_by: str) -> dict[str, list[int]]:
    """Member indices grouped by role (``"role"``) or one group per member (``"member"``).

    Raises:
        ValueError: If ``group_by`` is not recognized.
    """
    if group_by == "role":
        groups: dict[str, list[int]] = defaultdict(list)
        for index, member in enumerate(design.members):
            groups[member.role].append(index)
        return dict(groups)
    if group_by == "member":
        return {f"#{index} {member.role}": [index] for index, member in enumerate(design.members)}
    raise ValueError(f"unknown grouping {group_by!r}; expected one of {GROUPINGS}")


def _mass_per_area(design: BridgeDesign, members: list[int], materials: list[Material]) -> float:
    """Mass a group gains per unit increase in its section area, ``sum(rho L)``, in kg/m^2."""
    return sum(materials[index].density * design.members[index].required_length for index in members)


def _worst_forces(report: LoadTestReport) -> tuple[np.ndarray, np.ndarray]:
    """Largest tension and largest compression (as positive numbers) of each member over all cases."""
    forces = np.array([result.response.axial_forces for result in report.results])
    return np.maximum(forces.max(axis=0), 0.0), np.maximum(-forces.min(axis=0), 0.0)


def _section_carries(
    section: CrossSection,
    members: list[int],
    design: BridgeDesign,
    tension: np.ndarray,
    compression: np.ndarray,
    materials: list[Material],
    max_slenderness: float | None,
    margin: float = 1.0,
) -> bool:
    """Whether ``section`` carries the given worst forces of every member in a group."""
    for index in members:
        length = design.members[index].required_length
        worst = [tension[index]] + ([-compression[index]] if compression[index] > 0 else [])
        for force in worst:
            _, utilization, _ = check_member(length, force, section, materials[index], max_slenderness)
            if utilization > margin:
                return False
    return True


def size_members(
    design: BridgeDesign,
    load_cases: list[LoadCase],
    catalog: list[CrossSection] | None = None,
    *,
    materials: MaterialsArgument = None,
    group_by: str = "role",
    deflection_limit_ratio: float | None = None,
    max_compression_slenderness: float | None = RECOMMENDED_MAX_COMPRESSION_SLENDERNESS,
    max_passes: int = 100,
) -> SizingResult:
    """Choose the lightest catalog section for each member group that passes every load case.

    Sections of the inventory edges are ignored: the result says what section each group
    would need, given its members' materials.

    Args:
        design: The bridge; coordinates in m.
        load_cases: Cases every member must carry.
        catalog: Available sections; :func:`square_hollow_catalog` by default.
        materials: One material for all members, one per member, or None (the default) for
            the material of each member's inventory edge. Each member is checked by its
            material's strength method and safety factor.
        group_by: ``"role"`` gives all members of a role the same section; ``"member"`` sizes
            each member individually.
        deflection_limit_ratio: If given, deflection must not exceed span / ratio.
        max_compression_slenderness: Limit on ``L/r`` for compression members, or None.
        max_passes: Safety limit on re-analysis passes in each phase.

    Returns:
        The sized design and its load-test report.

    Raises:
        SizingError: If some group cannot be made strong enough, or the deflection limit
            cannot be met, with the catalog provided.
        ValueError: If ``group_by`` is not recognized, the catalog is empty, or materials
            are missing.
    """
    catalog = sorted(
        catalog if catalog is not None else square_hollow_catalog(),
        key=lambda section: (section.area, -section.radius_of_gyration),
    )
    if not catalog:
        raise ValueError("catalog is empty")
    per_member = member_materials(design, materials)
    groups = group_members(design, group_by)
    mass_per_area = {group: _mass_per_area(design, members, per_member) for group, members in groups.items()}
    choice = dict.fromkeys(groups, 0)  # catalog index per group

    def analyze() -> LoadTestReport:
        sections = [catalog[0]] * len(design.members)
        for group, members in groups.items():
            for index in members:
                sections[index] = catalog[choice[group]]
        return run_load_tests(
            design,
            load_cases,
            sections=sections,
            materials=per_member,
            deflection_limit_ratio=deflection_limit_ratio,
            max_compression_slenderness=max_compression_slenderness,
        )

    def size_for_strength(report: LoadTestReport) -> tuple[LoadTestReport, int]:
        for passes in range(1, max_passes + 1):
            tension, compression = _worst_forces(report)
            changed = False
            for group, members in groups.items():
                required = next(
                    (
                        position
                        for position in range(choice[group], len(catalog))
                        if _section_carries(
                            catalog[position],
                            members,
                            design,
                            tension,
                            compression,
                            per_member,
                            max_compression_slenderness,
                        )
                    ),
                    None,
                )
                if required is None:
                    worst = max(members, key=lambda index: max(tension[index], compression[index]))
                    strongest = catalog[-1]
                    available, _, limit_state = check_member(
                        design.members[worst].required_length,
                        -compression[worst] if compression[worst] >= tension[worst] else tension[worst],
                        strongest,
                        per_member[worst],
                        max_compression_slenderness,
                    )
                    raise SizingError(
                        f"no catalog section is strong enough for group '{group}': member #{worst} carries "
                        f"{tension[worst] / 1e3:,.0f} kN tension / {compression[worst] / 1e3:,.0f} kN compression, "
                        f"but the largest section ({strongest.name}) provides {available / 1e3:,.0f} kN "
                        f"({limit_state}). Forces this large usually mean an inefficient load path in the design"
                    )
                if required != choice[group]:
                    choice[group] = required
                    changed = True
            report = analyze()
            if not changed:
                return report, passes
        raise SizingError("strength sizing did not converge; raise max_passes")

    report, strength_passes = size_for_strength(analyze())

    deflection_steps = 0
    best_deflection, steps_without_progress = math.inf, 0
    while report.deflection_limit is not None and report.max_vertical_deflection > report.deflection_limit:
        if report.max_vertical_deflection < best_deflection * 0.999:
            best_deflection, steps_without_progress = report.max_vertical_deflection, 0
        else:
            steps_without_progress += 1
        if steps_without_progress >= _STALL_STEPS:
            raise SizingError(
                f"the deflection limit cannot be met: deflection stalled at {best_deflection * 1e3:.2f} mm. "
                "Self-weight deflection does not shrink as sections grow, since weight and stiffness both "
                "scale with area; a stiffer layout or a lighter material is needed"
            )
        if deflection_steps >= max_passes:
            raise SizingError("deflection sizing did not converge; raise max_passes")
        sensitivities = _deflection_sensitivities(design, groups, per_member, report)
        targets = _optimality_criteria_areas(
            mass_per_area,
            {group: catalog[choice[group]].area for group in groups},
            sensitivities,
            report.deflection_limit,
        )
        changed = False
        for group, target in targets.items():
            fitting = next(
                (index for index in range(choice[group], len(catalog)) if catalog[index].area >= target * (1 - 1e-12)),
                len(catalog) - 1,
            )
            if catalog[fitting].area > catalog[choice[group]].area:
                choice[group], changed = fitting, True
        if not changed:
            group = _best_group_for_deflection(mass_per_area, choice, catalog, sensitivities)
            if group is None:
                raise SizingError("the deflection limit cannot be met with the sections in the catalog")
            choice[group] = _next_heavier(catalog, choice[group])
        deflection_steps += 1
        report, _ = size_for_strength(analyze())

    downsizing_changes = 0
    improved = True
    while improved:
        improved = False
        tension, compression = _worst_forces(report)
        sensitivities = (
            _deflection_sensitivities(design, groups, per_member, report) if report.deflection_limit is not None else {}
        )
        screen = _DownsizeScreen(
            design,
            groups,
            catalog,
            per_member,
            report,
            sensitivities,
            tension,
            compression,
            max_compression_slenderness,
        )

        # Batch step: apply the best downsizes together within the deflection budget; verify once.
        proposals = []
        for group in groups:
            candidates = screen.lighter_positions(group, choice[group], margin=1.0)
            if candidates:
                position = candidates[0]
                saved = (catalog[choice[group]].area - catalog[position].area) * mass_per_area[group]
                added = (
                    sensitivities[group] * (1 / catalog[position].area - 1 / catalog[choice[group]].area)
                    if report.deflection_limit is not None
                    else 0.0
                )
                proposals.append((saved / max(added, 1e-15), group, position, added))
        if len(proposals) > 1:
            previous = dict(choice)
            budget = (report.deflection_limit - report.max_vertical_deflection) if report.deflection_limit else math.inf
            for _, group, position, added in sorted(proposals, reverse=True):
                if added <= budget:
                    choice[group] = position
                    budget -= max(added, 0.0)
            changed = [group for group in groups if choice[group] != previous[group]]
            if len(changed) > 1:
                trial = analyze()
                if trial.passed:
                    report, improved = trial, True
                    downsizing_changes += len(changed)
                    continue
            choice.clear()
            choice.update(previous)

        # Sequential step: heaviest group first, one verified change at a time.
        by_mass = sorted(groups, key=lambda group: -catalog[choice[group]].area * mass_per_area[group])
        for group in by_mass:
            original = choice[group]
            for position in screen.lighter_positions(group, original, margin=_SCREEN_MARGIN):
                choice[group] = position
                trial = analyze()
                if trial.passed:
                    report, improved = trial, True
                    downsizing_changes += 1
                    break
                choice[group] = original

    if not report.passed:
        raise SizingError("sizing finished without a passing design")  # defensive; phases 1-2 ensure a pass
    return SizingResult(
        design,
        per_member,
        groups,
        {group: catalog[choice[group]] for group in groups},
        report,
        strength_passes,
        deflection_steps,
        downsizing_changes,
    )


@dataclass
class _DownsizeScreen:
    """Cheap estimates of whether a group could take a lighter section, from the current analysis."""

    design: BridgeDesign
    groups: dict[str, list[int]]
    catalog: list[CrossSection]
    materials: list[Material]
    report: LoadTestReport
    sensitivities: dict[str, float]
    tension: np.ndarray
    compression: np.ndarray
    max_compression_slenderness: float | None

    def lighter_positions(self, group: str, current: int, margin: float) -> list[int]:
        """Lighter catalog positions passing the strength and deflection screens, lightest first.

        Args:
            group: Group to screen.
            current: The group's current catalog position.
            margin: Largest estimated utilization accepted.
        """
        found = []
        current_area = self.catalog[current].area
        for position in range(current):
            area = self.catalog[position].area
            if area >= current_area:
                continue
            limit = self.report.deflection_limit
            if limit is not None:
                change = self.sensitivities[group] * (1 / area - 1 / current_area)
                if self.report.max_vertical_deflection + change > limit:
                    continue
            if _section_carries(
                self.catalog[position],
                self.groups[group],
                self.design,
                self.tension,
                self.compression,
                self.materials,
                self.max_compression_slenderness,
                margin=margin,
            ):
                found.append(position)
        return found


def _deflection_sensitivities(
    design: BridgeDesign, groups: dict[str, list[int]], materials: list[Material], report: LoadTestReport
) -> dict[str, float]:
    """Each group's coefficient ``c_g`` in the governing deflection ``sum(c_g / A_g)``.

    By the unit-load method the downward deflection of node ``k`` is ``sum(N n L / (E A))``,
    where ``N`` are member forces in the governing case and ``n`` those from a unit
    downward load at ``k``; ``c_g = sum over the group of N n L / E``, with each member's
    own ``E``.
    """
    governing = max(report.results, key=lambda result: result.max_vertical_deflection)
    node = int(np.argmin(governing.response.displacements[:, 2]))
    coordinates = np.asarray(design.nodes, float)
    endpoints = [(member.start_node, member.end_node) for member in design.members]
    moduli = [material.elastic_modulus for material in materials]
    stiffness = np.array([modulus * section.area for modulus, section in zip(moduli, report.sections, strict=True)])
    unit_load = np.zeros((len(coordinates), 3))
    unit_load[node, 2] = -1.0
    virtual = solve_truss(coordinates, endpoints, stiffness, bearing_restraints(design), unit_load).axial_forces
    real = governing.response.axial_forces
    return {
        group: sum(real[i] * virtual[i] * design.members[i].required_length / moduli[i] for i in members)
        for group, members in groups.items()
    }


def _optimality_criteria_areas(
    mass_per_area: dict[str, float],
    current_areas: dict[str, float],
    sensitivities: dict[str, float],
    deflection_limit: float,
) -> dict[str, float]:
    """Least-mass group areas meeting the deflection limit, holding member forces fixed.

    Minimizes ``sum(w_g A_g)``, with ``w_g`` the group's mass per unit area, subject to
    ``sum(c_g / A_g) <= limit`` and ``A_g >= current``. Groups with ``c_g <= 0`` stay at
    their current area, since enlarging them would not reduce the deflection. For the rest
    the Lagrange conditions give ``A_g = max(current_g, sqrt(c_g / (lambda w_g)))``;
    ``lambda`` is found by bisection on a logarithmic scale.

    Returns:
        Target area for each group whose area should grow; empty if the limit is already
        met at current areas, or every positive group would need to grow without bound.
    """
    fixed = sum(c / current_areas[g] for g, c in sensitivities.items() if c <= 0)
    positive = {g: c for g, c in sensitivities.items() if c > 0}
    budget = deflection_limit - fixed
    if not positive or budget <= 0:
        return {}

    def areas(multiplier: float) -> dict[str, float]:
        return {g: max(current_areas[g], math.sqrt(c / (multiplier * mass_per_area[g]))) for g, c in positive.items()}

    def deflection(multiplier: float) -> float:
        return sum(c / area for (g, c), area in zip(positive.items(), areas(multiplier).values(), strict=True))

    low, high = 1e-30, 1e30  # deflection(low) ~ 0, deflection(high) = value at current areas
    if deflection(high) <= budget:
        return {}
    for _ in range(200):
        middle = math.sqrt(low * high)
        if deflection(middle) > budget:
            high = middle
        else:
            low = middle
    return {g: area for g, area in areas(low).items() if area > current_areas[g]}


def _best_group_for_deflection(
    mass_per_area: dict[str, float],
    choice: dict[str, int],
    catalog: list[CrossSection],
    sensitivities: dict[str, float],
) -> str | None:
    """Group whose next larger-area section removes the most deflection per added kilogram.

    Enlarging a group from area ``A`` to ``A'`` reduces the deflection by ``c_g (1/A - 1/A')``.
    """
    best_group, best_rate = None, 0.0
    for group, weight in mass_per_area.items():
        current = choice[group]
        heavier = _next_heavier(catalog, current)
        if heavier is None:
            continue
        area_now, area_next = catalog[current].area, catalog[heavier].area
        reduction = sensitivities[group] * (1 / area_now - 1 / area_next)
        added_mass = weight * (area_next - area_now)
        rate = reduction / added_mass
        if reduction > 0 and rate > best_rate and math.isfinite(rate):
            best_group, best_rate = group, rate
    return best_group


def _next_heavier(catalog: list[CrossSection], position: int) -> int | None:
    """Index of the first section with strictly larger area than ``catalog[position]``.

    The catalog is sorted by area, then by descending radius of gyration, so among
    sections of equal area the one returned is the most resistant to buckling.
    """
    area = catalog[position].area
    larger = (index for index in range(position + 1, len(catalog)) if catalog[index].area > area * (1 + 1e-12))
    return next(larger, None)
