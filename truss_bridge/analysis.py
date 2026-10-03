"""truss_bridge/analysis.py

Structural load testing of a bridge design by the direct stiffness method.

The bridge is analyzed as a linear-elastic, pin-jointed space truss: members carry axial
force only and loads are applied at nodes. Analysis uses SI units: node coordinates in
meters, forces in newtons, stresses and moduli in pascals.

Each member is checked with the strength method of its own material, and nominal
strengths are divided by that material's safety factor.

``"aisc_360_16"`` (structural steel) follows ANSI/AISC 360-16 for axially loaded members,
allowable strength design (AISC, 2016):

* tension yielding (Section D2): ``Pn = Fy * Ag``; AISC's ``Omega_t`` is 1.67;
* compression, flexural buckling (Section E3) with effective length factor K = 1 for
  pin-ended members: ``Fe = pi^2 E / (L/r)^2``; ``Fcr = 0.658^(Fy/Fe) * Fy`` when
  ``L/r <= 4.71 sqrt(E/Fy)``, otherwise ``Fcr = 0.877 Fe``; ``Pn = Fcr * Ag``; AISC's
  ``Omega_c`` is 1.67.

``"yield_euler"`` (other materials, such as plastics and fiber-reinforced plastics, where
the steel column curve does not apply): ``Pn = Fy * Ag`` in tension and
``Pn = min(Fy, Fe) * Ag`` in compression. The safety factor must cover what this omits:
imperfection sensitivity near ``Fe = Fy``, creep under sustained load, and temperature
and moisture effects on stiffness and strength.

Local buckling of slender walls (Section E7) is not computed. Instead, a compression
member whose section has slender walls under Table B4.1a (rectangular HSS walls,
``b/t > 1.40 sqrt(E/Fy)``, with ``b`` the outside width minus three wall thicknesses when
the corner radius is unknown, Section B4.1b) is reported as failing. The same screen is
applied for both methods; for other materials it is a conservative stand-in for elastic
plate buckling, whose limit is about ``1.9 sqrt(E/Fy)``. An optional limit on compression
slenderness ``L/r`` can also be applied; AISC recommends 200.

Not covered: tension rupture at connections, connection and bearing design, fatigue,
dynamic or vibration response, and load combinations or factors from a bridge design
code. This is a screening check, not a substitute for an engineer's design.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .model import AISC_360_16, YIELD_EULER, BridgeDesign, CrossSection, Material

STANDARD_GRAVITY = 9.80665
"""Standard acceleration of gravity in m/s^2 (exact by definition)."""

ALLOWABLE_STRENGTH_SAFETY_FACTOR = 1.67
"""AISC 360-16 ASD safety factor for tension yielding and compression (Omega_t, Omega_c)."""

STRENGTH_METHOD_LABELS = {AISC_360_16: "AISC 360-16", YIELD_EULER: "yield/Euler"}

SLENDER_WALL = "slender wall (local buckling not covered)"
SLENDERNESS_LIMIT = "compression slenderness over limit"


def slender_wall_limit(material: Material) -> float:
    """AISC 360-16 Table B4.1a limit ``1.40 sqrt(E/Fy)`` for rectangular HSS walls in compression."""
    return 1.40 * math.sqrt(material.elastic_modulus / material.yield_strength)


def has_slender_walls(section: CrossSection, material: Material) -> bool:
    """Whether the section's walls are slender for compression (local buckling would govern)."""
    return section.wall_slenderness is not None and section.wall_slenderness > slender_wall_limit(material)


def nominal_compressive_strength(length: float, section: CrossSection, material: Material) -> float:
    """Flexural-buckling strength Pn of a pin-ended member, in N, by the material's method.

    ``"aisc_360_16"`` uses the AISC 360-16 Section E3 column curve; ``"yield_euler"`` uses
    ``min(Fy, Fe)``.

    Args:
        length: Unbraced (node-to-node) length, in m; effective length factor K = 1.
        section: Member cross-section.
        material: Member material.

    Returns:
        Nominal compressive strength ``Fcr * Ag``.
    """
    slenderness = length / section.radius_of_gyration
    elastic_buckling_stress = math.pi**2 * material.elastic_modulus / slenderness**2
    yield_strength = material.yield_strength
    if material.strength_method == YIELD_EULER:
        critical_stress = min(yield_strength, elastic_buckling_stress)
    elif slenderness <= 4.71 * math.sqrt(material.elastic_modulus / yield_strength):
        critical_stress = 0.658 ** (yield_strength / elastic_buckling_stress) * yield_strength
    else:
        critical_stress = 0.877 * elastic_buckling_stress
    return critical_stress * section.area


def nominal_tensile_strength(section: CrossSection, material: Material) -> float:
    """Tension-yielding strength ``Pn = Fy * Ag``, in N (AISC 360-16 Section D2; same for both methods)."""
    return material.yield_strength * section.area


# --------------------------------------------------------------------------- generic truss solver


@dataclass
class TrussResponse:
    """Linear-elastic response of a space truss to one set of nodal forces.

    Attributes:
        displacements: Node displacements, shape ``(node_count, 3)``, in m.
        axial_forces: Member axial forces, tension positive, in N.
        reactions: Support reaction vector for each supported node, in N.
    """

    displacements: np.ndarray
    axial_forces: np.ndarray
    reactions: dict[int, np.ndarray]


def solve_truss_cases(
    node_coordinates: np.ndarray,
    member_endpoints: list[tuple[int, int]],
    axial_stiffness: np.ndarray,
    restraints: dict[int, tuple[bool, bool, bool]],
    force_sets: list[np.ndarray],
) -> list[TrussResponse]:
    """Direct stiffness analysis of a pin-jointed space truss under several sets of nodal forces.

    Each member contributes ``(EA / L) * [[n n^T, -n n^T], [-n n^T, n n^T]]`` to the global
    stiffness matrix, where ``n`` is its unit direction. Restrained degrees of freedom are
    removed, the remaining matrix is factorized once (sparse LU), and every force set is
    solved with that factorization.

    Args:
        node_coordinates: Shape ``(node_count, 3)``, in m.
        member_endpoints: ``(start_node, end_node)`` for every member.
        axial_stiffness: ``E * A`` for every member, in N.
        restraints: For each supported node, which of x, y, z are held fixed.
        force_sets: Applied forces for each case, each of shape ``(node_count, 3)``, in N.

    Returns:
        One response per force set, in order.

    Raises:
        ValueError: If the supported structure is a mechanism (singular stiffness).
    """
    node_count = len(node_coordinates)
    dof_count = 3 * node_count
    starts = np.array([start for start, _ in member_endpoints])
    ends = np.array([end for _, end in member_endpoints])
    vectors = node_coordinates[ends] - node_coordinates[starts]
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / lengths[:, None]
    member_stiffness = np.asarray(axial_stiffness, float) / lengths
    blocks = member_stiffness[:, None, None] * directions[:, :, None] * directions[:, None, :]
    elements = np.block([[blocks, -blocks], [-blocks, blocks]])  # (member_count, 6, 6)
    dofs = np.concatenate([3 * starts[:, None] + np.arange(3), 3 * ends[:, None] + np.arange(3)], axis=1)
    rows = np.repeat(dofs, 6, axis=1).ravel()
    columns = np.tile(dofs, (1, 6)).ravel()
    stiffness = coo_matrix((elements.ravel(), (rows, columns)), shape=(dof_count, dof_count)).tocsr()

    restrained = sorted(3 * node + axis for node, fixed in restraints.items() for axis in range(3) if fixed[axis])
    free = np.setdiff1d(np.arange(dof_count), restrained)
    free_stiffness = stiffness[free][:, free].tocsc()
    try:
        factorization = splu(free_stiffness)
    except RuntimeError as error:  # exactly singular
        raise ValueError("the supported structure is a mechanism (singular stiffness matrix)") from error

    responses = []
    for nodal_forces in force_sets:
        forces = np.asarray(nodal_forces, float).reshape(-1)
        with np.errstate(all="ignore"):
            solution = factorization.solve(forces[free])
        residual = free_stiffness @ solution - forces[free]
        scale = max(1.0, float(np.abs(forces).max(initial=0.0)))
        if not np.all(np.isfinite(solution)) or np.abs(residual).max(initial=0.0) > 1e-6 * scale:
            raise ValueError("the supported structure is a mechanism (singular stiffness matrix)")
        displacement_vector = np.zeros(dof_count)
        displacement_vector[free] = solution
        displacements = displacement_vector.reshape(-1, 3)
        axial_forces = member_stiffness * np.einsum("ij,ij->i", directions, displacements[ends] - displacements[starts])
        reaction_vector = stiffness @ displacement_vector - forces
        reactions = {node: reaction_vector[3 * node : 3 * node + 3].copy() for node in restraints}
        responses.append(TrussResponse(displacements, axial_forces, reactions))
    return responses


def solve_truss(
    node_coordinates: np.ndarray,
    member_endpoints: list[tuple[int, int]],
    axial_stiffness: np.ndarray,
    restraints: dict[int, tuple[bool, bool, bool]],
    nodal_forces: np.ndarray,
) -> TrussResponse:
    """Direct stiffness analysis under one set of nodal forces; see :func:`solve_truss_cases`.

    Raises:
        ValueError: If the supported structure is a mechanism (singular stiffness).
    """
    return solve_truss_cases(node_coordinates, member_endpoints, axial_stiffness, restraints, [nodal_forces])[0]


# --------------------------------------------------------------------------- supports and loads


def bearing_restraints(design: BridgeDesign) -> dict[int, tuple[bool, bool, bool]]:
    """Bearings under the four deck corners: pinned at x = 0, sliding along x at the far end.

    Restraints (x, y, z): near-left corner (fixed, fixed, fixed); near-right (fixed, free,
    fixed); far-left (free, fixed, fixed); far-right (free, free, fixed). This removes all
    six rigid-body motions and lets the span expand freely along x.

    Raises:
        ValueError: If a deck corner node is missing.
    """
    coordinates = np.asarray(design.nodes, float)
    x_min, y_min, z_min = coordinates.min(axis=0)
    x_max, y_max, _ = coordinates.max(axis=0)

    def node_at(x: float, y: float) -> int:
        matches = np.flatnonzero(np.all(np.isclose(coordinates, [x, y, z_min]), axis=1))
        if len(matches) != 1:
            raise ValueError(f"no unique deck corner node at x={x:g}, y={y:g}")
        return int(matches[0])

    return {
        node_at(x_min, y_min): (True, True, True),
        node_at(x_min, y_max): (True, False, True),
        node_at(x_max, y_min): (False, True, True),
        node_at(x_max, y_max): (False, False, True),
    }


@dataclass
class LoadCase:
    """A set of nodal forces applied together.

    Attributes:
        name: Description used in reports.
        nodal_forces: Force vector in N for each loaded node.
        include_self_weight: Also apply the weight of the members.
    """

    name: str
    nodal_forces: dict[int, np.ndarray] = field(default_factory=dict)
    include_self_weight: bool = True

    def combined_with(self, other: LoadCase, name: str | None = None) -> LoadCase:
        """A new case applying both sets of forces (self-weight if either includes it)."""
        forces = {node: force.copy() for node, force in self.nodal_forces.items()}
        for node, force in other.nodal_forces.items():
            forces[node] = forces.get(node, np.zeros(3)) + force
        return LoadCase(
            name or f"{self.name} + {other.name}", forces, self.include_self_weight or other.include_self_weight
        )


def _tributary_lengths(values: list[float]) -> dict[float, float]:
    """Half the distance to each neighbor, for sorted grid positions."""
    widths = {}
    for position, value in enumerate(values):
        before = values[position - 1] if position > 0 else value
        after = values[position + 1] if position + 1 < len(values) else value
        widths[value] = (after - before) / 2
    return widths


def _grid_nodes(design: BridgeDesign, fixed_axis: int, fixed_value: float) -> dict[tuple[float, float], int]:
    """Nodes lying in the plane ``coordinate[fixed_axis] == fixed_value``, keyed by their other two coordinates."""
    other_axes = [axis for axis in range(3) if axis != fixed_axis]
    return {
        (round(node[other_axes[0]], 9), round(node[other_axes[1]], 9)): index
        for index, node in enumerate(design.nodes)
        if math.isclose(node[fixed_axis], fixed_value, abs_tol=1e-9)
    }


def _pressure_on_face(
    design: BridgeDesign, fixed_axis: int, fixed_value: float, pressure: float, direction: np.ndarray
) -> dict[int, np.ndarray]:
    """Nodal forces equivalent to a uniform pressure on the face ``coordinate[fixed_axis] == fixed_value``."""
    grid = _grid_nodes(design, fixed_axis, fixed_value)
    first_values = sorted({key[0] for key in grid})
    second_values = sorted({key[1] for key in grid})
    if len(grid) != len(first_values) * len(second_values):
        raise ValueError("face nodes do not form a full grid; cannot assign tributary areas")
    first_widths, second_widths = _tributary_lengths(first_values), _tributary_lengths(second_values)
    return {
        node: pressure * first_widths[first] * second_widths[second] * direction
        for (first, second), node in grid.items()
    }


def deck_pressure_load(design: BridgeDesign, pressure: float, name: str | None = None) -> LoadCase:
    """Uniform downward pressure on the deck, lumped to deck nodes by tributary area.

    Args:
        design: The bridge.
        pressure: Load per unit deck area, in Pa (N/m^2).
        name: Report label.

    Returns:
        A load case whose total equals ``pressure * span * width``, without self-weight.
    """
    z_min = min(node[2] for node in design.nodes)
    forces = _pressure_on_face(design, 2, z_min, pressure, np.array([0.0, 0.0, -1.0]))
    return LoadCase(name or f"deck pressure {pressure / 1e3:g} kPa", forces, include_self_weight=False)


def lateral_pressure_load(design: BridgeDesign, pressure: float, name: str | None = None) -> LoadCase:
    """Uniform lateral (wind) pressure on the y = 0 side face, acting in +y.

    Args:
        design: The bridge.
        pressure: Load per unit side-face area, in Pa.
        name: Report label.

    Returns:
        A load case whose total equals ``pressure * span * height``, without self-weight.
    """
    y_min = min(node[1] for node in design.nodes)
    forces = _pressure_on_face(design, 1, y_min, pressure, np.array([0.0, 1.0, 0.0]))
    return LoadCase(name or f"lateral pressure {pressure / 1e3:g} kPa", forces, include_self_weight=False)


def axle_load(design: BridgeDesign, force: float, x_position: float, name: str | None = None) -> LoadCase:
    """A downward line load across the deck at ``x_position``, such as one vehicle axle.

    The force is split between the two deck stations either side of ``x_position`` by the
    lever rule, and shared equally by the deck nodes across each station.

    Args:
        design: The bridge.
        force: Total axle force, in N.
        x_position: Position along the span, in m.
        name: Report label.

    Raises:
        ValueError: If ``x_position`` lies outside the span.
    """
    z_min = min(node[2] for node in design.nodes)
    deck = [(index, node) for index, node in enumerate(design.nodes) if math.isclose(node[2], z_min, abs_tol=1e-9)]
    stations = sorted({round(node[0], 9) for _, node in deck})
    if not stations[0] - 1e-9 <= x_position <= stations[-1] + 1e-9:
        raise ValueError(f"x_position {x_position} is outside the span")
    upper = min(max(1, int(np.searchsorted(stations, x_position))), len(stations) - 1)
    lower = upper - 1
    span = stations[upper] - stations[lower]
    upper_share = (x_position - stations[lower]) / span
    shares = {stations[lower]: 1 - upper_share, stations[upper]: upper_share}
    forces: dict[int, np.ndarray] = {}
    for station, share in shares.items():
        at_station = [index for index, node in deck if math.isclose(node[0], station, abs_tol=1e-9)]
        for index in at_station:
            forces[index] = forces.get(index, np.zeros(3)) + np.array([0.0, 0.0, -force * share / len(at_station)])
    return LoadCase(name or f"axle {force / 1e3:g} kN at x={x_position:g} m", forces, include_self_weight=False)


def moving_axle_loads(design: BridgeDesign, force: float) -> list[LoadCase]:
    """One :func:`axle_load` case at every deck station, to envelope a load crossing the bridge."""
    z_min = min(node[2] for node in design.nodes)
    stations = sorted({round(node[0], 9) for node in design.nodes if math.isclose(node[2], z_min, abs_tol=1e-9)})
    return [axle_load(design, force, station) for station in stations]


SectionsArgument = CrossSection | Sequence[CrossSection] | None
"""One section for every member, one section per member in ``design.members`` order, or
None to use the section of each member's inventory edge."""

MaterialsArgument = Material | Sequence[Material] | None
"""One material for every member, one material per member in ``design.members`` order, or
None to use the material of each member's inventory edge."""


def _per_member(design: BridgeDesign, value, kind: type, attribute: str) -> list:
    """Expand a sections or materials argument to one entry per member."""
    if value is None:
        missing = [a.edge.index for a in design.assignments if getattr(a.edge, attribute) is None]
        if missing:
            raise ValueError(
                f"inventory edges {missing[:5]}{' ...' if len(missing) > 5 else ''} have no {attribute}; "
                f"pass {attribute}s explicitly or use an inventory with {attribute} columns"
            )
        return [getattr(assignment.edge, attribute) for assignment in design.assignments]
    if isinstance(value, kind):
        return [value] * len(design.members)
    expanded = list(value)
    if len(expanded) != len(design.members):
        raise ValueError(f"need {len(design.members)} {attribute}s, one per member; got {len(expanded)}")
    return expanded


def member_sections(design: BridgeDesign, sections: SectionsArgument = None) -> list[CrossSection]:
    """Expand a sections argument to one section per member.

    Raises:
        ValueError: If a sequence of the wrong length is given, or sections are taken from
            the inventory and some edge has none.
    """
    return _per_member(design, sections, CrossSection, "section")


def member_materials(design: BridgeDesign, materials: MaterialsArgument = None) -> list[Material]:
    """Expand a materials argument to one material per member.

    Raises:
        ValueError: If a sequence of the wrong length is given, or materials are taken from
            the inventory and some edge has none.
    """
    return _per_member(design, materials, Material, "material")


def self_weight_forces(
    design: BridgeDesign, sections: SectionsArgument = None, materials: MaterialsArgument = None
) -> np.ndarray:
    """Member weights lumped half to each end node, shape ``(node_count, 3)``, in N."""
    forces = np.zeros((len(design.nodes), 3))
    for member, section, material in zip(
        design.members, member_sections(design, sections), member_materials(design, materials), strict=True
    ):
        weight = section.mass_per_length(material) * member.required_length * STANDARD_GRAVITY
        forces[member.start_node, 2] -= weight / 2
        forces[member.end_node, 2] -= weight / 2
    return forces


def total_mass(design: BridgeDesign, sections: SectionsArgument = None, materials: MaterialsArgument = None) -> float:
    """Total mass of all members, in kg."""
    return sum(
        section.mass_per_length(material) * member.required_length
        for member, section, material in zip(
            design.members, member_sections(design, sections), member_materials(design, materials), strict=True
        )
    )


def check_member(
    length: float,
    axial_force: float,
    section: CrossSection,
    material: Material,
    max_compression_slenderness: float | None = None,
) -> tuple[float, float, str]:
    """Strength check of one pin-ended member, by its material's strength method.

    Args:
        length: Member length, in m.
        axial_force: Tension positive, in N.
        section: Member cross-section.
        material: Member material; its safety factor divides nominal strengths.
        max_compression_slenderness: Optional limit on ``L/r`` for members in compression.

    Returns:
        ``(available_strength, utilization, limit_state)``. Utilization is infinite when a
        compression member has slender walls or exceeds the slenderness limit, because
        those cases fall outside what this check covers.
    """
    safety_factor = material.safety_factor
    if axial_force >= 0:
        available = nominal_tensile_strength(section, material) / safety_factor
        return available, axial_force / available, "tension yielding"
    available = nominal_compressive_strength(length, section, material) / safety_factor
    if has_slender_walls(section, material):
        return available, math.inf, SLENDER_WALL
    if max_compression_slenderness is not None and length / section.radius_of_gyration > max_compression_slenderness:
        return available, math.inf, SLENDERNESS_LIMIT
    return available, -axial_force / available, "flexural buckling"


# --------------------------------------------------------------------------- load testing


@dataclass(frozen=True)
class MemberCheck:
    """Strength check of one member under one load case.

    Attributes:
        member_index: Index into ``design.members``.
        axial_force: Tension positive, in N.
        slenderness: Length over least radius of gyration.
        available_strength: Nominal strength divided by the safety factor, for the
            governing sense of force, in N.
        utilization: ``|axial_force| / available_strength``; above 1 fails; infinite for
            cases outside the check's scope (see :func:`check_member`).
        limit_state: ``"tension yielding"``, ``"flexural buckling"``, or the reason the
            member falls outside the check's scope.
        section_name: Name of the member's cross-section.
        material_name: Name of the member's material.
    """

    member_index: int
    axial_force: float
    slenderness: float
    available_strength: float
    utilization: float
    limit_state: str
    section_name: str = ""
    material_name: str = ""


@dataclass
class LoadCaseResult:
    """Outcome of one load case.

    Attributes:
        load_case: The case analyzed.
        response: Displacements, member forces and reactions.
        member_checks: One strength check per member.
        applied_force: Sum of all applied forces including self-weight, in N.
    """

    load_case: LoadCase
    response: TrussResponse
    member_checks: list[MemberCheck]
    applied_force: np.ndarray

    @property
    def governing_check(self) -> MemberCheck:
        """The member check with the highest utilization."""
        return max(self.member_checks, key=lambda check: check.utilization)

    @property
    def max_vertical_deflection(self) -> float:
        """Largest downward node displacement, in m (positive downward)."""
        return float(max(0.0, -self.response.displacements[:, 2].min()))

    @property
    def equilibrium_error(self) -> float:
        """Norm of applied force plus reactions, in N; near zero for a correct solution."""
        return float(np.linalg.norm(self.applied_force + sum(self.response.reactions.values())))


@dataclass
class LoadTestReport:
    """Results of every load case on one design.

    Attributes:
        design: The bridge tested.
        sections: Cross-section of each member, in ``design.members`` order.
        materials: Material of each member, in ``design.members`` order.
        results: One result per load case.
        deflection_limit: Allowed downward deflection in m, or None for no check.
    """

    design: BridgeDesign
    sections: list[CrossSection]
    materials: list[Material]
    results: list[LoadCaseResult]
    deflection_limit: float | None = None

    @property
    def total_mass(self) -> float:
        """Total mass of all members, in kg."""
        return total_mass(self.design, self.sections, self.materials)

    def materials_summary(self) -> str:
        """Each distinct material with its member count, strength method and safety factor."""
        counts: dict[Material, int] = {}
        for material in self.materials:
            counts[material] = counts.get(material, 0) + 1
        return ", ".join(
            f"{material.name} x{count} ({STRENGTH_METHOD_LABELS[material.strength_method]}, "
            f"safety factor {material.safety_factor:g})"
            for material, count in sorted(counts.items(), key=lambda item: -item[1])
        )

    @property
    def max_vertical_deflection(self) -> float:
        """Largest downward deflection over all load cases, in m."""
        return max(result.max_vertical_deflection for result in self.results)

    @property
    def governing_result(self) -> LoadCaseResult:
        """The load case producing the highest member utilization."""
        return max(self.results, key=lambda result: result.governing_check.utilization)

    @property
    def passed(self) -> bool:
        """True if every member is within strength in every case and deflection is within the limit."""
        strength_ok = all(result.governing_check.utilization <= 1.0 for result in self.results)
        deflection_ok = self.deflection_limit is None or all(
            result.max_vertical_deflection <= self.deflection_limit for result in self.results
        )
        return strength_ok and deflection_ok

    def summary(self) -> str:
        """One line per load case, then the verdict and governing member."""
        distinct = sorted({section.name for section in self.sections})
        section_text = distinct[0] if len(distinct) == 1 else f"{len(distinct)} section sizes"
        lines = [
            f"Load test: {section_text}, member mass {self.total_mass:,.0f} kg"
            + (f", deflection limit {self.deflection_limit * 1e3:.1f} mm" if self.deflection_limit else ""),
            f"  materials: {self.materials_summary()}",
            f"  {'load case':<42} {'max util.':>9} {'governing member':<40} {'deflection':>11}",
        ]
        for result in self.results:
            check = result.governing_check
            member = self.design.members[check.member_index]
            label = f"#{check.member_index} {member.role} ({check.limit_state})"
            lines.append(
                f"  {result.load_case.name[:42]:<42} {check.utilization:9.3f} {label[:40]:<40} "
                f"{result.max_vertical_deflection * 1e3:8.2f} mm"
            )
        governing = self.governing_result
        check = governing.governing_check
        member_index = check.member_index
        lines.append(
            f"  {'PASS' if self.passed else 'FAIL'}: governing case '{governing.load_case.name}', member "
            f"#{member_index} ({self.design.members[member_index].role}, {check.material_name} "
            f"{check.section_name}), "
            f"force {check.axial_force / 1e3:+.1f} kN vs available {check.available_strength / 1e3:.1f} kN"
        )
        return "\n".join(lines)


def _applied_forces(
    design: BridgeDesign, sections: list[CrossSection], materials: list[Material], load_case: LoadCase
) -> np.ndarray:
    """Nodal forces of a load case, including self-weight if the case asks for it."""
    forces = np.zeros((len(design.nodes), 3))
    for node, force in load_case.nodal_forces.items():
        forces[node] += force
    if load_case.include_self_weight:
        forces += self_weight_forces(design, sections, materials)
    return forces


def _analyze_cases(
    design: BridgeDesign,
    sections: list[CrossSection],
    materials: list[Material],
    load_cases: list[LoadCase],
    max_compression_slenderness: float | None,
) -> list[LoadCaseResult]:
    """Analyze several load cases with one stiffness factorization and check every member."""
    coordinates = np.asarray(design.nodes, float)
    endpoints = [(member.start_node, member.end_node) for member in design.members]
    stiffness = np.array(
        [material.elastic_modulus * section.area for section, material in zip(sections, materials, strict=True)]
    )
    force_sets = [_applied_forces(design, sections, materials, case) for case in load_cases]
    responses = solve_truss_cases(coordinates, endpoints, stiffness, bearing_restraints(design), force_sets)
    lengths = [member.required_length for member in design.members]
    results = []
    for load_case, forces, response in zip(load_cases, force_sets, responses, strict=True):
        checks = []
        for member_index, (length, section, material, axial_force) in enumerate(
            zip(lengths, sections, materials, response.axial_forces, strict=True)
        ):
            available, utilization, limit_state = check_member(
                length, float(axial_force), section, material, max_compression_slenderness
            )
            checks.append(
                MemberCheck(
                    member_index,
                    float(axial_force),
                    length / section.radius_of_gyration,
                    available,
                    utilization,
                    limit_state,
                    section.name,
                    material.name,
                )
            )
        results.append(LoadCaseResult(load_case, response, checks, forces.sum(axis=0)))
    return results


def analyze_load_case(
    design: BridgeDesign,
    load_case: LoadCase,
    *,
    sections: SectionsArgument = None,
    materials: MaterialsArgument = None,
    max_compression_slenderness: float | None = None,
) -> LoadCaseResult:
    """Analyze one load case and check every member's strength.

    Args:
        design: The bridge; coordinates in m.
        load_case: Forces to apply.
        sections: One section for all members, one per member, or None (the default) for
            the section of each member's inventory edge.
        materials: One material for all members, one per member, or None (the default) for
            the material of each member's inventory edge.
        max_compression_slenderness: Optional limit on ``L/r`` for members in compression.

    Returns:
        The analysis result.

    Raises:
        ValueError: If sections or materials are missing or of the wrong length.
    """
    per_member_sections = member_sections(design, sections)
    per_member_materials = member_materials(design, materials)
    return _analyze_cases(
        design, per_member_sections, per_member_materials, [load_case], max_compression_slenderness
    )[0]


def run_load_tests(
    design: BridgeDesign,
    load_cases: list[LoadCase],
    *,
    sections: SectionsArgument = None,
    materials: MaterialsArgument = None,
    deflection_limit_ratio: float | None = None,
    max_compression_slenderness: float | None = None,
) -> LoadTestReport:
    """Analyze every load case and collect a pass/fail report.

    All cases share one stiffness factorization, so many cases cost little more than one.
    Each member is checked by its own material's strength method and safety factor.

    Args:
        design: The bridge; coordinates in m.
        load_cases: Cases to analyze.
        sections: One section for all members, one per member, or None (the default) for
            the section of each member's inventory edge.
        materials: One material for all members, one per member, or None (the default) for
            the material of each member's inventory edge.
        deflection_limit_ratio: If given, deflection must not exceed span / ratio.
        max_compression_slenderness: Optional limit on ``L/r`` for members in compression.

    Returns:
        The report.

    Raises:
        ValueError: If sections or materials are missing or of the wrong length.
    """
    per_member_sections = member_sections(design, sections)
    per_member_materials = member_materials(design, materials)
    results = _analyze_cases(design, per_member_sections, per_member_materials, load_cases, max_compression_slenderness)
    limit = design.span / deflection_limit_ratio if deflection_limit_ratio else None
    return LoadTestReport(design, per_member_sections, per_member_materials, results, limit)


def basic_load_cases(
    design: BridgeDesign, *, deck_pressure: float, axle_force: float, lateral_pressure: float
) -> list[LoadCase]:
    """A basic set of cases: self-weight, uniform deck load, an axle at every station, and wind.

    Every case includes self-weight. Values are the caller's; no design code is implied.

    Args:
        design: The bridge.
        deck_pressure: Uniform deck load, in Pa.
        axle_force: Moving axle force, in N.
        lateral_pressure: Lateral pressure on one side face, in Pa.
    """
    dead = LoadCase("self-weight")
    cases = [dead, dead.combined_with(deck_pressure_load(design, deck_pressure))]
    cases += [dead.combined_with(axle) for axle in moving_axle_loads(design, axle_force)]
    cases.append(dead.combined_with(lateral_pressure_load(design, lateral_pressure)))
    return cases
