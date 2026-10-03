"""truss_bridge/model.py

Data model shared by the design, verification, analysis and plotting modules.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

Point3D = tuple[float, float, float]

AISC_360_16 = "aisc_360_16"
"""Strength method for structural steel: AISC 360-16 tension yielding and column curve."""
YIELD_EULER = "yield_euler"
"""Strength method for other materials, such as plastics: yield in tension, the lesser of
yield and elastic (Euler) buckling in compression."""
STRENGTH_METHODS = (AISC_360_16, YIELD_EULER)


class NoDesignFoundError(Exception):
    """Raised when no rigid, rule-compliant bridge can be built from the given inventory."""


class SearchBudgetExceededError(NoDesignFoundError):
    """Raised when the exact search runs out of time or cut rounds before proving a result."""


@dataclass(frozen=True)
class Material:
    """Linear-elastic material properties and how member strength is checked.

    Attributes:
        name: Short identifier, for example ``"steel"`` or ``"gfrp"``.
        elastic_modulus: Young's modulus E, in Pa.
        yield_strength: Yield (or design strength) stress Fy, in Pa.
        density: Mass density, in kg/m^3.
        strength_method: One of :data:`STRENGTH_METHODS`; see :func:`analysis.check_member`.
        safety_factor: Divides nominal strengths, for example 1.67 (AISC ASD Omega) for steel.

    Raises:
        ValueError: If a property is not positive, the method is unknown, or the safety
            factor is below 1.
    """

    name: str
    elastic_modulus: float
    yield_strength: float
    density: float
    strength_method: str
    safety_factor: float

    def __post_init__(self) -> None:
        if min(self.elastic_modulus, self.yield_strength, self.density) <= 0:
            raise ValueError(f"material {self.name!r}: elastic_modulus, yield_strength and density must be positive")
        if self.strength_method not in STRENGTH_METHODS:
            raise ValueError(
                f"material {self.name!r}: unknown strength_method {self.strength_method!r}; "
                f"expected one of {STRENGTH_METHODS}"
            )
        if not self.safety_factor >= 1:
            raise ValueError(f"material {self.name!r}: safety_factor must be at least 1")


@dataclass(frozen=True)
class CrossSection:
    """Member cross-section properties.

    Attributes:
        name: Description.
        area: Gross area Ag, in m^2.
        least_second_moment: Smallest second moment of area I, in m^4 (governs buckling).
        wall_slenderness: Width-to-thickness ratio ``b/t`` of the most slender wall, or None
            if not known (then no local-buckling screen is applied).
        outer_width: Outside width of a square hollow section, in m, or None for other shapes.
        wall_thickness: Wall thickness of a square hollow section, in m, or None.
    """

    name: str
    area: float
    least_second_moment: float
    wall_slenderness: float | None = None
    outer_width: float | None = None
    wall_thickness: float | None = None

    @property
    def radius_of_gyration(self) -> float:
        """Least radius of gyration ``r = sqrt(I / A)``, in m."""
        return math.sqrt(self.least_second_moment / self.area)

    @classmethod
    def square_hollow(cls, outer_width: float, wall_thickness: float) -> CrossSection:
        """A square hollow section with sharp corners (corner radii ignored).

        A wall thickness of half the width gives a solid square bar.

        Args:
            outer_width: Outside width b, in m.
            wall_thickness: Wall thickness t, in m.

        Returns:
            Section with ``A = b^2 - (b - 2t)^2``, ``I = (b^4 - (b - 2t)^4) / 12`` and wall
            slenderness ``(b - 3t) / t`` (AISC 360-16 Section B4.1b, corner radius unknown).

        Raises:
            ValueError: If the dimensions are not positive or the wall is too thick.
        """
        inner_width = outer_width - 2 * wall_thickness
        if outer_width <= 0 or wall_thickness <= 0 or inner_width < 0:
            raise ValueError("need outer_width >= 2 * wall_thickness > 0")
        return cls(
            f"SHS {outer_width * 1e3:g}x{outer_width * 1e3:g}x{wall_thickness * 1e3:g} mm",
            outer_width**2 - inner_width**2,
            (outer_width**4 - inner_width**4) / 12,
            max(0.0, (outer_width - 3 * wall_thickness) / wall_thickness),
            outer_width,
            wall_thickness,
        )

    def mass_per_length(self, material: Material) -> float:
        """Mass per unit length, in kg/m."""
        return material.density * self.area


@dataclass(frozen=True)
class Edge:
    """A straight stock member available for construction.

    Attributes:
        index: Caller-supplied identifier, unique within the inventory.
        length: Usable length, in the same unit as the bridge dimensions.
        material: What the part is made of, or None if unknown (enough for design, not for
            load testing).
        section: The part's cross-section, or None if unknown.
    """

    index: int
    length: float
    material: Material | None = None
    section: CrossSection | None = None


@dataclass(frozen=True)
class Member:
    """A straight, pin-ended bar of the bridge connecting two nodes.

    Attributes:
        start_node: Index into ``BridgeDesign.nodes`` of one end.
        end_node: Index into ``BridgeDesign.nodes`` of the other end.
        role: Structural role, for example ``"bottom_chord"`` or ``"sway_brace"``.
        required_length: Node-to-node distance the member must span.
    """

    start_node: int
    end_node: int
    role: str
    required_length: float


@dataclass(frozen=True)
class MemberAssignment:
    """The inventory edge chosen to build one member.

    Attributes:
        member: The member being built.
        edge: The stock edge cut to length for it.
    """

    member: Member
    edge: Edge

    @property
    def trim(self) -> float:
        """Length cut off the edge so that it fits the member exactly."""
        return self.edge.length - self.member.required_length


@dataclass
class BridgeDesign:
    """A complete bridge: node coordinates, members and the edges used to build them.

    Attributes:
        node_set: Name of the node set the design was found on, for example ``"chords"``.
        lattice_unit: Spacing of the integer lattice the nodes were placed on.
        nodes: Node coordinates (x along the span, y across the width, z upward).
        assignments: One entry per member, pairing it with its inventory edge.
        unused_edges: Inventory edges left over after construction.
        search_method: ``"exact"`` if the total trim is proven minimal for the node set, or
            ``"constructive"`` if the design is rigid and valid but not proven optimal.
    """

    node_set: str
    lattice_unit: float
    nodes: list[Point3D]
    assignments: list[MemberAssignment]
    unused_edges: list[Edge] = field(default_factory=list)
    search_method: str = "exact"

    @property
    def members(self) -> list[Member]:
        """Members of the bridge, in assignment order."""
        return [assignment.member for assignment in self.assignments]

    @property
    def total_trim(self) -> float:
        """Total length cut off inventory edges, a measure of material waste."""
        return sum(assignment.trim for assignment in self.assignments)

    @property
    def span(self) -> float:
        """Overall length of the bridge along x."""
        return max(x for x, _, _ in self.nodes) - min(x for x, _, _ in self.nodes)

    def role_counts(self) -> Counter[str]:
        """Number of members in each structural role."""
        return Counter(member.role for member in self.members)

    def summary(self, include_assignments: bool = False) -> str:
        """Human-readable description of the design.

        Args:
            include_assignments: Also list every member with its edge and trim.

        Returns:
            A multi-line string.
        """
        lines = [
            f"{self.node_set} node set ({self.search_method} search), lattice unit {self.lattice_unit:g}, "
            f"{len(self.nodes)} nodes, "
            f"{len(self.assignments)} members, total trim {self.total_trim:.4f}",
            "  " + ", ".join(f"{role}: {count}" for role, count in sorted(self.role_counts().items())),
        ]
        if include_assignments:
            for assignment in self.assignments:
                member, edge = assignment.member, assignment.edge
                part = " ".join(
                    text for text in (edge.material and edge.material.name, edge.section and edge.section.name) if text
                )
                lines.append(
                    f"  edge #{edge.index:<5} ({edge.length:9.4f}) -> {member.role:<17} "
                    f"n{member.start_node}-n{member.end_node} needs {member.required_length:9.4f}, "
                    f"trim {assignment.trim:.4f}" + (f", {part}" if part else "")
                )
        return "\n".join(lines)
