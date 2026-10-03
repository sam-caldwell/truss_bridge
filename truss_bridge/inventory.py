"""truss_bridge/inventory.py

Reading, writing and generating inventories of parts as CSV.

Columns, in this order when the file has no header row::

    index,length,material,elastic_modulus,yield_strength,density,strength_method,safety_factor,section_width,section_wall

Only ``index`` and ``length`` are required, which is enough to design a bridge. Load tests
also need each part's material (the six material columns, filled in together) and
section (``section_width`` and ``section_wall`` of a square hollow section, in m; a wall
of half the width is a solid square bar). Units are SI: m, Pa, kg/m^3. Every row naming
the same material must give it the same properties. A row may leave a whole column group
blank when that part's material or section is unknown.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import csv
import math
import random

from .analysis import ALLOWABLE_STRENGTH_SAFETY_FACTOR
from .model import AISC_360_16, YIELD_EULER, CrossSection, Edge, Material
from .solver import lattice_units

MATERIAL_COLUMNS = ("material", "elastic_modulus", "yield_strength", "density", "strength_method", "safety_factor")
SECTION_COLUMNS = ("section_width", "section_wall")
INVENTORY_COLUMNS = ("index", "length", *MATERIAL_COLUMNS, *SECTION_COLUMNS)


def _parse_material(values: dict[str, str], where: str) -> Material | None:
    filled = [column for column in MATERIAL_COLUMNS if values.get(column, "").strip()]
    if not filled:
        return None
    if len(filled) != len(MATERIAL_COLUMNS):
        missing = [column for column in MATERIAL_COLUMNS if column not in filled]
        raise ValueError(f"{where}: material columns must be filled in together; missing {', '.join(missing)}")
    try:
        return Material(
            values["material"].strip(),
            float(values["elastic_modulus"]),
            float(values["yield_strength"]),
            float(values["density"]),
            values["strength_method"].strip(),
            float(values["safety_factor"]),
        )
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


def _parse_section(values: dict[str, str], where: str) -> CrossSection | None:
    filled = [column for column in SECTION_COLUMNS if values.get(column, "").strip()]
    if not filled:
        return None
    if len(filled) != len(SECTION_COLUMNS):
        raise ValueError(f"{where}: section_width and section_wall must be given together")
    try:
        return CrossSection.square_hollow(float(values["section_width"]), float(values["section_wall"]))
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


def read_inventory(path: str) -> list[Edge]:
    """Read parts from a CSV file; see the module docstring for the columns.

    Args:
        path: CSV file. A header row is optional; with one, columns may be in any order.

    Returns:
        One edge per row, carrying its material and section when given.

    Raises:
        ValueError: On an unknown or missing column, a malformed row, a partly filled
            material or section, or one material name given different properties.
    """
    edges: list[Edge] = []
    materials: dict[str, tuple[Material, str]] = {}
    with open(path, newline="") as handle:
        rows = [(number, row) for number, row in enumerate(csv.reader(handle), start=1) if any(c.strip() for c in row)]
    if not rows:
        return edges
    header = [cell.strip() for cell in rows[0][1]]
    if header[0].lstrip("-").isdigit():
        header = list(INVENTORY_COLUMNS)
    else:
        rows = rows[1:]
        unknown = [column for column in header if column not in INVENTORY_COLUMNS]
        if unknown or "index" not in header or "length" not in header:
            raise ValueError(
                f"{path}: header needs index and length, and may add {', '.join(INVENTORY_COLUMNS[2:])}; "
                f"got {', '.join(header)}"
            )
    for number, row in rows:
        where = f"{path}, line {number}"
        if len(row) > len(header):
            raise ValueError(f"{where}: {len(row)} values but only {len(header)} columns")
        values = dict(zip(header, row, strict=False))
        try:
            index, length = int(values["index"]), float(values["length"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"{where}: index and length must be numbers") from error
        material = _parse_material(values, where)
        if material is not None:
            known, first = materials.setdefault(material.name, (material, where))
            if known != material:
                raise ValueError(f"{where}: material {material.name!r} has different properties than at {first}")
            material = known
        edges.append(Edge(index, length, material, _parse_section(values, where)))
    return edges


def _number(value: float) -> str:
    """Up to 12 significant digits; scientific notation such as ``2e11`` from a million up."""
    if abs(value) < 1e6:
        return f"{value:.12g}"
    mantissa, exponent = f"{value:.11e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def write_inventory(path: str, edges: list[Edge]) -> str:
    """Write parts to a CSV file with a header row, leaving unknown materials and sections blank.

    Returns:
        ``path``.

    Raises:
        ValueError: If a section was not made by :meth:`CrossSection.square_hollow`.
    """
    for edge in edges:
        if edge.section is not None and (edge.section.outer_width is None or edge.section.wall_thickness is None):
            raise ValueError(f"edge #{edge.index}: only square hollow sections can be written")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INVENTORY_COLUMNS)
        for edge in edges:
            row = [edge.index, _number(edge.length)]
            material = edge.material
            if material is None:
                row += [""] * len(MATERIAL_COLUMNS)
            else:
                properties = (material.elastic_modulus, material.yield_strength, material.density)
                row += [material.name, *map(_number, properties), material.strength_method]
                row.append(_number(material.safety_factor))
            if edge.section is None:
                row += [""] * len(SECTION_COLUMNS)
            else:
                row += [_number(edge.section.outer_width), _number(edge.section.wall_thickness)]
            writer.writerow(row)
    return path


_EXAMPLE_PARTS = (
    (
        Material("steel", 200e9, 250e6, 7850.0, AISC_360_16, ALLOWABLE_STRENGTH_SAFETY_FACTOR),
        CrossSection.square_hollow(0.100, 0.005),
    ),
    (
        Material("gfrp", 23e9, 200e6, 1900.0, YIELD_EULER, 3.0),
        CrossSection.square_hollow(0.150, 0.010),
    ),
)
"""Nominal example parts, not specified values: steel tube, and glass-fiber-reinforced
plastic (pultruded) tube with a safety factor of 3 for creep and environmental effects."""


def example_inventory(length: float, width: float, height: float, copies: int = 120, seed: int = 7) -> list[Edge]:
    """Lattice-multiple lengths and their 45-degree diagonals, slightly long, plus random offcuts.

    Lengths are rounded up to 0.1 mm so no edge is shorter than the member it targets.
    Every third part is example plastic (GFRP) tube; the rest are example steel tube.
    """
    generator = random.Random(seed)
    unit = lattice_units(length, width, height)[0]
    multiples = range(1, round(max(width, height) / unit) + 1)
    lengths = [k * unit for k in multiples] + [k * unit * math.sqrt(2) for k in multiples]
    stock = [value + generator.choice([0.0, 0.0, 0.05]) for value in lengths for _ in range(copies)]
    stock += [generator.uniform(0.5, 2 * max(width, height)) for _ in range(40)]
    steel, plastic = _EXAMPLE_PARTS
    return [
        Edge(index, math.ceil(value * 1e4) / 1e4, *(plastic if index % 3 == 2 else steel))
        for index, value in enumerate(stock)
    ]
