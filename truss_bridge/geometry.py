"""truss_bridge/geometry.py

Exact integer geometry on the node lattice.

All functions here work on integer lattice coordinates, so every test is exact; fractions
are used where a parameter along a segment is needed.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

from fractions import Fraction

IntVector = tuple[int, int, int]


def subtract(minuend: IntVector, subtrahend: IntVector) -> IntVector:
    """Component-wise difference ``minuend - subtrahend``."""
    return (minuend[0] - subtrahend[0], minuend[1] - subtrahend[1], minuend[2] - subtrahend[2])


def dot(first: IntVector, second: IntVector) -> int:
    """Dot product of two integer vectors."""
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def cross(first: IntVector, second: IntVector) -> IntVector:
    """Cross product of two integer vectors."""
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def is_lattice_direction(direction: IntVector) -> bool:
    """Whether a member may run along ``direction``.

    Allowed directions are the coordinate axes and the 45-degree diagonals that lie in a
    coordinate plane (two equal non-zero components, one zero component). These are the
    only directions that meet axis-aligned members at multiples of 45 degrees.

    Args:
        direction: Vector from one end of the member to the other.

    Returns:
        True for an axis or in-plane 45-degree direction, False otherwise (including zero).
    """
    magnitudes = sorted(abs(component) for component in direction)
    smallest, middle, largest = magnitudes
    return largest > 0 and smallest == 0 and middle in (0, largest)


def lies_strictly_inside_segment(point: IntVector, segment_start: IntVector, segment_end: IntVector) -> bool:
    """Whether ``point`` lies on the segment, excluding its two endpoints."""
    segment = subtract(segment_end, segment_start)
    offset = subtract(point, segment_start)
    if cross(segment, offset) != (0, 0, 0):
        return False
    projection = dot(offset, segment)
    return 0 < projection < dot(segment, segment)


def segments_overlap(
    first_start: IntVector, first_end: IntVector, second_start: IntVector, second_end: IntVector
) -> bool:
    """Whether two segments share any point other than a common endpoint.

    Two members may meet at a shared node, but may not cross or overlap anywhere else.

    Returns:
        True if the segments cross, touch mid-span, or overlap collinearly.
    """
    shared_endpoints = {first_start, first_end} & {second_start, second_end}
    first_direction = subtract(first_end, first_start)
    second_direction = subtract(second_end, second_start)
    start_offset = subtract(second_start, first_start)
    if dot(start_offset, cross(first_direction, second_direction)) != 0:
        return False  # the lines are skew, so they never meet
    normal = cross(first_direction, second_direction)
    if normal == (0, 0, 0):  # parallel
        if cross(first_direction, start_offset) != (0, 0, 0):
            return False  # parallel but not collinear
        length_squared = dot(first_direction, first_direction)
        param_a = Fraction(dot(start_offset, first_direction), length_squared)
        param_b = Fraction(dot(subtract(second_end, first_start), first_direction), length_squared)
        low, high = sorted((param_a, param_b))
        overlap_low, overlap_high = max(low, 0), min(high, 1)
        return overlap_high > overlap_low or (overlap_high == overlap_low and not shared_endpoints)
    normal_squared = dot(normal, normal)
    first_param = Fraction(dot(cross(start_offset, second_direction), normal), normal_squared)
    second_param = Fraction(dot(cross(start_offset, first_direction), normal), normal_squared)
    if not (0 <= first_param <= 1 and 0 <= second_param <= 1):
        return False
    meets_at_shared_endpoint = shared_endpoints and first_param in (0, 1) and second_param in (0, 1)
    return not meets_at_shared_endpoint


def classify_member(start: IntVector, end: IntVector, *, width_units: int, height_units: int) -> str:
    """Name the structural role of a member from its position in the box section.

    Coordinates are lattice integers: x along the span, y across the width
    (0..width_units), z upward (0..height_units).

    Args:
        start: One end of the member.
        end: The other end.
        width_units: Bridge width in lattice units.
        height_units: Bridge height in lattice units.

    Returns:
        A role name such as ``"bottom_chord"``, ``"side_diagonal"`` or ``"sway_brace"``.
    """
    direction = subtract(end, start)
    varying_axes = [axis for axis in range(3) if direction[axis]]
    in_side_face = start[1] == end[1] and start[1] in (0, width_units)
    at_deck = start[2] == end[2] == 0
    at_top = start[2] == end[2] == height_units
    if varying_axes == [0]:
        if in_side_face and at_deck:
            return "bottom_chord"
        if in_side_face and at_top:
            return "top_chord"
        return "stringer"
    if varying_axes == [1]:
        return "floor_beam" if at_deck else "top_strut" if at_top else "cross_strut"
    if varying_axes == [2]:
        return "vertical" if in_side_face else "interior_post"
    if varying_axes == [0, 2]:
        return "side_diagonal" if in_side_face else "interior_diagonal"
    if varying_axes == [0, 1]:
        return "bottom_lateral" if at_deck else "top_lateral" if at_top else "lateral"
    return "sway_brace"
