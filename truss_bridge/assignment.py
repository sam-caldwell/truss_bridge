"""truss_bridge/assignment.py

Matching inventory edges to members.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import heapq

from .model import Edge, Member, MemberAssignment


def assign_edges_to_members(
    members: list[Member], edges: list[Edge], max_trim: float, tolerance: float = 1e-9
) -> list[MemberAssignment] | None:
    """Give every member its own edge, cutting each edge by at most ``max_trim``.

    An edge fits a member when ``required_length <= edge.length <= required_length + max_trim``.
    The edges that fit a member form a contiguous run of the length-sorted inventory, so the
    problem is a matching on a convex bipartite graph. Glover's (1967) algorithm scans edges
    from shortest to longest and gives each one to the open member whose window closes
    first; it finds a maximum matching, and because it takes each edge exactly when the set
    taken stays matchable, it also uses the shortest possible edges and so minimizes total
    trim (Edmonds, 1971).

    Args:
        members: Members to build.
        edges: Available inventory.
        max_trim: Largest length that may be cut off one edge.
        tolerance: Allowance for floating-point error in length comparisons.

    Returns:
        One assignment per member, in the order of ``members``, or None if some member
        cannot be built.
    """
    members_by_length = sorted(range(len(members)), key=lambda index: members[index].required_length)
    open_members: list[tuple[float, int]] = []  # (latest usable edge length, member index)
    chosen_edge: dict[int, Edge] = {}
    next_member = 0
    for edge in sorted(edges, key=lambda candidate: candidate.length):
        while (
            next_member < len(members_by_length)
            and members[members_by_length[next_member]].required_length <= edge.length + tolerance
        ):
            member_index = members_by_length[next_member]
            heapq.heappush(open_members, (members[member_index].required_length + max_trim, member_index))
            next_member += 1
        if open_members and open_members[0][0] < edge.length - tolerance:
            return None  # this member's window closed before any edge was left for it
        if open_members:
            _, member_index = heapq.heappop(open_members)
            chosen_edge[member_index] = edge
    if len(chosen_edge) != len(members):
        return None
    return [MemberAssignment(members[index], chosen_edge[index]) for index in range(len(members))]
