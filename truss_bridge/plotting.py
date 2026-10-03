"""truss_bridge/plotting.py

Orthographic 2D drawings of a design: elevation, plan and cross-section.

Each view shows only the members parallel to its drawing plane, as in an engineering
drawing. Members in the reference plane (near side y = min, deck z = min, end x = min)
are drawn solid and bold; those in the opposite plane dashed; those in between thin.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import numpy as np

from .model import BridgeDesign

ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "chords": ("bottom_chord", "top_chord"),
    "verticals and posts": ("vertical", "interior_post"),
    "side diagonals": ("side_diagonal", "interior_diagonal"),
    "floor beams and struts": ("floor_beam", "top_strut", "cross_strut"),
    "lateral bracing": ("bottom_lateral", "top_lateral", "lateral"),
    "sway bracing": ("sway_brace",),
    "stringers": ("stringer",),
}
"""Member roles grouped for coloring; at most eight groups so each gets a distinct hue."""

GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
"""Categorical colors, assigned to ROLE_GROUPS in order (validated for color-vision deficiency)."""

COMPRESSION_COLOR, ZERO_FORCE_COLOR, TENSION_COLOR = "#2a78d6", "#b4b2ad", "#e34948"
"""Diverging colors for member forces: blue compression, gray zero, red tension."""

UTILIZATION_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
"""Sequential blue ramp for utilization 0 to 1 (light = lightly used, dark = near capacity)."""

OVER_CAPACITY_COLOR = "#d03b3b"
"""Reserved critical-status color for members over capacity; always paired with a legend label."""

INK, MUTED_INK = "#1f1f1e", "#6b6a66"

_VIEWS = (
    ("Elevation (x–z)", 0, 2, "near side solid, far side dashed"),
    ("Plan (x–y)", 0, 1, "deck solid, top dashed"),
    ("Section (y–z)", 1, 2, "end frame solid, far end dashed"),
)


def _group_of(role: str) -> str:
    for group, roles in ROLE_GROUPS.items():
        if role in roles:
            return group
    return "other"


def plot_design(
    design: BridgeDesign,
    path: str = "bridge.png",
    *,
    member_forces: np.ndarray | None = None,
    member_utilization: np.ndarray | None = None,
    member_sections: list | None = None,
    title: str | None = None,
) -> str:
    """Draw elevation, plan and cross-section views of a design to a PNG file.

    Args:
        design: The bridge to draw.
        path: Output file.
        member_forces: Optional axial force per member (tension positive, in N). When
            given, members are colored by force instead of role, with a color bar.
        member_utilization: Optional utilization per member (demand over available
            strength). When given, members are colored on a sequential scale from 0 to 1,
            and members above 1 are drawn in the critical color with a legend label.
        member_sections: Optional cross-section per member (objects with an ``area``); line
            widths then grow with section size.
        title: Figure title; a summary of the design by default.

    Returns:
        ``path``.

    Raises:
        ValueError: If both ``member_forces`` and ``member_utilization`` are given.
    """
    if member_forces is not None and member_utilization is not None:
        raise ValueError("pass member_forces or member_utilization, not both")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D

    coordinates = np.asarray(design.nodes, float)
    span, width, height = coordinates.max(axis=0) - coordinates.min(axis=0)
    members = design.members

    if member_forces is not None:
        force_kn = np.asarray(member_forces, float) / 1e3
        limit = max(float(np.abs(force_kn).max(initial=0.0)), 1e-9)
        color_map = LinearSegmentedColormap.from_list(
            "compression_tension", [COMPRESSION_COLOR, ZERO_FORCE_COLOR, TENSION_COLOR]
        )
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        member_colors = [color_map(norm(value)) for value in force_kn]
    elif member_utilization is not None:
        utilization = np.asarray(member_utilization, float)
        color_map = LinearSegmentedColormap.from_list("utilization", list(UTILIZATION_RAMP))
        norm = Normalize(vmin=0.0, vmax=1.0)
        member_colors = [OVER_CAPACITY_COLOR if value > 1.0 else color_map(norm(value)) for value in utilization]
        over_capacity = int((utilization > 1.0).sum())
    else:
        groups_present = [group for group in ROLE_GROUPS if any(_group_of(m.role) == group for m in members)]
        color_of_group = {group: GROUP_COLORS[position] for position, group in enumerate(ROLE_GROUPS)}
        member_colors = [color_of_group.get(_group_of(member.role), MUTED_INK) for member in members]

    width_scale = np.ones(len(members))
    if member_sections is not None:
        root_area = np.sqrt([section.area for section in member_sections])
        spread = root_area.max() - root_area.min()
        width_scale = 0.6 + 1.6 * ((root_area - root_area.min()) / spread if spread > 0 else 0.5)

    figure = plt.figure(figsize=(14, 7.8))
    grid = figure.add_gridspec(2, 2, width_ratios=[max(span, 1e-9), max(width, height, 1e-9) * 1.8])
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[:, 1])]
    for axis, (name, horizontal, vertical, legend_note) in zip(axes, _VIEWS, strict=True):
        depth_axis = 3 - horizontal - vertical
        near, far = coordinates[:, depth_axis].min(), coordinates[:, depth_axis].max()
        drawn = []
        for member, color, scale in zip(members, member_colors, width_scale, strict=True):
            start, end = coordinates[member.start_node], coordinates[member.end_node]
            if abs(start[depth_axis] - end[depth_axis]) > 1e-9:
                continue  # perpendicular to this view: seen end-on
            depth = start[depth_axis]
            if np.isclose(depth, near):
                layer, style = 0, dict(ls="-", lw=2.4, alpha=1.0)
            elif np.isclose(depth, far):
                layer, style = 2, dict(ls=(0, (4, 3)), lw=1.3, alpha=0.75)
            else:
                layer, style = 1, dict(ls="-", lw=1.0, alpha=0.55)
            style["lw"] *= scale
            drawn.append((layer, start, end, color, style))
        for _, start, end, color, style in sorted(drawn, key=lambda item: -item[0]):
            axis.plot(
                [start[horizontal], end[horizontal]],
                [start[vertical], end[vertical]],
                color=color,
                solid_capstyle="round",
                **style,
            )
        in_near_plane = np.isclose(coordinates[:, depth_axis], near)
        axis.scatter(
            coordinates[in_near_plane, horizontal],
            coordinates[in_near_plane, vertical],
            s=14,
            c=INK,
            zorder=3,
            linewidths=0,
        )
        axis.set_title(f"{name}: {legend_note}", fontsize=9, color=INK)
        axis.set_aspect("equal")
        axis.set_xlabel(f"{'xyz'[horizontal]} (m)", color=MUTED_INK, fontsize=8)
        axis.set_ylabel(f"{'xyz'[vertical]} (m)", color=MUTED_INK, fontsize=8)
        axis.tick_params(colors=MUTED_INK, labelsize=8)
        axis.grid(alpha=0.2)
        for spine in axis.spines.values():
            spine.set_color("#d6d4cf")

    if member_forces is not None or member_utilization is not None:
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
        bar = figure.colorbar(scalar, ax=axes, orientation="horizontal", fraction=0.035, pad=0.09, aspect=60)
        label = (
            "axial force (kN): compression < 0 < tension"
            if member_forces is not None
            else "utilization (demand / available strength)"
        )
        if member_sections is not None:
            label += "; line width grows with section size"
        bar.set_label(label, color=INK, fontsize=9)
        bar.ax.tick_params(labelsize=8, colors=MUTED_INK)
        if member_utilization is not None and over_capacity:
            figure.legend(
                handles=[Line2D([], [], color=OVER_CAPACITY_COLOR, lw=2.6, label=f"over capacity ({over_capacity})")],
                loc="upper right",
                fontsize=8,
                frameon=False,
                labelcolor=INK,
            )
    else:
        counts = {group: sum(_group_of(m.role) == group for m in members) for group in groups_present}
        handles = [
            Line2D([], [], color=color_of_group[group], lw=2.6, label=f"{group} ({counts[group]})")
            for group in groups_present
        ]
        figure.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False, labelcolor=INK)
    figure.suptitle(
        title
        or f"{design.node_set} node set, {len(design.nodes)} nodes, {len(members)} members, "
        f"total trim {design.total_trim:.3f} m",
        fontsize=11,
        color=INK,
    )
    if member_forces is None and member_utilization is None:
        figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(figure)
    return path
