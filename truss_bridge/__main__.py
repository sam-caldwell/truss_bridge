"""truss_bridge/__main__.py

Command-line entry point: ``python -m truss_bridge LENGTH WIDTH HEIGHT [options]``.

Designs a bridge, verifies it, draws it, and optionally load-tests it or sizes its members
for the lightest sections that pass every load case. Load tests use the material and
section of each member's inventory part. Without ``--inventory`` an example inventory of
steel and plastic parts is generated that suits the given dimensions.

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import (
    CrossSection,
    NoDesignFoundError,
    SizingError,
    basic_load_cases,
    design_bridge,
    example_inventory,
    plot_design,
    read_inventory,
    run_load_tests,
    size_members,
    verify_design,
)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="python -m truss_bridge", description=__doc__)
    parser.add_argument("length", type=float, help="span along x, in m")
    parser.add_argument("width", type=float, help="width along y, in m")
    parser.add_argument("height", type=float, help="height along z, in m")
    parser.add_argument(
        "--inventory",
        help="CSV of parts: index,length and optionally material and section columns (see "
        "truss_bridge/inventory.py); default: generated example",
    )
    parser.add_argument("--max-trim", type=float, default=0.25, help="largest cut per edge, in m")
    parser.add_argument("--angle-rule", choices=["planar", "pairwise"], default="planar")
    parser.add_argument("--refinement", type=int, default=1, help="finer lattice spacings to try")
    parser.add_argument("--output-dir", default="output", help="folder for the PNG drawings (created if missing)")
    parser.add_argument("--plot", default=None, help="PNG path for the main drawing; default: in --output-dir")
    parser.add_argument("--list-members", action="store_true", help="print every member and its edge")
    load = parser.add_argument_group("load test (SI units; example values, not code requirements)")
    load.add_argument("--load-test", action="store_true", help="run structural load tests")
    load.add_argument("--section-width", type=float, help="override every part's section: square hollow width, m")
    load.add_argument("--section-wall", type=float, help="override every part's section: wall thickness, m")
    load.add_argument("--deck-pressure", type=float, default=4.0e3, help="uniform deck load, Pa")
    load.add_argument("--axle-force", type=float, default=50e3, help="moving axle force, N")
    load.add_argument("--lateral-pressure", type=float, default=1.0e3, help="wind pressure on one side, Pa")
    load.add_argument("--deflection-ratio", type=float, default=None, help="limit deflection to span / ratio")
    load.add_argument(
        "--size-members",
        action="store_true",
        help="choose the lightest passing section per group (overrides --section-*)",
    )
    load.add_argument("--group-by", choices=["role", "member"], default="role", help="sizing groups")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns the process exit status."""
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    if (arguments.section_width is None) != (arguments.section_wall is None):
        print("--section-width and --section-wall must be given together")
        return 1
    dimensions = (arguments.length, arguments.width, arguments.height)
    try:
        edges = read_inventory(arguments.inventory) if arguments.inventory else example_inventory(*dimensions)
    except (OSError, ValueError) as error:
        print(f"cannot read inventory: {error}")
        return 1

    started = time.perf_counter()
    try:
        design = design_bridge(
            edges,
            *dimensions,
            max_trim=arguments.max_trim,
            angle_rule=arguments.angle_rule,
            refinement=arguments.refinement,
        )
    except NoDesignFoundError as error:
        print(f"no design: {error}")
        return 1
    print(design.summary(include_assignments=arguments.list_members))
    violations = verify_design(design, angle_rule=arguments.angle_rule)
    print(f"designed in {time.perf_counter() - started:.2f} s; violations: {violations or 'none'}")

    os.makedirs(arguments.output_dir, exist_ok=True)
    stem = os.path.join(arguments.output_dir, f"bridge_{arguments.length:g}x{arguments.width:g}x{arguments.height:g}")
    if arguments.plot and os.path.dirname(arguments.plot):
        os.makedirs(os.path.dirname(arguments.plot), exist_ok=True)
    print("drawing:", plot_design(design, arguments.plot or f"{stem}.png"))

    if not (arguments.load_test or arguments.size_members):
        return 0 if not violations else 2
    cases = basic_load_cases(
        design,
        deck_pressure=arguments.deck_pressure,
        axle_force=arguments.axle_force,
        lateral_pressure=arguments.lateral_pressure,
    )
    sized = arguments.size_members
    try:
        if sized:
            sizing = size_members(
                design,
                cases,
                group_by=arguments.group_by,
                deflection_limit_ratio=arguments.deflection_ratio,
            )
            print(sizing.summary())
            report = sizing.report
        else:
            override = (
                CrossSection.square_hollow(arguments.section_width, arguments.section_wall)
                if arguments.section_width is not None
                else None
            )
            report = run_load_tests(
                design, cases, sections=override, deflection_limit_ratio=arguments.deflection_ratio
            )
    except SizingError as error:
        print(f"sizing failed: {error}")
        return 3
    except ValueError as error:
        print(f"cannot load-test: {error}")
        return 1
    print(report.summary())

    governing = report.governing_result
    widths = report.sections if len({section.name for section in report.sections}) > 1 else None
    print(
        "force drawing:",
        plot_design(
            design,
            f"{stem}_forces.png",
            member_forces=governing.response.axial_forces,
            member_sections=widths,
            title=f"Member forces, governing case: {governing.load_case.name}",
        ),
    )
    worst_utilization = np.max(
        [[check.utilization for check in result.member_checks] for result in report.results], axis=0
    )
    print(
        "utilization drawing:",
        plot_design(
            design,
            f"{stem}_utilization.png",
            member_utilization=worst_utilization,
            member_sections=widths,
            title=f"Worst utilization over {len(report.results)} load cases, member mass {report.total_mass:,.0f} kg",
        ),
    )
    return 0 if report.passed and not violations else 2


if __name__ == "__main__":
    sys.exit(main())
