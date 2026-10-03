# truss_bridge

## Purpose

Designs a rigid box-truss bridge of a given length, width and height from an inventory of
indexed parts, with every joint at a multiple of 45°. It verifies the design, load-tests it
as a space truss using each part's own material and section (steel and plastic parts can be
mixed), sizes its members for the lightest sections that pass every load case, and draws
elevation, plan and section views.

Let's be fair. This is all just to impress a really special gear head with my mathematical
code monkey skills.

## Requirements

Python 3.10 or later with the packages in `requirements.txt`: numpy, scipy (1.9 or later,
for `scipy.optimize.milp`) and matplotlib, plus flake8 for `make lint`.

```bash
make deps                        # create ./.venv/ and install requirements.txt into it
source .venv/bin/activate        # optional: make deps targets use .venv/ automatically
```

`make deps` creates the environment with `python3`; pick another interpreter with
`make deps BOOTSTRAP_PYTHON=python3.12`. There is no install step for the package itself:
run from the project root, or put the zip from `make build` on `PYTHONPATH`.

## Command line

```bash
python -m truss_bridge 12 3 3 --inventory example.csv --load-test     # each member's own part
python -m truss_bridge 12 3 3 --load-test --deflection-ratio 800        # generated example parts
python -m truss_bridge 12 3 3 --load-test --section-width 0.12 --section-wall 0.006  # override sections
python -m truss_bridge 12 3 3 --size-members --deflection-ratio 2000    # lightest section per role
python -m truss_bridge 16 2 4 --size-members --group-by member          # lightest section per member
python -m truss_bridge 12 3 3 --output-dir drawings                     # PNGs somewhere else
python -m truss_bridge --help                                           # every option
```

Without `--inventory`, an example inventory of steel and plastic parts is generated to suit
the dimensions. The exit status is 0 on success, 1 if no design is found or the inventory
cannot be read or load-tested, 2 if verification or a load test fails, and 3 if sizing fails.

Drawings are written to `output/` (created if missing): `bridge_<L>x<W>x<H>.png` on every
run, plus `_forces.png` and `_utilization.png` with `--load-test` or `--size-members`.
`--output-dir` changes the folder, and `--plot PATH` moves only the main drawing.

## Python API

```python
from truss_bridge import (basic_load_cases, design_bridge, plot_design, read_inventory,
                          run_load_tests, size_members, square_hollow_catalog, verify_design)

edges = read_inventory("example.csv")
design = design_bridge(edges, 12, 3, 3, max_trim=0.25)
assert not verify_design(design)
cases = basic_load_cases(design, deck_pressure=4e3, axle_force=50e3, lateral_pressure=1e3)
report = run_load_tests(design, cases)  # materials and sections from each member's part
print(report.summary())
plot_design(design, "forces.png", member_forces=report.governing_result.response.axial_forces)

sizing = size_members(design, cases, square_hollow_catalog(), group_by="role",
                      deflection_limit_ratio=2000)  # sections from the catalog, materials from the parts
print(sizing.summary())          # section, utilization and mass per group
sizing.report.passed             # True; sizing.report is a full LoadTestReport
```

`run_load_tests`, `analyze_load_case` and `size_members` also accept `sections=` and
`materials=`: one value for every member, or one per member, overriding the parts.
`write_inventory(path, edges)` and `example_inventory(length, width, height)` write and
generate inventories.

## Inventory CSV

```csv
index,length,material,elastic_modulus,yield_strength,density,strength_method,safety_factor,section_width,section_wall
0,3,steel,2e11,2.5e8,7850,aisc_360_16,1.67,0.1,0.005
2,3,gfrp,2.3e10,2e8,1900,yield_euler,3,0.15,0.01
```

Only `index` and `length` are required; that is enough to design a bridge. Load tests also
need the six material columns (filled in together) and the square hollow section's outside
width and wall (a wall of half the width is a solid square bar). Units are SI. Every row
naming a material must give it the same properties. Without a header row, columns are read
in the order above. `example.csv` is `example_inventory(12, 3, 3)` written by
`write_inventory`: every third part is example GFRP tube, the rest example steel tube.

| `strength_method` | Tension | Compression | Use for |
| --- | --- | --- | --- |
| `aisc_360_16` | `Fy·A` | AISC 360-16 E3 column curve | structural steel (safety factor 1.67 = ASD Ω) |
| `yield_euler` | `Fy·A` | `min(Fy, π²E/(L/r)²)·A` | plastics, FRP and other non-steel materials |

The safety factor divides both nominal strengths. For `yield_euler` it must also cover what
the formula leaves out: imperfections near `Fe ≈ Fy`, creep under sustained load, and
temperature and moisture effects. Typical values for plastics are 2.5–4. The slender-wall
screen `b/t > 1.40·√(E/Fy)` applies to both methods.

## Development

| Command | What it does |
| --- | --- |
| `make help` | list the targets (also the default) |
| `make deps` | create `./.venv/` and install `requirements.txt` into it; safe to rerun after editing it |
| `make clean` | remove `__pycache__/`, `dist/` and `output/` (keeps `.venv/`) |
| `make lint` | flake8, 120-column lines, E203 ignored |
| `make test` | all 86 tests, including the slow performance tests (about 30–60 s) |
| `make build` | clean, then zip `truss_bridge/`, `README.md`, `LICENSE.txt`, `requirements.txt` and `example.csv` into `dist/truss_bridge.zip`, leaving out caches, `output/` and `.venv/` |

`make` uses `.venv/bin/python` once `make deps` has run, and `python` before that; override
with `make test PYTHON=/path/to/python`. Git ignores `.venv/`, `dist/`, `output/` and caches
(see `.gitignore`).
To skip the performance tests (about 3 s), run
`TRUSS_BRIDGE_SKIP_PERFORMANCE=1 python -m unittest discover -s tests -t .`.

## Modules

| Module | Responsibility |
| --- | --- |
| `__init__.py` | public API |
| `__main__.py` | command line |
| `model.py` | `Edge`, `Material`, `CrossSection`, `Member`, `MemberAssignment`, `BridgeDesign`, errors |
| `inventory.py` | inventory CSV read/write, example inventory |
| `geometry.py` | exact integer lattice geometry, member roles |
| `lattice.py` | candidate node sets and members |
| `constraints.py` | joint-angle rules, conflicting member pairs |
| `rigidity.py` | rigidity matrix and mechanisms |
| `assignment.py` | least-trim edge matching (Glover's algorithm) |
| `linear_rows.py` | sparse constraint rows for the MILP |
| `exact_search.py` | MILP with lazy rigidity cuts: proven least trim |
| `constructive_search.py` | polynomial search for dense node sets and long spans |
| `design_builder.py` | chosen candidate members to a `BridgeDesign` |
| `solver.py` | `design_bridge` entry point |
| `verification.py` | independent validity checks |
| `analysis.py` | direct stiffness analysis, load cases, member checks per material |
| `sizing.py` | lightest passing section per role or per member, with deflection limits |
| `plotting.py` | 2D orthographic drawings, colored by role, member force or utilization |

Tests are in `tests/`, one `test_<area>.py` per area, with shared fixtures in
`tests/support.py`.

## Scope and limitations

Units: analysis is SI (m, N, Pa).

The design search minimizes trim among minimally rigid layouts and does not consider load
paths. Faces-lattice designs (width ≠ height) carry up to about 2.8 times the ideal truss
chord force and deflect 4 to 14 times more, so they size heavier than a well-braced layout
would.

Parts are matched to members by length alone, to minimize trim. Material and section play
no part in that choice, so a weak part can land in a heavily loaded member. The load test
then reports it, but the search does not avoid it.

## License

MIT; see [LICENSE.txt](LICENSE.txt).

---

(c) 2026 Sam Caldwell <mail@samcaldwell.net>
