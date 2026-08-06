@../pipeline-docs/CLAUDE.md

# CLAUDE.md

## What this repo is

Selects and enumerates internal coordinates (bonds, angles, dihedrals,
1-n LJ pairs, exclusions) from a molecular geometry by connectivity +
point-group/local symmetry analysis, and writes a complete GROMACS `.top`
ready for parameterization by the external program **Joyce3**. Also
provides standalone atom-mapping/renumbering utilities to reconcile
different atom numbering schemes between two structures of the same
molecule (e.g. an experimental crystal geometry vs. a force-field
geometry). Targets rigid/(semi-)planar organic-semiconductor cores with
optional flexible side chains (tests: BTBT, DNBDT, pentacene).

## Role in the pipeline

Step 1 — QMD-FF parameterization. Upstream: an initial geometry/minimal
`.top` (atom types only) and the Joyce3 program itself, mostly run and
automated externally. Downstream: `oligomer_builder`, which consumes this
package's completed `.top` + matching geometry as a `Fragment` to build
oligomers/polymers.

## Public interface (console scripts, from `setup.py`)

- `make_top -p initial.top -m geometry.xyz [-l geom|avg] [-o output.top]`
  — the main Joyce3-prep tool. Reads a minimal `.top` (atom types/atoms,
  no bonded terms) + an `.xyz` with matching atom order; guesses bonds via
  MDAnalysis, builds a NetworkX connectivity graph, enumerates
  bonds/angles/proper(stiff+flexible)/improper dihedrals/LJ 1,4-1,7+pairs/
  exclusions, detects point group (psi4) and local/ring symmetry
  equivalence (`dihed_deps.py`). Writes: the completed `.top`, a
  `<stem>_deps.dat` (Joyce `$dependence 1.2` format), and a
  `<stem>.csv` coordinate-count summary (columns: `Type, Number, Start,
  End`).
- `map_atoms -r reference.xyz -t target.xyz [-m map.txt] [-o reordered.xyz] [-v]`
  — graph-isomorphism + Kabsch-RMSD atom correspondence between two XYZ
  structures of the same molecule with different atom orderings.
  `AtomMapper(reference, target).run()` returns an `AtomMapping(mapping,
  rmsd, n_isomorphisms)`; `.write()` produces a two-column `reference
  target` map file (1-based by default).
- `renumber_top -p topology.top -m map.txt [-n 0] [-o renumbered.top] [--allow-partial] [-v]`
  — applies a `reference target` map (from `map_atoms`, or hand-written)
  to renumber a `.top`'s atoms in place; bonded terms stay consistent
  automatically since they hold live `Atom` object references, not
  indices (see `renumber.py`'s module docstring). Strict bijection
  validation by default; `--allow-partial` relaxes it.

## Input format

`make_top`: a minimal `.top` (atom types/atoms block only) + `.xyz`
geometry, atom order and count must match exactly.
`map_atoms`/`renumber_top`: plain `.xyz` files; the map file is
whitespace `reference target` integer pairs, one per line, `#`-comments
and blank lines skipped.

## Output format

`make_top`'s completed `.top` + `_deps.dat` (Joyce dependency syntax:
`%5d = %5d*1.d0 ;` lines under `$dependence 1.2` / `$end`) is exactly
what Joyce3 consumes externally. This `.top` (paired with the geometry it
was built from) is also exactly the input `oligomer_builder` expects for
a `Fragment`.

## Known gaps / TODOs

- `sel_intcoords.py`'s `get_sp2(u, alkyl=True, ether=False)`: the
  `ether=True` branch references an undefined name `oxy` (line ~101) —
  latent `NameError` if ever called with `ether=True` (nothing in this
  repo currently does; `list_intcoords()` always calls with defaults).
  The `keep` list in the same function is built but never populated
  (only `delete` is appended to) — the function still returns correct
  results because the `unsat` mask already accounts for this, but the
  dead `keep`-handling code (lines ~106-129) should be removed or
  clarified rather than left looking load-bearing.
- `make_top.py` still carries a complete `old_main()` (~130 lines),
  fully superseded by `main()` (which adds ring-dihedral dependency
  handling) — safe to delete, not called from anywhere including the
  `make_top` console-script entry point (which points at `main`).
- `sel_intcoords.py`'s top-of-file `try: import psi4, pyscf... except:
  import (same) ... finally: import (same)` (lines 11-19) provides no
  actual error handling — if the import fails in `try`, it fails
  identically in `except`, and `finally` then re-raises unconditionally
  anyway. Either drop the try/except/finally (it's equivalent to a plain
  import today) or convert to the `HAS_X`-guarded pattern used elsewhere
  in Daniele's newer code if psi4 is meant to be optional.
- `list_intcoords()`'s docstring `Returns` section omits `eq` (the
  equivalence dict), which is in fact the 9th/last returned value —
  minor doc/code mismatch.

## Dependencies

MDAnalysis 2.7.0, networkx 3.2.1, numpy 1.26.4, pandas 2.2.1, pyscf 2.5.0
(`requirements.txt`); `psi4` is also required at runtime for
`make_top`/`sel_intcoords`'s point-group detection (not needed by
`map_atoms`/`renumber_top`, which use only MDAnalysis/networkx/numpy).
`requirements.yml` is a full conda-env snapshot, much broader than the
actual runtime dependency set — don't treat it as the minimal spec.

`blocks.py` and `top.py` are vendored from
[GromacsWrapper](https://github.com/Becksteinlab/GromacsWrapper)
(GPLv3) — the same files (verbatim) also live in `oligomer_builder`.
Not Daniele's own design; leave alone unless he explicitly asks to
fork/modify them.
