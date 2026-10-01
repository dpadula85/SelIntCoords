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
- `joyce_to_openmm -p topology.top -o output_dir/ [-v]` — pure-text
  converter (no OpenMM/GROMACS dependency, unlike everything else in
  this pipeline) that rewrites a Joyce-generated `.top` (+ every file it
  `#include`s) into a copy readable *identically* by both real GROMACS
  and OpenMM: funct-2 `[ pairs ]` lines rewritten to funct-1, duplicate
  `[ pairs ]` lines for the same atom pair commented out (not deleted),
  `[ defaults ]`'s global `fudgeQQ` forced to 0 (raises
  `ChargeOverrideError` if any kept pair's own `fudgeQQ*qi*qj` is
  nonzero, since that value cannot survive the funct-1 rewrite). See
  `joyce_to_openmm.py`'s module docstring for the three bugs this fixes
  and how they were found; verified against a real GROMACS oracle in
  `oligomer_builder` (`HANDOFF.md`, "OpenMM as a future GROMACS
  alternative"), which also carries the OpenMM-`System`-internal fixes
  (comb-rule units, dihedral mult=0, spurious auto-1-4s, dispersion
  correction) that are out of scope for this pure `.top`-in/`.top`-out
  tool. **Converting the file makes both programs able to read it; it
  does not make them compute the same energy** — see `openmm_compat`
  below for the piece that closes that gap.
  `#include`s are resolved relative to each including file and the
  rewritten copies keep that directory layout (no basename flattening,
  so `../ffs/x.itp` and same-named `.itp`s in different directories
  work). A `[ pairs ]` line without explicit `V W` raises
  `UnparameterisedPairError` (a bare `ai aj 1` would take its values
  from `[ pairtypes ]`/`gen-pairs`, which the pair rebuild never reads,
  so the pair would silently vanish).
- `openmm_compat.load_top(top_path, gro_path, ...)` (no CLI, library
  only) — builds an OpenMM `(topology, system, dropped_dihedral_offset)`
  that actually agrees with real GROMACS, not just one OpenMM can build
  without error. Patches the four `createSystem()`-internal mistakes
  `joyce_to_openmm`'s file-level fix cannot touch (comb-rule-dependent
  `V`/`W` misread, a zero-multiplicity-dihedral crash, auto-generated
  1-4s at a merged junction, the dispersion-correction default) by
  reusing `joyce_to_openmm`'s own rewrite internals rather than
  duplicating them. A deliberate duplication of `oligomer_builder.
  openmm_compat` (where this fix originates and was verified against a
  real GROMACS oracle on 18 systems, ~20 to ~146 000 atoms) —
  `oligomer_builder` needs its own copy regardless, since the merged-
  junction bug this fixes can only arise after `oligomer_builder` merges
  two fragments this package never sees combined. Needs `openmm`
  (`pip install openmm`), not in `requirements.txt` — the only module
  here that uses it, documented the same way `psi4` already is.
  `load_top(top, gro=None, nonbonded_method=app.NoCutoff)` is the
  vacuum path (GROMACS `pbc = no`, infinite cut-offs): no `.gro`/box
  needed. The dispersion flag is applied to every `NonbondedForce` and
  `CustomNonbondedForce` after `createSystem` (OpenMM 8.2 rejects the
  keyword; 8.6 leaves the comb-rule-3 `CustomNonbondedForce` correction
  on by default). Checked against `gmx_d -rerun` (2020.5) on 33 `.top`
  files in `QMD-FFs` (identical `.gro` coordinates: <= 2e-3 kJ/mol,
  except the PM6 multimers at 0.01-0.05 kJ/mol, where a hand sum of the
  pair lines reproduces OpenMM's LJ-14, not GROMACS's) and against the
  per-term `Scan.dat` of three DTS and one Y6 relaxed scans (73 frames
  each, <= 1e-3 kJ/mol per term).

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

- **Fixed 2026-08-06:** `sel_intcoords.py`'s `get_sp2(u, alkyl=True,
  ether=False)` previously referenced an undefined name `oxy` in its
  `ether=True` branch (latent `NameError`; unreachable in practice since
  `list_intcoords()` always calls with defaults), and its `allcheck`
  variable was computed but never actually wired into the sp2/sp3
  classification in any branch (so `alkyl`/`ether` did nothing
  functionally different — `alkyl` mode only "worked" because sp3 atoms
  are separately excluded by the `unsat` test itself). Synced to
  match `oligomer_builder`'s cleaner copy (removed the dead `keep`-list
  code, replaced the crash with a safe branch), then fixed further: the
  `unsat`-derived candidate set is now explicitly reduced by `allcheck`
  (`np.setdiff1d`), so `ether=True` genuinely excludes ether oxygens
  (any atom of type `"O"`, matching `chain_cropper`'s convention) from
  `sp2`, while `alkyl`/default mode is provably unchanged (verified with
  a diethyl-ether test molecule: identical `sp2` for `alkyl=True` before
  and after, oxygen present in `sp2` before the fix and correctly absent
  under `ether=True` after it).
- **Superseded 2026-08-13:** the fix above was still incomplete — the
  function tested `if alkyl: ... elif ether: ...`, and `alkyl` defaults
  `True`, so the `ether` branch stayed unreachable unless a caller also
  passed `alkyl=False`, which `list_intcoords()` never did. `get_sp2` is
  no longer defined here: `sel_intcoords.py` now imports it from
  `chain_cropper.topology`, the single place this logic lives (also used
  by `oligomer_builder.enhanced_breaker`, which imports the same symbol).
  This package gained a runtime dependency on `chain-cropper`
  (`setup.py`). `alkyl`-mode behaviour is unchanged; `ether=True`
  callers now get correct results instead of the alkyl-mode ones.
- **Fixed 2026-08-06:** `make_top.py` carried a complete `old_main()`
  (~130 lines), fully superseded by `main()` (which adds ring-dihedral
  dependency handling) — deleted; the `make_top` console-script entry
  point already pointed at `main`, not `old_main`.
- **Investigated, deliberately left as-is:** `sel_intcoords.py`'s
  top-of-file `try: import psi4, pyscf... except: import (same) ...
  finally: import (same)` (lines 11-19) looks like a no-op (if the
  import fails in `try` it fails identically in `except`, and `finally`
  re-raises unconditionally anyway) — a plain single import reproduced
  identical behavior in repeated local testing (5 runs, same import
  order as the real file, psi4 1.9.1). **But Daniele confirmed this
  triple import is a deliberate workaround for a real psi4 quirk seen on
  some of his other machines, not dead code** — the quirk didn't
  reproduce in this environment, so don't "clean this up" again without
  actually reproducing the failure it guards against.
- **Fixed 2026-08-06:** `list_intcoords()`'s docstring `Returns` section
  omitted `eq` (the equivalence dict), which is in fact the 9th/last
  returned value — added.

## Dependencies

MDAnalysis 2.7.0, networkx 3.2.1, numpy 1.26.4, pandas 2.2.1, pyscf 2.5.0
(`requirements.txt`); `psi4` is also required at runtime for
`make_top`/`sel_intcoords`'s point-group detection (not needed by
`map_atoms`/`renumber_top`, which use only MDAnalysis/networkx/numpy).
`openmm` is required only by `openmm_compat.py`, documented here rather
than declared in `requirements.txt`, same treatment as `psi4`.
`requirements.yml` is a full conda-env snapshot, much broader than the
actual runtime dependency set — don't treat it as the minimal spec.

`blocks.py` and `top.py` are vendored from
[GromacsWrapper](https://github.com/Becksteinlab/GromacsWrapper)
(GPLv3) — the same files (verbatim) also live in `oligomer_builder`.
Not Daniele's own design; leave alone unless he explicitly asks to
fork/modify them.
