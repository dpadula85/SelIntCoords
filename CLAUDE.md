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
optional flexible side chains (examples: BTBT, DNBDT, pentacene; `examples/` has the inputs, `tests/` the pytest suite, the OPLS-AA files for perception ship in `SelIntCoords/data/top`).

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
  bonds/angles/proper(stiff+flexible)/improper dihedrals/the LJ pairs that bump (see `pair_selection.py`)/
  exclusions, takes the equivalent atoms from the symmetry of the bond graph (`symmetry.graph_equivalences`:
  orbits of the automorphisms that keep element and atom type, individualisation-refinement, no group listing, no
  geometry tolerance; on 8 example molecules it loses no dependency of the former point-group + local-hydrogen
  rule and adds some on ZR1, DPP, rPSB11 where the geometry is not symmetric; his hand ties of the 11-cis retinal
  PSB are reproduced exactly; `symmetry.equivalent_atoms`, the point group within 0.01 A, stays as a library function
  that make_top no longer calls) and ring symmetry equivalence (`dihed_deps.py`). Writes: the completed `.top`, a
  `<stem>_deps.dat` (Joyce `$dependence 1.2` format), and a
  `<stem>.csv` coordinate-count summary (columns: `Type, Number, Start,
  End`).
- `make_top -m geometry.xyz` (no `-p`; writes `<stem>.top`, class-named atomtypes like CA1/HC, atoms named element+number, the opls type in a comment; `--ff-type-names` keeps opls_NNN) perceives the atom types inside
  (`perceive_to_top` into a temp dir, deleted) and keeps only the report
  `<out stem>.types` next to the outputs; with `-p` nothing is perceived.
  `perceive_types -m geometry.xyz [-o out.top] [--ff oplsaa] [--gmx-top DIR] [--ff-type-names]`
  is the same perception alone: force-field atom types from a bare geometry,
  written as the minimal `.top` plus a `<stem>_types.types` report. Types, masses and LJ parameters
  come from the OPLS-AA files shipped in `SelIntCoords/data/top/oplsaa.ff` (4 files copied
  unchanged from GROMACS 2020, LGPL-2.1, references in its README; `ffdb.py`), or for
  another force field from a GROMACS `<ff>.ff` (`--gmx-top`, `$GMXLIB`, `$GMXDATA/top`,
  or the `gmx` executable). Atoms are
  classified by `mol_features.py` (bond graph + geometry only: hybridisation
  from angle sums, aromatic = planar 5/6-ring with Hueckel count and no
  carbonyl C; no bond orders, no RDKit: generic bond-order assignment
  mistypes thiophene S in DPP/PNDI2OD). Engine 1: curated OPLS-AA rules
  (`oplsaa_rules.py`, ordered, first match wins; `analogue=True` rules borrow
  thiazole S opls_633 / benzene C opls_145 / imidazole N opls_557 /
  pyrrole C2,C3 opls_543,544 and azo N (opls_520, pyridine N) as the examples do, and four conjugated-carbon
  rules typing ring-ketone C between two conjugated C, alkene C without H with
  two conjugated C neighbours, pyramidalised cage sp2 C, and quaternary sp3 C
  between two conjugated C as opls_145, as in PM6/IDIC/Y6/PCBM; `REFUSE` lists
  charged environments). Engine 2: `learned_types.py`, Morgan-label table (radius
  <= 3) from the `.rtp` residues, back-off to smaller radius, radius 0 never
  used; fallback for uncovered atoms and cross-check (LJ-class disagreement
  at radius >= 2 is reported). Untyped atom = `ValueError`. Charges are 0.
  Validation: LJ+mass equal to the hand-assigned types for 360/362 example
  atoms (the 2 misses: PBDTTT-O furan O, hand epsilon 0.5876 vs opls_566
  0.58576), 11070/11074 OPLS-valued atoms of QMD-FFs (74 molecule types; misses:
  NAP fusion C, ETD hydroxyl, hand-typed with other values; refitted-LJ
  molecules are not comparable), 726/726 of the walter BHJ polymers; perceived
  types give identical bonded terms for BTBT, DPP, PNDI2OD; learned engine
  leave-residue-out same-LJ-class 93% (OPLS-AA). Rhodanine rule (ring C all
  opls_145, ring S and thione S opls_633, N opls_239). `extra_types.py`: Se and
  Si from the literature (his PBDTTT-Se / TIPS_PN notes, with DOIs); any other
  element without parameters gets a placeholder atomtype whose sigma/epsilon
  are `FILL_ME` in the final topology (`mark_missing_parameters`), plus
  `<stem>_types.top` to fill and re-run with `-p`.
  Other force fields: learned engine only, unvalidated; neutral molecules only.
- `make_top` output organisation, as in his hand-made topologies (QMD-FFs
  `ZR1.itp`, `Y6.itp`): `ring_groups.py` groups bonds, angles and stiff
  dihedrals by fused ring system under `; <ring name>` headers (headers are
  inserted into the written text by the trailing `; <global index>`
  comments; the arrays are reordered before the dependencies are computed,
  so `_deps.dat` follows the order of the file). Link chains between two
  systems form `A to B`; methyl involvement gives `methyls`; impropers are
  `starlike`. `flex_classes.py` sorts rotors: methyl (on C/N/O/S/Se) and sp3 ring-bond
  dihedrals -> funct 1 `k = 0.651, n = 3, delta = 0` (= GAFF X-c3-c3-X 0.1556
  kcal/mol, Wang 2004 doi:10.1002/jcc.20035; his retinal paper takes ring
  torsions from GAFF; ZR1/Y6/ISOTIPS_PN/retinal files carry it; older D18/PBDTTT-O
  files have Joyce-fitted 4.128/5.381, not used); on an sp3 C all neighbours
  of a methyl are kept, on sp2/N/O/S one heavy neighbour. Single bond between
  two rigid units (rings, sp2, azo N) directly or through sp3 linker atoms (ether, ethylene) -> `delta<N>`; double bond with no ring atom -> `phi<N>`; a side chain whose far end has no ring or sp2 atom stays unclassified
  (retinal chain), each one dihedral (sp2 carbons preferred), `n = 2`
  placeholder as in his Step1 files, header shared by atoms of the same
  connectivity class (`symmetry.topological_classes`). Double bonds =
  shortest candidate at each atom below 0.92 x covalent radii sum. Double bond
  with an atom in a ring (exocyclic, ZR1/Y6) -> all its dihedrals join the stiff
  ones; others keep null parameters under `flexible (merge or by hand)`.
  `make_top.find_deps` groups coordinates by the class of each atom when the
  equivalences partition the atoms (same result as substituting every atom by
  its equivalents, which was 60**4 per dihedral for C60; falls back to that
  loop otherwise). Checked against ZR1 (bonds, angles identical to his Step1; the four `delta`
  dihedrals and the exocyclic C=C stiff ones equal his) and the 11-cis retinal PSB
  (his delta1-5 and phi1-5 quadruplets, tests/data).
- `make_top` pair selection (`pair_selection.py`, `--pair-ratio` 0.8, `--pair-fraction` 0.01): rigid-rotor
  conformers (rotors = bridges that are not double bonds and end on no terminal atom, smaller side turns;
  400 Metropolis chains stepped together as NumPy arrays, 20000 conformers per round). A 1,5+ pair joins the
  set when it is below 0.8 sigma_ij (mixing-rule sigma) in at least 1 % of the conformers allowed by the set
  so far; any bump between two ring systems adds all their heavy pairs; repeat until stable (empty start,
  2-3 rounds); pairs in one orbit of the bond-graph automorphisms (atom colour = element + atomtype name,
  compared never read; individualisation-refinement, no group listing) are written together. Written:
  heavy-heavy and H...H only; H-heavy pairs are hard spheres but not written; 1,4 never. Everything else is
  commented after the written pairs with its closest approach / sigma. Without `pair_selection` `add_terms`
  writes every pair. Compared with joyce_autoval on trans (azo dye, `tests/data/DAB.xyz`): 207 pairs, 171 of
  autoval's 201 heavy/H...H (85 %); misses one ring pair seen only in MM scans (bending). Retinals: 36-63
  pairs against ~200 in his hand lists, which were chosen by scan relevance (not a reference). Cost
  ~ atoms x candidate pairs (trans 8 s, ZR1 15 s, 254-atom oligomer ~5 min with ~2500 pairs from ring blocks: meant for monomers and
  cores). Not validated beyond that; other force fields and charged species untested.
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
  here that uses it.
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

`make_top`: a minimal `.top` (atom types/atoms block only; optional, see
`perceive_types`) + `.xyz` geometry, atom order and count must match exactly.
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
- **Fixed 2026-08-06:** `list_intcoords()`'s docstring `Returns` section
  omitted `eq` (the equivalence dict), which is in fact the 9th/last
  returned value — added.

## Dependencies

MDAnalysis 2.7.0, networkx 3.2.1, numpy 1.26.4, pandas 2.2.1, scipy
(`requirements.txt`). No psi4, no pyscf: the symmetry-equivalent atoms come
from `symmetry.py` (numpy + scipy KD-tree).
`perceive_types` needs a GROMACS `top` database on disk (no new Python
dependency; `rdkit` only in the tests extra, for the SMILES-built cases).
`openmm` is required only by `openmm_compat.py`, documented here rather
than declared in `requirements.txt`.
`requirements.yml` is a full conda-env snapshot, much broader than the
actual runtime dependency set — don't treat it as the minimal spec.

`blocks.py` and `top.py` are vendored from
[GromacsWrapper](https://github.com/Becksteinlab/GromacsWrapper)
(GPLv3) — the same files (verbatim) also live in `oligomer_builder`.
Not Daniele's own design; leave alone unless he explicitly asks to
fork/modify them.
