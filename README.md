<div align="center">

# SelIntCoords

**Select and group internal coordinates from a molecular geometry for QMD-FF parameterization**

[![version](https://img.shields.io/badge/version-1.1-blue)](setup.py)
[![python](https://img.shields.io/badge/python-3-blue)](setup.py)
[![license](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

</div>

---

A Python package to automatically select and group internal coordinates
(bonds, angles, dihedrals) from a molecular geometry, and to produce a
GROMACS topology ready to be used with the
[Joyce](https://www.dsf.unica.it/~fabio/Joyce.php) program to parameterise
a force field. It also includes tools to map and renumber atoms so that a
force-field topology follows an arbitrary external numbering, e.g. an
experimental crystal structure.

The package targets rigid, (semi-)planar organic semiconductor cores with
optional alkyl/ether side chains — the included tests use
BTBT-, DNBDT- and pentacene(PN)-type molecules — but the underlying
graph-based logic is general.

## Contents

- [Installation](#installation)
- [Workflow overview](#workflow-overview)
- [`make_top` — build a topology with grouped internal coordinates](#make_top--build-a-topology-with-grouped-internal-coordinates)
- [`map_atoms` — map atoms between two structures](#map_atoms--map-atoms-between-two-structures)
- [`renumber_top` — renumber a topology's atoms](#renumber_top--renumber-a-topologys-atoms)
- [`joyce_to_openmm` — make a topology dual-valid for GROMACS and OpenMM](#joyce_to_openmm--make-a-topology-dual-valid-for-gromacs-and-openmm)
- [`openmm_compat` — build a correct OpenMM `System`, not just a parseable one](#openmm_compat--build-a-correct-openmm-system-not-just-a-parseable-one)
- [Python API](#python-api)
- [Package layout](#package-layout)
- [Tests](#tests)

## Installation

```bash
git clone https://github.com/dpadula85/SelIntCoords.git
cd SelIntCoords
pip install -e .
```

This installs the package and four console scripts: `make_top`,
`map_atoms`, `renumber_top`, and `joyce_to_openmm`.

### Requirements

See `requirements.txt`. Notably:

- `MDAnalysis` — geometry/connectivity handling
- `networkx` — graph representation of molecular connectivity (paths,
  rings, isomorphism)
- `pyscf` and `psi4` — point-group detection and symmetry-equivalent atom
  identification (used by `make_top`/`sel_intcoords` only)
- `numpy`, `pandas`

`map_atoms` and `renumber_top` only need `MDAnalysis`, `networkx`, and
`numpy` — they do not require `pyscf`/`psi4`. `joyce_to_openmm` needs
none of the above at all: it is pure standard library (`re`, `pathlib`).
`openmm_compat` needs `openmm` (`pip install openmm`) — also not listed
in `requirements.txt`, the same way `psi4` is documented rather than
declared, since it's the only module in the package that uses it.

## Workflow overview

A typical use case looks like this:

1. Generate an initial, minimal `.top` file containing only
   `[ atomtypes ]` and `[ atoms ]` sections (atom names, types, charges,
   masses — no bonded terms) for your molecule, on some geometry.
2. Run **`make_top`** on that geometry to automatically derive bonds,
   angles, and dihedrals from connectivity, group together those that are
   equivalent by symmetry, and write a complete `.top` file (plus a
   Joyce-format dependency file and a CSV summary of coordinate counts).
3. If the geometry the force field was generated on uses a different atom
   numbering than the one you actually need (e.g. an experimental crystal
   structure), use **`map_atoms`** to find the atom correspondence between
   the two geometries, then **`renumber_top`** to renumber the topology
   accordingly — bonded terms are kept fully consistent automatically.

```
  crystal.xyz  ff_geometry.xyz
       \            |
        \       (1) make_top   -->  ff_geometry_symm.top, .csv, _deps.dat
         \           |
          \-- (2) map_atoms --> map.txt (+ optionally reordered.xyz)
                      |
              (3) renumber_top --> ff_geometry_symm.renumbered.top
                                    (atom numbers now follow crystal.xyz)
```

## `make_top` — build a topology with grouped internal coordinates

Given an initial topology (atom types/charges only) and a geometry file,
`make_top`:

- Guesses bonds from the geometry (MDAnalysis) and builds a connectivity
  graph (NetworkX).
- Enumerates bonds, angles, proper dihedrals (flexible and stiff),
  improper dihedrals, 1-4/1-5/1-6/1-7/other Lennard-Jones pair
  interactions, and pairwise exclusions, by walking the connectivity
  graph (`sel_intcoords.list_intcoords`).
- Detects the molecular point group (via `pyscf`/`psi4`) to identify
  atoms that are equivalent by symmetry, and separately identifies atoms
  equivalent by local connectivity (e.g. the three H's of a methyl group).
- Distinguishes **stiff** (ring / conjugated-core, near-planar) from
  **flexible** (side-chain rotatable) proper dihedrals, and detects rings
  (core atoms and core+substituents) to group ring dihedrals that should
  share a force constant (`dihed_deps.py`).
- Assigns default (zero) force-field parameters for every new bonded term
  and appends them to the topology.
- Writes:
  - the completed `.top` file,
  - a `_deps.dat` file listing coordinate dependencies in Joyce's
    `$dependence` format, so that symmetry-equivalent coordinates are
    fit together,
  - a `.csv` summary with the number of bonds/angles/stiff and flexible
    dihedrals/impropers/LJ pairs and their index ranges in the topology.

### CLI

```bash
make_top -p initial.top -m geometry.xyz [-l geom|avg] [-o output.top]
```

| Flag | Description |
|---|---|
| `-p, --top` | Initial topology (`.top`, atoms/atomtypes only). **Required.** |
| `-m, --mol` | Molecule geometry matching the topology's atom order. **Required.** |
| `-l, --lj` | Lennard-Jones mixing rule for 1-4 pair parameters: `geom` (geometric average, default) or `avg` (arithmetic average). |
| `-o, --out` | Output topology file. Defaults to `<TopFile>.new.top`. The `.csv` and `_deps.dat` files are named after this (or the input) automatically. |

Example (from `tests/run_test.sh`):

```bash
make_top -p BTBT.top -m BTBT.xyz -o BTBT_symm.top
```

## `map_atoms` — map atoms between two structures

Finds the atom-by-atom correspondence between two XYZ structures of the
same molecule that have different atom orderings — e.g. a force-field
geometry vs. an experimental crystal structure. It builds a bond graph
for each structure (element identity as node label) and searches for the
graph isomorphism that minimises the Kabsch-aligned RMSD between the two
sets of coordinates, so the atom correspondence is chosen based on actual
3D similarity rather than an arbitrary isomorphism.

### CLI

```bash
map_atoms -r reference.xyz -t target.xyz [-m map.txt] [-o reordered.xyz] [-v]
```

| Flag | Description |
|---|---|
| `-r, --ref` | Reference structure — defines the atom numbering you want to end up with (e.g. the crystal structure). **Required.** |
| `-t, --tgt` | Target structure to be matched onto the reference (e.g. the force-field geometry). **Required.** |
| `-m, --map` | Output file for the atom map: two columns, `reference target`, 1-based. Default `map.txt`. |
| `-o, --output` | Optional: write the target structure reordered to follow the reference's atom order (format inferred from the extension, e.g. `.xyz`). |
| `-v, --verbose` | Log progress, including intermediate isomorphism counts for symmetric molecules. |

**Note:** molecules with internal symmetry (fused aromatics, alkyl
chains, phenyl substituents, ...) can have very many graph automorphisms;
the search evaluates the RMSD of every one to find the best match, so it
can take a while for heavily symmetric structures. Use `-v` to monitor
progress.

## `renumber_top` — renumber a topology's atoms

Applies a `reference target` atom map (such as the one written by
`map_atoms`) to a `.top` file, renumbering its atoms in place so they
follow the reference (e.g. crystallographic) numbering. All bonded terms
(bonds, angles, dihedrals, impropers, pairs, exclusions, cmaps) reference
the same atom objects internally and are written out using their live
atom number, so renumbering the atom list is enough to keep the whole
topology consistent — nothing else needs to be touched.

The renumbering is validated before it's applied: every atom must have an
entry in the map, and the resulting numbering must be a genuine bijection
(no two atoms ending up with the same number). This is deliberately
strict by default, since a partial or malformed map applied silently can
otherwise create duplicate atom numbers and corrupt the topology.

### CLI

```bash
renumber_top -p topology.top -m map.txt [-n 0] [-o renumbered.top] [--allow-partial] [-v]
```

| Flag | Description |
|---|---|
| `-p, --top` | Topology to renumber. **Required.** |
| `-m, --map` | Two-column `reference target` atom map (1-based), e.g. from `map_atoms`. **Required.** |
| `-n, --molecule` | Index of the molecule (within `top.molecules`) to renumber. Default `0`. |
| `--allow-partial` | Allow atoms missing from the map to keep their original number instead of raising an error. Unsafe — can create duplicate atom numbers if the kept numbers collide with mapped ones. |
| `-o, --output` | Output file. Defaults to `<TopFile (no extension)>.renumbered.top`. |
| `-v, --verbose` | Print every old → new atom number pair applied. |

Example, chaining both new tools onto a `make_top` output:

```bash
map_atoms   -r crystal.xyz -t BTBT_symm_geometry.xyz -m map.txt
renumber_top -p BTBT_symm.top -m map.txt -o BTBT_symm.renumbered.top -v
```

## `joyce_to_openmm` — make a topology dual-valid for GROMACS and OpenMM

Rewrites a Joyce-generated `.top` (and every file it `#include`s) into a
copy that real GROMACS and OpenMM read **identically**. It fixes three
things purely in the file's own text, never loading GROMACS, OpenMM, or
any simulation package itself:

- `[ pairs ]` funct 2 (`ai aj 2 fudgeQQ qi qj V W`, Joyce's format for
  *every* 1-4-or-further nonbonded pair) is rejected outright by
  OpenMM's topology reader, which only understands funct 1
  (`ai aj 1 V W`) — rewritten accordingly.
- `[ pairs ]` is a *list*, not a *set*: the same atom pair can
  legitimately appear on more than one line, and GROMACS sums every
  listed line as an independent interaction. A duplicate line for a
  pair already seen is commented out (kept, tagged `; DUPLICATE ...`,
  never deleted), first occurrence kept live.
- A funct-1 line has no per-pair charge override, so naively rewriting
  funct 2 to funct 1 can silently reintroduce a nonzero 1-4 Coulomb
  term. `[ defaults ]`'s global `fudgeQQ` is forced to `0.0` to prevent
  this — safe only once every kept pair is confirmed to have
  `fudgeQQ*qi*qj == 0`; if one doesn't, conversion is refused
  (`ChargeOverrideError`) rather than silently producing a file GROMACS
  and OpenMM would disagree on.

See `joyce_to_openmm.py`'s module docstring for the full diagnosis of
each of the three issues above, including how each was confirmed real.
Verified against a real GROMACS single-point-energy oracle on 18 real
systems (~20 to ~146 000 atoms) in the `oligomer_builder` pipeline
package, where the OpenMM-`System`-internal issues this pure converter
does *not* address (comb-rule-dependent unit handling, a dihedral
multiplicity-0 crash, spurious auto-generated 1-4s at a merged
junction, a dispersion-correction default mismatch) are also fixed.

**Important: converting the file makes both programs able to *read* it;
it does not make them compute the *same* energy.** Once a `.top` has
been through this conversion, real GROMACS reads it and computes the
correct energy — that's what the conversion was verified against. Bare
OpenMM (`GromacsTopFile.createSystem()`, with no further patching) will
now *parse* the same file without crashing, but will still get the
energy wrong in four more ways that have nothing to do with the file's
text: a comb-rule-dependent misread of an explicit pair's `V W`, a crash
on any zero-multiplicity dihedral, auto-generated 1-4 interactions for
every topological 1-4 pair (wrong the moment a merged junction has one
with no `[ pairs ]` line), and a dispersion-correction default GROMACS's
own default doesn't match. Making OpenMM actually *agree* with GROMACS —
not just read the same file — needs `openmm_compat.load_top` below on
top of this conversion, not instead of it.

### CLI

```bash
joyce_to_openmm -p topology.top -o output_dir/ [-v]
```

| Flag | Description |
|---|---|
| `-p, --top` | Topology to convert. **Required.** |
| `-o, --output-dir` | Directory the converted topology (and any `#include`d file it needs) is written into. **Required.** |
| `-v, --verbose` | Print progress information. |

Example, converting a `make_top` output before handing it to OpenMM:

```bash
joyce_to_openmm -p BTBT_symm.top -o BTBT_symm_openmm/ -v
```

## `openmm_compat` — build a correct OpenMM `System`, not just a parseable one

A library module (no CLI): `openmm_compat.load_top(top_path, gro_path,
...)` builds an OpenMM `(topology, system, dropped_dihedral_offset)`
from a Joyce-generated `.top` + matching `.gro`, patching the *four*
things `GromacsTopFile.createSystem()` still gets wrong even on a file
already fixed by `joyce_to_openmm` (see the note above): a comb-rule-
dependent misread of an explicit pair's `V`/`W`, a crash on any
zero-multiplicity `[ dihedrals ]` term, auto-generated 1-4 interactions
for every topological 1-4 pair (only wrong once a merged junction has
one with no `[ pairs ]` line — never the case for a lone Joyce monomer,
the only kind of `.top` this package produces, but the fix is applied
unconditionally and is correct either way), and a dispersion-correction
default that doesn't match GROMACS's own. It reuses `joyce_to_openmm`'s
own file-rewriting internals rather than duplicating them, so both the
file-level and `System`-level fixes come from the same single parse.

This is a deliberate duplication of `oligomer_builder.openmm_compat`
(where this fix originates and was verified against a real GROMACS
oracle on 18 real systems, ~20 to ~146 000 atoms) — see that module's
`HANDOFF.md` entry for the full diagnosis and verification record.
`oligomer_builder` needs its own copy regardless of this one: the
auto-generated-1-4-at-a-junction bug only exists once two Joyce
fragments are merged into one system, which never happens to a `.top`
this package (SelIntCoords) ever sees.

```python
from SelIntCoords.openmm_compat import load_top, single_point_energy

topology, system, dropped_offset = load_top("BTBT_symm.top", "BTBT.gro")
energy, offset = single_point_energy("BTBT_symm.top", "BTBT.gro")
```

Needs `openmm` (`pip install openmm`) — not in `requirements.txt`,
matching how `psi4` is already documented rather than declared: this is
the only module in the package that uses it.

## Python API

Everything above is also usable programmatically.

```python
from SelIntCoords.top import TOP
from SelIntCoords.sel_intcoords import list_intcoords
from SelIntCoords.make_top import add_terms, geom_avg_mixing
from SelIntCoords.map_atoms import AtomMapper
from SelIntCoords.renumber import read_map, renumber_molecule
from SelIntCoords.joyce_to_openmm import convert_top, ChargeOverrideError

# 1. Derive internal coordinates and build a topology
data = list_intcoords("geometry.xyz")
bds, angles, stiff, impdiheds, flex, LJs, excls, rings, eq = data
topobj = add_terms("initial.top", bds, angles, stiff, impdiheds, flex,
                    LJs, excls, mixing=geom_avg_mixing)
topobj.write("output.top")

# 2. Map atoms between two geometries
mapper = AtomMapper("crystal.xyz", "geometry.xyz")
result = mapper.run()
result.as_dict(one_based=True)     # {ref_atom: tgt_atom, ...}
result.as_array(one_based=True)    # (N, 2) numpy array
result.write("map.txt")
mapper.write_reordered("reordered.xyz")

# 3. Renumber a topology in place
top = TOP("output.top")
renummap = read_map("map.txt")
renumber_molecule(top.molecules[0], renummap)
top.write("renumbered.top")

# 4. Make a topology dual-valid for GROMACS and OpenMM
converted = convert_top("output.top", "output_dir")

# 5. Build a correct OpenMM System from it (not just a parseable one)
from SelIntCoords.openmm_compat import load_top
topology, system, dropped_offset = load_top("output.top", "output.gro")
```

## Package layout

```
SelIntCoords/
├── setup.py
├── requirements.txt
├── SelIntCoords/
│   ├── sel_intcoords.py   # internal-coordinate enumeration, symmetry/ring/
│   │                      # equivalence detection (library, no CLI)
│   ├── dihed_deps.py      # ring-based stiff-dihedral equivalence/dependency
│   │                      # helpers used by make_top (library, no CLI)
│   ├── make_top.py        # CLI: build a full topology from geometry + LJ/
│   │                      # dependency/coordinate-count outputs
│   ├── map_atoms.py       # CLI: graph-isomorphism + RMSD atom mapping
│   │                      # between two structures
│   ├── renumber.py        # CLI: renumber a topology's atoms from a map
│   ├── joyce_to_openmm.py # CLI: rewrite a .top for dual GROMACS/OpenMM
│   │                      # validity (pure text, no other dependency)
│   ├── openmm_compat.py   # library: build a correct (not just parseable)
│   │                      # OpenMM System from a Joyce .top (needs openmm)
│   ├── blocks.py          # core topology data model (Atom, Molecule,
│   │                      # Bond/Angle/Dihedral/... Param types, System)
│   └── top.py             # GROMACS .top parser/writer (TOP class),
│                           # adapted from GromacsWrapper
└── tests/
    ├── BTBT.top / BTBT.xyz
    ├── DNBDT.top / DNBDT.xyz
    ├── PN.top / PN.xyz
    └── run_test.sh
```

`blocks.py` and `top.py` are adapted from
[GromacsWrapper](https://github.com/Becksteinlab/GromacsWrapper)'s
topology file-format handling and provide the underlying data model:
`TOP.molecules` is a tuple of `Molecule` objects, each holding `.atoms`
and lists of bonded terms (`.bonds`, `.angles`, `.dihedrals`,
`.impropers`, `.pairs`, `.exclusions`, `.cmaps`, ...), where every bonded
term stores direct references to the relevant `Atom` objects rather than
plain indices.

## Tests

`tests/run_test.sh` runs `make_top` on three example cores (BTBT, DNBDT,
pentacene) starting from a minimal `.top` (atoms/atomtypes only) and the
corresponding `.xyz` geometry:

```bash
cd tests
bash run_test.sh
```

This produces `BTBT_symm.top`, `DNBDT_symm.top`, and `PN_symm.top` (plus
their `.csv` and `_deps.dat` companions) to compare against known-good
references, then runs `joyce_to_openmm` on `BTBT_symm.top` as a smoke
test of the conversion.
