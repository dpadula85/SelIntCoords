<div align="center">

# SelIntCoords

**Select and group internal coordinates from a molecular geometry for QMD-FF parameterisation**

[![tests](https://github.com/dpadula85/SelIntCoords/actions/workflows/tests.yml/badge.svg)](https://github.com/dpadula85/SelIntCoords/actions/workflows/tests.yml)
[![version](https://img.shields.io/badge/version-1.2-blue)](setup.py)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](setup.py)
[![license](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

</div>

---

[Joyce](https://www.dsf.unica.it/~fabio/Joyce.php) fits a force field to
quantum-mechanical data, one set of parameters per *internal coordinate*
(bond, angle, dihedral). Which coordinates exist, which are rigid and
which rotate, and which are equivalent by symmetry and must share a
parameter, has to be told to Joyce up front. SelIntCoords works that out
from the geometry, writes the GROMACS topology Joyce starts from, and
provides the tools to keep that topology usable afterwards: renumbering
it to an experimental structure, and making it read identically by GROMACS
and OpenMM.

It targets rigid, (semi-)planar organic semiconductor cores with optional
alkyl/ether side chains (examples: BTBT, DNBDT, pentacene, DPP, D18). The
output feeds [oligomer_builder](https://github.com/dpadula85/oligomer_builder).

## Installation

```bash
git clone https://github.com/dpadula85/chain_cropper.git
pip install -e chain_cropper          # not on PyPI: install it first
git clone https://github.com/dpadula85/SelIntCoords.git
pip install -e SelIntCoords
```

`make_top` also needs [psi4](https://psi4.org) for point-group detection,
which is conda-only: `conda install -c conda-forge psi4`. Every other tool
runs without it. `openmm_compat` needs `openmm` (`pip install openmm`).

## Which tool do I need?

| I want to... | Tool | Needs |
|---|---|---|
| Get the topology and `$dependence` file Joyce starts from | `make_top` | psi4 |
| Make atom numbering follow another structure (e.g. crystal) | `map_atoms`, then `renumber_top` | - |
| Make a Joyce `.top` readable by both GROMACS and OpenMM | `joyce_to_openmm` | - |
| Get OpenMM energies equal to GROMACS ones | `openmm_compat.load_top` | openmm |

```
 initial.top + geometry.xyz ──(1) make_top──> geometry_symm.top, .csv, _deps.dat ──> Joyce
                                                        │
 crystal.xyz ──(2) map_atoms──> map.txt ──> renumber_top┘   (atoms now follow crystal.xyz)
                                                        │
                                  (3) joyce_to_openmm ──┴──> .top for GROMACS and OpenMM
```

## make_top

Input: a minimal `.top` with only `[ atomtypes ]` and `[ atoms ]`, and a
geometry with the same atom order. It guesses the bonds, then enumerates
bonds, angles, proper and improper dihedrals, 1-n Lennard-Jones pairs and
exclusions from the bond graph. Proper dihedrals are split into **stiff**
(ring and conjugated core, near-planar) and **flexible** (rotatable side
chains). Atoms equivalent by point-group symmetry, or by local
connectivity (the three H of a methyl), are identified so that
equivalent coordinates are fitted together. All new terms get zero
parameters for Joyce to fit.

```bash
make_top -p BTBT.top -m BTBT.xyz [-l geom|avg] [-o BTBT_symm.top]
```

`-l` picks the mixing rule for the 1-4 pair parameters: geometric (default)
or arithmetic. Output, next to `-o`: the topology, a `_deps.dat` in Joyce's
`$dependence` format, and a `.csv` with the number of each term and its
index range in the topology.

## map_atoms and renumber_top

A force field is built on one geometry, but a crystal structure numbers
the same atoms differently. `map_atoms` finds the atom-to-atom
correspondence between two structures of one molecule: it compares bond
graphs and, among all graph isomorphisms, keeps the one with the lowest
Kabsch-aligned RMSD, so symmetric molecules map onto the geometrically
sensible copy. Fused aromatics and alkyl chains have many automorphisms
and can take a while: use `-v` to follow progress.

`renumber_top` applies that map to a topology. Every bonded term refers to
its atoms directly, so renumbering the atoms keeps everything consistent.
It refuses a map that misses atoms or is not a bijection, since that would
silently corrupt the topology (`--allow-partial` overrides, at your risk).

```bash
map_atoms -r crystal.xyz -t BTBT.xyz -m map.txt [-o reordered.xyz] [-v]
renumber_top -p BTBT_symm.top -m map.txt [-o out.top] [-n MOL_INDEX] [--allow-partial] [-v]
```

`map.txt` has two columns, `reference target`, 1-based.

## joyce_to_openmm

Joyce writes every non-bonded 1-4-or-further pair as `[ pairs ]` funct 2,
with a per-pair charge. GROMACS reads that; OpenMM's reader rejects it.
`joyce_to_openmm` rewrites the `.top`, and every file it `#include`s, so
both read it identically:

- `[ pairs ]` funct 2 becomes funct 1;
- a pair listed twice (GROMACS sums both) is commented out on the second
  line, tagged `; DUPLICATE`;
- `fudgeQQ` is set to 0, since funct 1 has no per-pair charge. This is only
  safe if every pair has `fudgeQQ*qi*qj == 0`; otherwise the conversion is
  refused (`ChargeOverrideError`) instead of changing the energy.

```bash
joyce_to_openmm -p BTBT_symm.top -o out_dir/ [-v]
```

The converted file can be *read* by both programs, which is not the same
as getting the *same energy*: OpenMM's `createSystem()` still misreads
explicit pair parameters under some combination rules, crashes on
zero-multiplicity dihedrals, adds 1-4 interactions the topology does not
list, and uses a different dispersion-correction default than GROMACS.

## openmm_compat

`load_top(top_path, gro_path)` builds an OpenMM `(topology, system,
dropped_dihedral_offset)` with those four problems fixed, so energies
agree with GROMACS. `single_point_energy` wraps it. A copy of the same
module lives in oligomer_builder, where it was validated against GROMACS
on 18 systems (about 20 to 146 000 atoms).

```python
from SelIntCoords.openmm_compat import load_top, single_point_energy

topology, system, offset = load_top("BTBT_symm.top", "BTBT.gro")
energy, offset = single_point_energy("BTBT_symm.top", "BTBT.gro")
```

## Python API

The same steps as functions; [examples/run_example.py](examples/run_example.py)
runs them end to end.

```python
from SelIntCoords.sel_intcoords import list_intcoords
from SelIntCoords.make_top import add_terms, geom_avg_mixing
from SelIntCoords.map_atoms import AtomMapper
from SelIntCoords.renumber import read_map, renumber_molecule
from SelIntCoords.top import TOP
from SelIntCoords.joyce_to_openmm import convert_top

bds, angs, stiff, imps, flex, LJs, excls, rings, eq = list_intcoords("geometry.xyz")
add_terms("initial.top", bds, angs, stiff, imps, flex, LJs, excls,
          mixing=geom_avg_mixing).write("output.top")

mapper = AtomMapper("crystal.xyz", "geometry.xyz")
result = mapper.run()                       # AtomMapping: .mapping, .rmsd
result.as_dict(one_based=True)              # {reference atom: target atom}
result.write("map.txt")
mapper.write_reordered("reordered.xyz")

top = TOP("output.top")
renumber_molecule(top.molecules[0], read_map("map.txt"))
top.write("renumbered.top")

convert_top("output.top", "out_dir")
```

## Examples

[examples/](examples/) holds seven molecules, each with the two inputs
`make_top` needs: three cores (BTBT, DNBDT, pentacene) and four from
Joyce parameterisations (DPP, PNDI2OD, D18, PBDTTT-O). `run_example.sh`
runs `make_top` on all of them, then `joyce_to_openmm`;
`run_example.py` runs the same steps for BTBT through the Python API. The
examples README lists the number of terms found for each.

## Package layout

| Module | Role |
|---|---|
| `sel_intcoords.py` | enumerate internal coordinates; symmetry and ring equivalences |
| `dihed_deps.py` | group stiff ring dihedrals that share a force constant |
| `make_top.py` | `make_top` CLI: assemble topology, `$dependence` file, CSV |
| `map_atoms.py` | `map_atoms` CLI: graph isomorphism + RMSD atom mapping |
| `renumber.py` | `renumber_top` CLI |
| `joyce_to_openmm.py` | `joyce_to_openmm` CLI: text-only rewrite, standard library only |
| `openmm_compat.py` | `System` builder that matches GROMACS energies |
| `blocks.py`, `top.py` | topology data model and `.top` parser/writer, adapted from [GromacsWrapper](https://github.com/Becksteinlab/GromacsWrapper) |

## When something goes wrong

- **`ImportError: ... psi4`** from `make_top`: install psi4 from conda-forge (see Installation).
- **`map_atoms` says the graphs are not isomorphic**: the two files are different molecules, or bond guessing differs because one geometry is too distorted; compare the bond counts.
- **`map_atoms` is slow**: many symmetric groups mean many isomorphisms; run with `-v`.
- **`renumber_top` raises "not a bijection"**: the map has duplicate numbers; regenerate it with `map_atoms`.
- **`ChargeOverrideError`** from `joyce_to_openmm`: a pair has a nonzero `fudgeQQ*qi*qj`, so funct 1 cannot represent it; the topology is not converted.

## Tests

```bash
pip install -e "SelIntCoords[test]"
pytest tests/
```

The suite covers term counts for the three cores, atom mapping and
renumbering (including the refusals), the converter, and the OpenMM
energies. The one test that runs `make_top` with real point-group
detection skips itself when psi4 is not installed, which is also the case
on CI.
