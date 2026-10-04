# Examples

Seven molecules, one directory each, with the two inputs `make_top` needs:
`<name>.top` (atoms and atomtypes only, no bonded terms) and `<name>.xyz`
(the geometry, same atom order).

| Directory | Formula | What it is |
|---|---|---|
| `BTBT` | C18H16S2 | benzothieno-benzothiophene core with two ethyl chains |
| `DNBDT` | C26H14S2 | dinaphtho-benzodithiophene core, no side chains |
| `PN` | C22H14 | pentacene |
| `DPP` | C22H16N2O2S4 | diketopyrrolopyrrole with thiophene arms |
| `PNDI2OD` | C26H18N2O4S2 | repeat unit of the naphthalene diimide-bithiophene polymer, short side chains |
| `D18` | C44H30F2N2S9 | repeat unit of the donor polymer D18 |
| `PBDTTT-O` | C31H23FO4S4 | repeat unit of the donor polymer PBDTTT-O |

`DPP` and `PNDI2OD` are the files Joyce was started from. For `D18` and
`PBDTTT-O` the input `.top` is the final Joyce topology with every section
except `[ defaults ]`, `[ atomtypes ]`, `[ moleculetype ]`, `[ atoms ]`,
`[ system ]` and `[ molecules ]` removed (charges set to zero for
`PBDTTT-O`), since the original starting files were not kept in that form.

## Running

```bash
bash run_example.sh      # make_top on all seven (needs psi4), then joyce_to_openmm on BTBT
python3 run_example.py   # the same steps for BTBT through the Python API, plus map_atoms/renumber
```

Each run writes `<name>_symm.top`, `<name>_symm.csv` and
`<name>_symm_deps.dat` next to the inputs.

## What it produces

Number of terms found (the `.csv`). For `DPP` and `PNDI2OD` the topology is
identical (ignoring comments) to the one Joyce was started from. For `D18`
and `PBDTTT-O` the bonds, angles, and stiff dihedrals plus impropers equal
those of their final Joyce topology.

| | Bonds | Angles | Stiff dihedrals | Flexible dihedrals | Impropers |
|---|---|---|---|---|---|
| BTBT | 39 | 68 | 34 | 30 | 14 |
| DNBDT | 48 | 80 | 64 | 0 | 26 |
| PN | 40 | 66 | 52 | 0 | 22 |
| DPP | 51 | 88 | 46 | 38 | 20 |
| PNDI2OD | 57 | 98 | 52 | 38 | 24 |
| D18 | 97 | 167 | 84 | 80 | 36 |
| PBDTTT-O | 69 | 118 | 54 | 60 | 25 |

`make_top` gives a starting topology that is meant to be edited. The
flexible dihedrals are where the Joyce topologies of `D18` (81) and
`PBDTTT-O` (67) were edited by hand; `DPP` kept the generated 38.

## Things to be careful of

- **Atom order**: the `.xyz` must list atoms in the same order as the `.top`.
- **No psi4, no `make_top`**: the symmetry grouping needs it (see the main README).
- **Heavy molecules**: `map_atoms` on the larger repeat units can take long, because of the many symmetric alkyl and ring automorphisms; use `-v`.
