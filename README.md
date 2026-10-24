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

Nothing else is needed beyond `pip`: the OPLS-AA files for
[atom type perception](#atom-type-perception) are shipped with the package
(GROMACS is only needed to perceive types with another force field). `openmm_compat` needs `openmm` (`pip install openmm`).

## Which tool do I need?

| I want to... | Tool | Needs |
|---|---|---|
| Get the topology and `$dependence` file Joyce starts from, from the `.xyz` alone | `make_top -m geometry.xyz` | - |
| The same, with atom types you wrote or edited yourself | `make_top -p initial.top -m geometry.xyz` | - |
| Only look at the atom types a geometry gets (no bonded terms) | `perceive_types` | - |
| Make atom numbering follow another structure (e.g. crystal) | `map_atoms`, then `renumber_top` | - |
| Make a Joyce `.top` readable by both GROMACS and OpenMM | `joyce_to_openmm` | - |
| Get OpenMM energies equal to GROMACS ones | `openmm_compat.load_top` | openmm |

```
 geometry.xyz ──(1) make_top──> geometry.top, .csv, _deps.dat, .types ──> Joyce
                   (atom types perceived inside; or give your own with -p initial.top)
                                                        │
 crystal.xyz ──(2) map_atoms──> map.txt ──> renumber_top┘   (atoms now follow crystal.xyz)
                                                        │
                                  (3) joyce_to_openmm ──┴──> .top for GROMACS and OpenMM
```

## make_top

Input: a geometry. `make_top` perceives the atom type of every atom (see
[below](#atom-type-perception)), guesses the bonds, then enumerates
bonds, angles, proper and improper dihedrals, 1-n Lennard-Jones pairs and
exclusions from the bond graph. Proper dihedrals are split into **stiff**
(ring and conjugated core, near-planar) and **flexible** (rotatable side
chains). Atoms equivalent by the symmetry of the bond graph (the three H of
a methyl, the two methyls of a gem-dimethyl, the two halves of a symmetric
molecule, whatever the conformation; never atoms of different atom type) are
identified so that equivalent coordinates are fitted together. All new terms get zero
parameters for Joyce to fit, except the methyl rotations ([below](#rotatable-bonds)).

```bash
make_top -m BTBT.xyz                       # types perceived inside, full topology out
make_top -p BTBT.top -m BTBT.xyz           # your own minimal .top (atomtypes + atoms), same atom order
```

Without `-o` the outputs are `BTBT.top`, `BTBT.csv`, `BTBT_deps.dat` and the
report `BTBT.types`. With `-o X.top` they are `X.top`, `X.csv`, `X_deps.dat`,
`X.types`. When you give `-p`, nothing is perceived, your atom types are used as
they are, and the topology is `BTBT.new.top` (there is an older one to tell it from).

`-l` picks the mixing rule for the pair parameters: geometric (default)
or arithmetic. Output, next to `-o`: the topology, a `_deps.dat` in Joyce's
`$dependence` format, and a `.csv` with the number of each term and its index range in the topology.

### What make_top does, step by step

The rules in the order they are applied. Each step uses only what the steps above it produced, and
the later sections of this README give the details.

1. **Atom types.** Perceived from the geometry ([below](#atom-type-perception)) or read from `-p`.
   They give the Lennard-Jones parameters and tell atoms apart: two atoms of different type are never
   equivalent.
2. **Bonds.** Guessed from the distances between atoms. From here on the molecule is a graph: atoms
   are nodes, bonds are edges. Almost every rule below looks at the graph, not at the coordinates.
3. **Rings.** The smallest set of rings of the graph. A *ring system* is a set of rings that share a
   bond (fused rings are one system).
4. **Equivalent atoms.** Two atoms are equivalent when the graph can be relabelled, keeping every
   element and atom type, so that one lands on the other: the three H of a methyl, the two halves of a
   symmetric molecule. This depends on the bonds and types only, not on the conformation.
5. **Coordinates.** Every bond; every angle (three bonded atoms in a row); every proper dihedral (four
   bonded atoms in a row). For each *sp2 atom* (a heavy atom with fewer than four bonds that is not in a
   saturated side chain) with three or more neighbours, one **improper** (`starlike`): the atom and its
   neighbours, there to keep the atom planar.
6. **Stiff or flexible.** A proper dihedral is *stiff* when its two central atoms are sp2 and in the same
   ring. Each ring torsion exists twice (near 0 and near 180 degrees): the one within 15 degrees of 0 is
   kept. Every other proper dihedral is *flexible*.
7. **Kind of each flexible dihedral**, by the bond it turns about; the first rule that fits wins
   ([table](#rotatable-bonds)):
   1. a methyl: three H-C-X-Y dihedrals, parameters fixed from the literature;
   2. a ring bond that is not part of the planar framework: literature parameters;
   3. a double bond: stiff if an atom is in a ring, otherwise a rotor fitted to a scan (`phi`);
   4. a single bond between two rigid units: a rotor fitted to a scan (`delta`);
   5. the first bond of a side chain (aryl-S-CH3, ring-CH2-CH3, N-CH2-CH3): a rotor fitted to a scan,
      numbered after the ones in 4 (`delta`);
   6. anything else: left with zero parameters for you.

   Of the dihedrals about one scanned bond one is kept, and bonds that are equivalent by symmetry share it.
8. **Order.** Bonds, angles and stiff dihedrals are grouped by ring system under a header
   ([grouping](#grouping-by-ring-system)); then the impropers; then the flexible dihedrals, methyls first,
   then the scanned ones. The numbers are fixed *now*, so the dependencies refer to the order of the file.
9. **Pairs and exclusions.** All pairs of atoms are excluded, so that only the `[ pairs ]` lines written
   act. The 1,4 pairs are not written (the dihedral covers them); the others are chosen by twisting the
   molecule ([pairs](#which-pairs-are-written)). All parameters are zero, for Joyce to fit, except the
   methyl and ring-dihedral values of step 7 and the pair parameters (mixing rule).
10. **Dependencies** ([below](#dependencies)): which coordinates must share a force constant.

### Dependencies

`_deps.dat` tells Joyce which coordinates share a force constant. The rules, in the order applied:

1. **Symmetry.** Two bonds, angles, dihedrals or impropers are tied when their atoms, position by
   position, are equivalent atoms (step 4 above). Atom types count.
2. **One parameter per ring.** The stiff coordinates (torsions and starlike impropers) of one ring are tied.
   A coordinate belongs to the ring that holds the central bond of a torsion, or the centre of an improper
   (the larger ring where two share the bond, then the lowest atoms). Rings equivalent by symmetry count as
   one, and coordinates with a hydrogen are kept apart from those without.
3. **Starlike impropers.** The impropers of a ring are five times the torsions of the same ring (same
   hydrogen class).
4. **One reference per class.** Ties chain together into classes; the coordinate with the lowest number is
   the reference and every other member is tied to it, once.

Rule 2 is there because the six torsions of a planar ring are not independent at the harmonic level: the
Hessian fixes only their collective out-of-plane stiffness. Fitted separately, Joyce splits it between them
with arbitrary signs, and force constants below zero appear, which a torsion that keeps a ring planar should
never have. One parameter per ring leaves a single, well-determined number.

Each line ends with a `;` comment saying what is tied and why:

```
  7 =   1*1.d0 ; bond 4-5 = bond 1-2 (CA1-CA2): symm eq
364 = 363*1.d0 ; dihed 1-2-3-4 (CA1-CA2-CA1-CA1) = dihed 6-1-2-3 (CA1-CA1-CA2-CA1): same k in ring 1-2-3-4-5-6
479 = 363*5.d0 ; star 1-2-6-19 (CA1-CA2-CA1-CA1) = dihed 6-1-2-3 (CA1-CA1-CA2-CA1): oop in ring 1-2-3-4-5-6, 5x k
```

`symm eq` is symmetry (rule 1), `same k in ring` rule 2, `oop` (out-of-plane) rule 3, `k` the force constant; `dihed`
is a stiff dihedral, `star` a starlike improper.

### Which pairs are written

A topology made for Joyce excludes every non-bonded pair inside the molecule, so two atoms without a
`[ pairs ]` line pass through each other. `make_top` writes a pair when some conformer of the molecule
brings it too close. The conformers are those of a rigid-rotor model: bond lengths, angles, rings and
double bonds stay as they are, and only the single bonds that split the molecule turn (methyl rotors
included). The selection does by computer what is done by hand: twist the molecule, see what collides,
add the pair, twist again.

1. Start with no pairs. Draw random conformers in which the pairs chosen so far are hard spheres.
2. A pair that comes closer than 0.8 sigma (`--pair-ratio`) in at least 1 % of them
   (`--pair-fraction`) is added.
3. If any pair between two ring systems bumps even once, every heavy-atom pair between the two systems
   is added (a stacking contact).
4. Repeat until nothing is added. Pairs that a symmetry of the bond graph maps onto each other are
   written together.

Only heavy-heavy and H...H pairs that are 1,5 or farther apart are written. Hydrogen-heavy pairs act as
hard spheres in the sampling but are not written, and 1,4 pairs are covered by the dihedral. The pair
parameters are the mixing-rule values, unscaled. Every pair not written is listed commented out after
the written ones, with its closest approach in units of sigma, so that one can be uncommented by hand.

| molecule | atoms | pairs written | `make_top` time |
|---|---|---|---|
| BTBT | 36 | 12 | 3 s |
| trans azo dye (`tests/data/DAB.xyz`) | 52 | 207 | 8 s |
| D18 monomer | 87 | 109 | 8 s |
| ZR1 | 120 | 182 | 15 s |

The cost grows with the number of atoms and of 1,5-or-farther pairs. A 254-atom oligomer takes about
5 minutes and gets about 2500 pairs, because in a flexible chain every ring can touch every other ring
and a ring contact brings in all the pairs of the two rings: the selection is meant for monomers and
cores, not oligomers.

**What was checked.** On the trans azo dye, joyce_autoval starts from 26 pairs and adds about 205 after
its hot MD and relaxed scans. The selection, from an empty list and without MD, contains 171 of the 201
heavy-heavy and H...H pairs of that final list (85 %) and 36 that autoval did not add. The pairs it
misses are one ring pair that autoval found only in the relaxed MM scans, where bonds and angles bend: a
rigid-rotor model cannot bring those two rings together. This is one molecule, not a systematic
validation. For retinals the selection gives 36-63 pairs against about 200 in hand-made topologies,
which were chosen for the torsional scans they affect and not only for contacts.

### Grouping by ring system

Bonds, angles and stiff dihedrals are written in groups, each under a
`; header` line that names the ring they belong to (`thiophene 1`, `benzene`,
`pentacene`, `benzene + 2 thiophene (fused)`). A ring system is a set of
rings that share a bond; the hydrogens of a ring stay with it. The groups are:

| header | coordinates in it |
|---|---|
| `thiophene 1` | every atom in that ring system |
| `thiophene 1 substituents` | the ring plus a side chain bonded to it only |
| `thiophene 1 to benzene` | a link (single bond, vinylene...) between two ring systems |
| `methyls` | anything that involves a methyl group |
| `starlike` | the improper dihedrals (written after the stiff ones) |

The coordinates are renumbered in this order *before* the dependencies are
computed, so `_deps.dat` and the `.csv` refer to the order of the file.

### Rotatable bonds

The dihedrals about a bond that is not in a ring are treated as you would by
hand:

| bond | what `make_top` writes |
|---|---|
| methyl on C, N, O, S or Se | `; methyls`: the H-C-X-Y dihedrals, funct 1, `n = 3`, `k = 0.651` kJ/mol, no phase (fixed, not fitted) |
| bond in a ring that is not part of the planar framework (sp3 carbons of a locking ring) | `; ring dihedrals (literature)`: every dihedral about it, same parameters as the methyls |
| single bond between two rigid units (rings, sp2 atoms, azo N), directly or through sp3 linker atoms (ring-ring, ring-C=O, ring-O-CH2-CH2-ring) | `; delta1`, `; delta2`...: one dihedral per bond, the one over the sp2 carbons when there is a choice, with a single `n = 2` term; equivalent bonds share the header |
| double bond with no atom in a ring (the chain of a retinal) | `; phi1`, `; phi2`...: the same as a `delta`, for a bond that is double |
| double bond with an atom in a ring (exocyclic C=C, C=N joining two rigid units) | all its dihedrals join the stiff ones |
| first bond of a side chain: from a rigid unit to an sp3 atom with a heavy atom beyond it (aryl-S-CH3, ring-CH2-CH3, N-CH2-CH3) | `; delta<N>`, numbered after the bonds between rigid units: one dihedral per bond, the one over the heavy atoms; equivalent bonds share the header and the same dihedral, carried from one to the other by a symmetry of the bond graph |
| anything else (the bonds of a side chain beyond the first, a group that turns as a unit such as CF3 or tert-butyl, Si methyls) | `; flexible (merge or by hand)`: zero parameters, left to `oligomer_builder` or to you |

The value 0.651 kJ/mol is 0.1556 kcal/mol, the GAFF barrier of `X-c3-c3-X` (Wang
et al., [J. Comput. Chem. 2004, 25, 1157](https://doi.org/10.1002/jcc.20035)); the
retinal QMD-FFs take their ring torsions from GAFF in the same way.

A bond is double when it is shorter than 0.92 times the sum of the covalent
radii (C=C 1.34-1.38 Angstrom against 1.43-1.47 for an sp2-sp2 single bond) and
the shortest of the candidates at both atoms, which keeps the single and double
bonds of a conjugated chain alternating.

## Atom type perception

`make_top` needs the **atom type** of each atom: the label that tells a
force field which Lennard-Jones parameters (and mass) to use, for example
"aromatic carbon" or "thiophene sulfur". It reads the types from the
geometry, so no starting `.top` has to be written, using the OPLS-AA force
field (mass and Lennard-Jones parameters per type, and the residue templates
of the force field). Its files are shipped with the package, copied unchanged
from GROMACS 2020 ([`SelIntCoords/data/top`](SelIntCoords/data/top/README.md),
with the licence and references); any other force field of the GROMACS
database is read from a GROMACS installation.

Bonds are guessed as in `make_top`, then each atom is classified from its
neighbours, its hybridisation (from the bond angles) and its rings (aromatic
means a planar five- or six-membered ring with the right electron count; a
ring containing a carbonyl, such as a lactam or imide ring, is not
aromatic). Two engines type the atoms, in this order:

1. **Curated rules** for OPLS-AA (hydrocarbons, aromatic and heteroaromatic
   rings, ethers, alcohols, amines, amides and imides, esters, ketones,
   acids, nitriles, nitro, halogens, sulfur groups), checked against the
   types the example `.top` files were written with.
2. **Types learned from the residue templates** of the force field, for an
   atom no rule covers, and as a cross-check on all the others.

An atom neither engine can type, or a charged species (pyridinium-type
nitrogen), raises an error: it is never guessed. Charges in the written
`.top` are zero.

```bash
make_top -m BTBT.xyz [--ff oplsaa] [--gmx-top DIR] [--ff-type-names] [--resname MOL]
perceive_types -m BTBT.xyz [-o BTBT_types.top]    # only the starting topology and the report
```

`--ff` is the force field (default `oplsaa`). `--gmx-top` points at a directory with
the `<name>.ff` folders; by default the shipped OPLS-AA files, and for another
force field `$GMXLIB`, `$GMXDATA/top`, or the one next to the `gmx` executable. Atomtypes are named by local environment (`CA1`, `CA2` for aromatic carbons bonded to
different neighbours, `HC`, `S`...) and atoms by element and number (`S1`, `C2`);
the force-field type stays in the comment of each atomtype line. `--ff-type-names`
names the atomtypes by force-field type (`opls_145`) instead. `perceive_types` is the same
perception on its own: it writes the minimal `.top` that `make_top -p` takes,
useful to inspect or edit the types before building the terms.

The report (`<name>.types` for `make_top`, `<name>_types.types` for `perceive_types`). For PBDTTT-O:

```
Atom types perceived from PBDTTT-O.xyz (oplsaa)
63 atoms: 63 typed by rule, 0 learned from residue templates, 25 by analogy, 2 to check
...
Typed by analogy (no dedicated type in oplsaa, the closest published one used):
  opls_633  S   in thiazole  <- aromatic S: atoms 9 15 25 34
...
Atoms to check:
   35 H   opls_469 (H on C bonded to an ester O)
        - rules and learned types disagree on the Lennard-Jones class
        - alternative opls_140 [learned (radius 2)]: alkane H.
```

Two things in the report need your judgement:

- **Typed by analogy.** OPLS-AA has no thiophene, furan or thiadiazole
  types. The aromatic S is typed as thiazole S (`opls_633`), the carbons of
  heteroaromatic rings as benzene C (`opls_145`), a two-coordinate aromatic N
  of a five-membered ring as imidazole N (`opls_557`), the sp2 carbons of
  a lactam ring as pyrrole C2/C3 (`opls_543/544`), and the conjugated carbons
  that are not aromatic by the electron count as benzene C (`opls_145`): a
  ring ketone C between two conjugated carbons (the quinone of PM6, the
  indanone of IDIC and Y6), an alkene C without H and with two conjugated
  neighbours (the `C=C(CN)2` arm), a pyramidalised sp2 C of a fullerene, and a
  quaternary sp3 C between two conjugated carbons. The `=CH-` of the same units
  stays an alkene C, and isolated alkenes and acyclic ketones keep their
  OPLS-AA types. These are the choices of the [examples](examples/) and of the
  QMD-FFs, not the only possible ones.
- **Atoms to check.** The rules and the learned types give different
  Lennard-Jones classes, or the learned type was the only answer and rests on
  a short-range environment.

Numbers behind it, for OPLS-AA, comparing the mass and Lennard-Jones
parameters of the perceived type with those of the topology each molecule
was parameterised with:

- the seven [examples](examples/) (362 atoms): 360 equal; the other two are the
  furan oxygens of PBDTTT-O, whose hand-typed epsilon (0.5876 kJ/mol) differs
  from `opls_566` (0.58576);
- the 74 molecule types of the QMD-FFs repository that have a geometry next to
  the topology (11481 atoms): 11070 of the 11074 atoms whose topology holds
  OPLS-AA values (or the literature values of Se and Si) are equal; the others
  have refitted Lennard-Jones values and cannot be compared. The 4 differences
  are NAP's fusion carbon and the hydroxyl of ETD, typed by hand with other
  parameters. The polymer repeat units of the BHJ project (13 molecule types):
  726 of 726 comparable atoms are equal;
- the exact `opls_` number can differ where several types share their
  parameters (the examples use `opls_139` for every alkane carbon, the rules
  `opls_135`-`139` by number of hydrogens);
- fed to `make_top`, the perceived types give the same bonds, angles,
  dihedrals, pairs and exclusions as the hand-written topology (BTBT, DPP,
  PNDI2OD);
- the learned engine, tested by leaving a residue and its variants out of its
  own training set, recovers the Lennard-Jones class of 93% of the OPLS-AA
  atoms.

Selenium and silicon are typed from the literature, as in your PBDTTT-Se and
TIPS_PN topologies (`extra_types.py`: Se 78.971 u, sigma 0.37741 nm, epsilon
1.21754 kJ/mol; Si 28.0855 u, 0.4184 nm, 0.295 kJ/mol, each with its reference).
Any other element with no OPLS-AA type gets an atomtype whose sigma and epsilon
read `FILL_ME`: `make_top` writes the topology with the marker, keeps
`<name>_types.top` (the same with placeholder numbers), and tells you to fill in
the two values and run `make_top -p <name>_types.top`. A group no rule covers
is an error that names the atoms.
Bond guessing is the one of `make_top`: in unrelaxed oligomers it can add a bond
between an H and an S that touch (1.47 Angstrom), which makes the S untypeable:
`make_top` then stops with an error that names the atom.

**Not validated:** force fields other than OPLS-AA have no curated rules
(`--ff amber99sb-ildn` runs on the learned engine alone and is far less
reliable for anything outside the biomolecular templates); neutral molecules
only; little testing outside the organic-semiconductor chemistry of the examples and
QMD-FFs beyond the 31 small functional-group cases in `tests/`.

```python
from SelIntCoords.perceive_types import perceive_to_top

top, assignments = perceive_to_top("BTBT.xyz", ff_name="oplsaa", class_name=True)   # as perceive_types
assignments[0].type, assignments[0].rule      # ('opls_633', 'aromatic S')
```

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
| `dihed_deps.py` | stiff coordinates that share a force constant because they sit in the same ring |
| `symmetry.py` | atoms equivalent by the symmetry of the bond graph (what `make_top` uses), and by the point group of a geometry (a library function, no external program) |
| `ring_groups.py` | ring systems, their names, and the grouping of coordinates under headers |
| `flex_classes.py` | kind of each rotatable bond: methyl, scanned single bond, double bond, other |
| `mol_features.py`, `oplsaa_rules.py`, `learned_types.py`, `ffdb.py`, `extra_types.py`, `perceive_types.py` | [atom type perception](#atom-type-perception) |
| `make_top.py` | `make_top` CLI: assemble topology, `$dependence` file, CSV |
| `map_atoms.py` | `map_atoms` CLI: graph isomorphism + RMSD atom mapping |
| `renumber.py` | `renumber_top` CLI |
| `joyce_to_openmm.py` | `joyce_to_openmm` CLI: text-only rewrite, standard library only |
| `openmm_compat.py` | `System` builder that matches GROMACS energies |
| `blocks.py`, `top.py` | topology data model and `.top` parser/writer, adapted from [GromacsWrapper](https://github.com/Becksteinlab/GromacsWrapper) |

## When something goes wrong

- **`make_top` finds fewer (or more) equivalent coordinates than expected**: the symmetry is searched with a 0.01 Angstrom tolerance (`get_equivalent_atoms(u, tol=...)`); a geometry that is only approximately symmetric needs a larger one.
- **`map_atoms` says the graphs are not isomorphic**: the two files are different molecules, or bond guessing differs because one geometry is too distorted; compare the bond counts.
- **`map_atoms` is slow**: many symmetric groups mean many isomorphisms; run with `-v`.
- **`renumber_top` raises "not a bijection"**: the map has duplicate numbers; regenerate it with `map_atoms`.
- **`ChargeOverrideError`** from `joyce_to_openmm`: a pair has a nonzero `fudgeQQ*qi*qj`, so funct 1 cannot represent it; the topology is not converted.

## Tests

```bash
pip install -e "SelIntCoords[test]"
pytest tests/
```

The suite covers term counts for the example cores, atom type perception, the
symmetry detection, ring grouping and rotor classes, atom mapping and
renumbering (including the refusals), the converter, and the OpenMM
energies; the one test that reads a GROMOS force field needs a GROMACS
installation and skips itself without it, as on CI.
