"""The run_example.sh workflow through the Python API, for BTBT.

make_top needs psi4 for the symmetry grouping; map_atoms, renumber and
joyce_to_openmm do not.
"""
from pathlib import Path

from SelIntCoords.top import TOP
from SelIntCoords.map_atoms import AtomMapper
from SelIntCoords.renumber import read_map, renumber_molecule
from SelIntCoords.joyce_to_openmm import convert_top
from SelIntCoords.sel_intcoords import list_intcoords
from SelIntCoords.make_top import add_terms, geom_avg_mixing

here = Path(__file__).resolve().parent

# 1. Internal coordinates of the geometry, added to the minimal topology
bds, angs, stiff, imps, flex, LJs, excls, rings, eq = list_intcoords(str(here / "BTBT.xyz"))
top = add_terms(str(here / "BTBT.top"), bds, angs, stiff, imps, flex, LJs, excls,
                mixing=geom_avg_mixing)
top.write(str(here / "BTBT_symm.top"))
print(f"{len(bds)} bonds, {len(angs)} angles, {len(stiff)} stiff and {len(flex)} flexible dihedrals")

# 2. Number the atoms as in another structure (here, the same one: identity-like map)
mapper = AtomMapper(here / "BTBT.xyz", here / "BTBT.xyz")
result = mapper.run()
result.write(here / "map.txt")
print(f"mapped {len(result.mapping)} atoms, RMSD {result.rmsd:.4f} A")

top = TOP(str(here / "BTBT_symm.top"))
renumber_molecule(top.molecules[0], read_map(here / "map.txt"))
top.write(str(here / "BTBT_symm.renumbered.top"))

# 3. A copy that GROMACS and OpenMM read identically
print(convert_top(here / "BTBT_symm.top", here / "joyce_to_openmm_out"))
