#!/bin/bash
# Build the symmetry-grouped topology of three cores, then make one of them
# readable by both GROMACS and OpenMM. Needs psi4 (see ../README.md).
set -e

make_top -p BTBT.top -m BTBT.xyz -o BTBT_symm.top
make_top -p PN.top -m PN.xyz -o PN_symm.top
make_top -p DNBDT.top -m DNBDT.xyz -o DNBDT_symm.top
joyce_to_openmm -p BTBT_symm.top -o joyce_to_openmm_out
