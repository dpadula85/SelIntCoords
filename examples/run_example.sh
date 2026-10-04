#!/bin/bash
# make_top on every core in this directory (needs psi4, see ../README.md),
# then make one result readable by both GROMACS and OpenMM.
set -e
cd "$(dirname "$0")"

for m in BTBT DNBDT PN DPP PNDI2OD D18 PBDTTT-O; do
    make_top -p $m/$m.top -m $m/$m.xyz -o $m/${m}_symm.top
done
joyce_to_openmm -p BTBT/BTBT_symm.top -o BTBT/openmm
