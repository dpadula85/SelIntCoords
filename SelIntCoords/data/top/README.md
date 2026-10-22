# Bundled force-field data

`oplsaa.ff/` holds the four files of the OPLS-AA force field that atom type perception reads, copied unchanged
from the `share/gromacs/top/oplsaa.ff` directory of GROMACS 2020:

| file | used for |
|---|---|
| `ffnonbonded.itp` | mass, sigma and epsilon of every `opls_NNN` type |
| `atomtypes.atp` | one-line description of every type |
| `forcefield.itp` | combination rule |
| `aminoacids.rtp` | residue templates, from which the learned types are built |

GROMACS is distributed under the GNU Lesser General Public License, version 2.1 (https://www.gromacs.org).
The OPLS-AA parameters are those of

- W. L. Jorgensen, D. S. Maxwell, J. Tirado-Rives, J. Am. Chem. Soc. 1996, 118, 11225
  (https://doi.org/10.1021/ja9621760);
- G. A. Kaminski, R. A. Friesner, J. Tirado-Rives, W. L. Jorgensen, J. Phys. Chem. B 2001, 105, 6474
  (https://doi.org/10.1021/jp003919d), for the peptide dihedrals of the residue templates.

Any other force field of the GROMACS database (`--ff NAME`) is read from the GROMACS installation, given by
`--gmx-top`, `$GMXLIB`, `$GMXDATA/top` or the location of the `gmx` executable.
