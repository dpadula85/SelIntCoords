#!/usr/bin/env python

'''
extra_types - Lennard-Jones parameters for elements the GROMACS force fields
do not cover.

An atom of such an element gets an atomtype named after the element, with the
mass and the parameters below; sigma in nm, epsilon in kJ/mol,
for use with the geometric (comb-rule 3) mixing of OPLS-AA. An element without
an entry gets a placeholder whose parameters are written as `FILL_ME`.
'''

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtraType:
    '''Parameters of an element, with where they come from.'''
    mass: float
    sigma: float
    epsilon: float
    source: str


EXTRA_TYPES = {
    # as in the PBDTTT-Se topology; 0.291 kcal/mol = 1.21754 kJ/mol
    'Se': ExtraType(
        78.971, 0.37741, 1.21754,
        'Loschwitz et al., Data in Brief 2021, doi:10.1016/j.dib.2021.106948 (AMBER parameters)'),
    # as in the TIPS-pentacene, ISOTIPS-pentacene and DTS topologies
    'Si': ExtraType(
        28.0855, 0.4184, 0.295,
        'Lorenz et al., Tribol. Lett. 2005, doi:10.1007/s11249-005-5085-4'),
}
