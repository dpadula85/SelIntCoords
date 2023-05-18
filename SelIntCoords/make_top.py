#!/usr/bin/env python

import sys
import csv
import numpy as np
import pandas as pd
import argparse as arg

from .top import *
from .blocks import *
from .sel_intcoords import list_intcoords

# Default bond parameters
bparms = {
    'func' : 1,
    'param' : {
        'kb' : 0.0,
        'b0' : 0.0
    }
}

# Default angle parameters
aparms = {
    'func' : 1,
    'param' : {
        'ktetha' : 0.0,
        'tetha0' : 0.0,
        'kub'    : None,
        's0'     : None
    }
}

# Default flexible dihedral parameters
fparms = {
    'func' : 1,
    'param' : [{
        'delta' : 0.0,
        'kchi'  : 0.0,
        'n'     : 3
    }]
}

# Default stiff dihedral parameters
sparms = {
    'func' : 2,
    'param' : [{
        'kpsi'  : 0.0,
        'psi0'  : 0.0
    }]
}

# Default improper dihedral parameters
iparms = {
    'func' : 2,
    'param' : [{
        'kpsi'  : 0.0,
        'psi0'  : 0.0
    }]
}


def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(
                formatter_class=arg.ArgumentDefaultsHelpFormatter)

    #
    # Input Options
    #
    inp = parser.add_argument_group("Input Data")

    inp.add_argument(
            '-p',
            '--top',
            type=str,
            required=True,
            dest='TopFile',
            help='''Initial topology (.top format).'''
        )

    inp.add_argument(
            '-m',
            '--mol',
            type=str,
            required=True,
            dest='MolFile',
            help='''Molecule coordinates.'''
        )

    inp.add_argument(
            '-l',
            '--lj',
            type=str,
            dest='MixingRule',
            default='geom',
            choices = [ 'avg', 'geom' ],
            help='''LJ mixing rule to use.'''
        )

    #
    # Outut Options
    #
    out = parser.add_argument_group("Outut Data")

    out.add_argument(
            '-o',
            '--out',
            type=str,
            default=None,
            dest='OutFile',
            help='''Final topology.'''
        )

    args = parser.parse_args()
    Opts = vars(args)

    if Opts["MixingRule"] == "avg":
        Opts["MixingRule"] = avg_mixing
    elif Opts["MixingRule"] == "geom":
        Opts["MixingRule"] = geom_avg_mixing

    return Opts


def geom_avg_mixing(ei, ej, si, sj):
    '''
    Function compute LJ parameters for a pair of atoms, according to a
    geometric average mixing rule (see https://manual.gromacs.org/current/reference-manual/functions/nonbonded-interactions.html).

    Parameters
    ----------
    ei: float.
        Epsilon value for atom 1.
    ej: float.
        Epsilon value for atom 2.
    si: float.
        Sigma value for atom 1.
    sj: float.
        Sigma value for atom 2.

    Returns
    -------
    eij: float.
        Epsilon value for the LJ potential term.
    sij: float.
        Sigma value for the LJ potential term.
    '''

    eij = np.sqrt(ei * ej)
    sij = np.sqrt(si * sj)

    return eij, sij


def avg_mixing(ei, ej, si, sj):
    '''
    Function compute LJ parameters for a pair of atoms, according to an
    arithmetic average mixing rule (see https://manual.gromacs.org/current/reference-manual/functions/nonbonded-interactions.html).

    Parameters
    ----------
    ei: float.
        Epsilon value for atom 1.
    ej: float.
        Epsilon value for atom 2.
    si: float.
        Sigma value for atom 1.
    sj: float.
        Sigma value for atom 2.

    Returns
    -------
    eij: float.
        Epsilon value for the LJ potential term.
    sij: float.
        Sigma value for the LJ potential term.
    '''

    eij = np.sqrt(ei * ej)
    sij = 0.5 * (si + sj)

    return eij, sij


def add_terms(
        topfile,
        bds=None,
        angs=None,
        stiffs=None,
        imps=None,
        flexs=None,
        LJs=None,
        excls=None,
        mixing=geom_avg_mixing
    ):

    topobj = TOP(topfile)

    # Get nb parameters for each atom
    nb_params = {}
    for i, atom in enumerate(topobj.molecules[0].atoms):
        idx = atom.number
        attype = atom.atomtype

        for atype in topobj.atomtypes:
            if atype.atype == attype:
                eps = atype.gromacs['param']['lje']
                sig = atype.gromacs['param']['ljl']
                nb_params[i] = { 'epsilon' : eps, 'sigma' : sig }
                break

    # Add bonds
    counter = 1
    for bd in bds:
        ai, aj = bd
        bond = blocks.BondType('gromacs')
        bond.atom1 = topobj.molecules[0].atoms[ai]
        bond.atom2 = topobj.molecules[0].atoms[aj]
        bond.gromacs = bparms
        # bond.comment = f" {counter}"
        topobj.molecules[0].bonds.append(bond)
        counter += 1

    # Add angles
    for ang in angs:
        ai, aj, ak = ang
        angle = blocks.AngleType('gromacs')
        angle.atom1 = topobj.molecules[0].atoms[ai]
        angle.atom2 = topobj.molecules[0].atoms[aj]
        angle.atom3 = topobj.molecules[0].atoms[ak]
        angle.gromacs = aparms
        # angle.comment = f" {counter}"
        topobj.molecules[0].angles.append(angle)
        counter += 1

    # Add stiff dihedrals and impropers (those two go together)
    for stiff in stiffs:
        ai, aj, ak, al = stiff
        dih = blocks.ImproperType('gromacs')
        dih.atom1 = topobj.molecules[0].atoms[ai]
        dih.atom2 = topobj.molecules[0].atoms[aj]
        dih.atom3 = topobj.molecules[0].atoms[ak]
        dih.atom4 = topobj.molecules[0].atoms[al]
        dih.gromacs = sparms
        # dih.comment = f" {counter}"
        topobj.molecules[0].impropers.append(dih)
        counter += 1

    for imp in imps:
        try:
            ai, aj, ak, al = imp
        except:
            continue
        dih = blocks.ImproperType('gromacs')
        dih.atom1 = topobj.molecules[0].atoms[ai]
        dih.atom2 = topobj.molecules[0].atoms[aj]
        dih.atom3 = topobj.molecules[0].atoms[ak]
        dih.atom4 = topobj.molecules[0].atoms[al]
        dih.gromacs = sparms
        # dih.comment = f" {counter}"
        topobj.molecules[0].impropers.append(dih)
        counter += 1

    # Add flexible dihedrals
    for flex in flexs:
        ai, aj, ak, al = flex
        dih = blocks.DihedralType('gromacs')
        dih.atom1 = topobj.molecules[0].atoms[ai]
        dih.atom2 = topobj.molecules[0].atoms[aj]
        dih.atom3 = topobj.molecules[0].atoms[ak]
        dih.atom4 = topobj.molecules[0].atoms[al]
        dih.gromacs = fparms
        # dih.comment = f" {counter}"
        topobj.molecules[0].dihedrals.append(dih)
        counter += 1

    # Add LJs
    for k, v in LJs.items():
        for pair in v:
            ai, aj = pair
            ei = nb_params[ai]['epsilon']
            si = nb_params[ai]['sigma']

            ej = nb_params[aj]['epsilon']
            sj = nb_params[aj]['sigma']

            # eij = np.sqrt(ei * ej)
            # sij = 0.5 * (si + sj)
            eij, sij = mixing(ei, ej, si, sj)

            nb = blocks.InteractionType('gromacs')
            nb.atom1 = topobj.molecules[0].atoms[ai]
            nb.atom2 = topobj.molecules[0].atoms[aj]
            nb.gromacs = {
                    'param': {
                        'lje': None,
                        'ljl':None,
                        'lje14': eij,
                        'ljl14': sij,
                        'QQ': 1.000,
                        'qi': 0.000,
                        'qj': 0.000
                        },
                    'func': 2
                    }
            # nb.comment = f" {counter}"
            topobj.molecules[0].pairs.append(nb)
            counter += 1

    # Add exclusions
    for excl in excls:
        ai, aj = excl
        exc = blocks.Exclusion()
        exc.main_atom = topobj.molecules[0].atoms[ai]
        exc.other_atoms = [ topobj.molecules[0].atoms[aj] ]
        topobj.molecules[0].exclusions.append(exc)

    return topobj


def main():

    Opts = options()

    # Read in topology
    topfile = Opts['TopFile']
    data = list_intcoords(Opts['MolFile'])
    bds, angles, stiff, impdiheds, flex, LJs, excls, rings = data

    topobj = add_terms(
            topfile,
            bds,
            angles,
            stiff,
            impdiheds,
            flex,
            LJs,
            excls,
            mixing=Opts["MixingRule"]
        )

    if not Opts['OutFile']:
        outfile = '.'.join(Opts['TopFile'].split(".")[:-1]) + '.new.top'
        outcsv = '.'.join(Opts['TopFile'].split(".")[:-1]) + '.csv'
    else:
        outfile = Opts['OutFile']
        outcsv = '.'.join(Opts['OutFile'].split(".")[:-1]) + '.csv'

    topobj.write(outfile)

    df = pd.DataFrame({
            'Type' : [
                'Bonds',
                'Angles',
                'Stiff_dihedrals',
                'Impropers',
                'Flex_dihedrals',
                'LJ 1,4',
                'LJ 1,5',
                'LJ 1,6',
                'LJ 1,7',
                'LJ 1,n',
            ],

            'Number' : [
                bds.shape[0],
                angles.shape[0],
                stiff.shape[0],
                impdiheds.shape[0],
                flex.shape[0],
                LJs['1,4'].shape[0],
                LJs['1,5'].shape[0],
                LJs['1,6'].shape[0],
                LJs['1,7'].shape[0],
                LJs['other'].shape[0],
            ]
        })

    df['End'] = df['Number'].cumsum()
    df['Start'] = [ 1 ] + (df['End'] + 1).tolist()[:-1]
    df = df[['Type', 'Number', 'Start', 'End']]

    df.to_csv(outcsv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    return


if __name__ == '__main__':
    main()
    pass
