#!/usr/bin/env python

'''
make_top - complete a minimal GROMACS .top (atoms/atomtypes only) with every
bond, angle, dihedral, improper, 1-n LJ pair and exclusion derived from a
geometry, plus the Joyce $dependence file and a coordinate-count CSV.
'''

import os
import sys
import csv
import tempfile
import numpy as np
import MDAnalysis as mda
import pandas as pd
import argparse as arg
from itertools import product

from .top import *
from .blocks import *
from .sel_intcoords import list_intcoords, flip_bigger, flatten
from .perceive_types import (add_atomtype_comments, add_perception_args, atomtype_comments,
                             mark_missing_parameters, perceive_to_top)
from .mol_features import Features
from .flex_classes import classify_flex
from .ring_groups import RingGrouper, insert_headers
from .symmetry import topological_classes
from .pair_selection import append_commented_pairs, format_commented_pair, pair_ratios
from .dihed_deps import (
        get_ring_stiff_dihedral_equivalencies,
        integrate_ring_dihedral_equivalencies,
        find_stiff_dihedral_deps
    )

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
            default=None,
            dest='TopFile',
            help='''Initial topology (.top format). If omitted, the atom types are
                    perceived from the geometry (see the Atom type perception options).'''
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

    parser.add_argument(
            '--pair-ratio',
            type=float,
            default=0.8,
            dest='PairRatio',
            help='''A 1-n pair is written to the topology when it is separated by a
                    rotatable bond and, along the rotation, comes closer than this
                    fraction of its Lennard-Jones sigma; every other pair is
                    listed commented out at the end of the [ pairs ] section.'''
        )

    add_perception_args(parser)

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
        mixing=geom_avg_mixing,
        flex_params=None,
        pair_selection=None,
        pair_ratio=0.8
    ):
    '''
    Append bonded terms and 1-n Lennard-Jones pairs, all with zero
    parameters to be fitted by Joyce, to the first molecule of a topology.

    Parameters
    ----------
    topfile: str.
        Initial topology (atoms and atomtypes only).
    bds, angs, stiffs, imps, flexs: np.ndarray.
        Bonds, angles, stiff proper, improper and flexible proper dihedrals,
        as 0-based atom indices (from `sel_intcoords.list_intcoords`).
    LJs: dict.
        Lennard-Jones pairs by separation (1,4, 1,5, 1,6, 1,7, other).
    excls: np.ndarray (shape: (N, 2)).
        Pairs excluded from non-bonded interactions.
    mixing: callable.
        Mixing rule `f(eps_i, eps_j, sig_i, sig_j) -> (eps_ij, sig_ij)` for
        the pair parameters; `geom_avg_mixing` or `avg_mixing`.
    flex_params: list of dict.
        `{'delta': , 'kchi': , 'n': }` of each flexible dihedral; null
        parameters with `n = 3` when omitted.
    pair_selection: tuple, optional.
        `(graph, coords, rotors)`: bond graph, coordinates (Angstrom) and
        rotatable bonds. When given, only the pairs separated by a rotor whose
        closest approach is below `pair_ratio` times their sigma are added
        (`pair_selection.pair_ratios`); the others are kept in
        `topobj.commented_pairs` for `append_commented_pairs`. When omitted,
        every pair is added.
    pair_ratio: float.
        Largest `r_min / sigma` of a pair that is kept.

    Returns
    -------
    topobj: TOP.
        The completed topology, ready to be written.
    '''

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
    for c, flex in enumerate(flexs):
        ai, aj, ak, al = flex
        dih = blocks.DihedralType('gromacs')
        dih.atom1 = topobj.molecules[0].atoms[ai]
        dih.atom2 = topobj.molecules[0].atoms[aj]
        dih.atom3 = topobj.molecules[0].atoms[ak]
        dih.atom4 = topobj.molecules[0].atoms[al]
        dih.gromacs = fparms if flex_params is None else {'func': 1, 'param': [dict(flex_params[c])]}
        # dih.comment = f" {counter}"
        topobj.molecules[0].dihedrals.append(dih)
        counter += 1

    # Add LJs. Only the pairs whose distance changes with a torsion and that
    # come close along it are kept; the others are listed, commented out, at
    # the end of the [ pairs ] section
    pairs = []
    for k, v in LJs.items():
        for pair in v:
            ai, aj = pair
            eij, sij = mixing(nb_params[ai]['epsilon'], nb_params[aj]['epsilon'],
                              nb_params[ai]['sigma'], nb_params[aj]['sigma'])
            pairs.append((k, int(ai), int(aj), eij, sij))

    ratios = {}
    if pair_selection is not None:
        graph, coords, rotors = pair_selection
        sigma = { (min(ai, aj), max(ai, aj)) : sij for k, ai, aj, eij, sij in pairs }
        ratios = pair_ratios(graph, coords, rotors, LJs, sigma)

    topobj.pair_counts = { k : 0 for k in LJs }
    topobj.commented_pairs = []
    for k, ai, aj, eij, sij in pairs:
        ratio = ratios.get((min(ai, aj), max(ai, aj)))
        if pair_selection is not None and (ratio is None or ratio >= pair_ratio):
            topobj.commented_pairs.append((ratio, ai, aj, eij, sij, k))
            continue

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
        topobj.molecules[0].pairs.append(nb)
        topobj.pair_counts[k] += 1
        counter += 1

    # Add exclusions
    for excl in excls:
        ai, aj = excl
        exc = blocks.Exclusion()
        exc.main_atom = topobj.molecules[0].atoms[ai]
        exc.other_atoms = [ topobj.molecules[0].atoms[aj] ]
        topobj.molecules[0].exclusions.append(exc)

    return topobj


def _atom_classes(eqs):
    '''
    Class label of each atom when the equivalences `eqs` partition the atoms
    (each atom equivalent to itself, and every equivalent of an atom has the
    same equivalents), None otherwise.
    '''

    classes = {}
    for i, others in eqs.items():
        members = set(int(j) for j in others) | { int(i) }
        for j in members:
            if j not in eqs or (set(int(x) for x in eqs[j]) | { int(j) }) != members:
                return None
        classes[i] = min(members)
    return classes


def find_deps(coords, eqs):
    '''
    Find internal coordinates that are equivalent by symmetry.

    Parameters
    ----------
    coords: np.ndarray (shape: (N, k)).
        Internal coordinates as atom indices.
    eqs: dict.
        Maps each atom index to the array of atoms equivalent to it.

    Returns
    -------
    equivs: np.ndarray (shape: (M, 2)).
        Pairs of indices into `coords`, smaller first, of coordinates that
        map onto each other under the atom equivalences; a single empty
        row if there are none.
    '''

    eqcoords_all = []
    classes = _atom_classes(eqs)
    if classes is not None:
        # equivalent atoms form a partition: coordinates are equivalent when
        # their atoms are, position by position, in the same class (the same
        # as testing every substitution of an atom by an equivalent one, at a
        # cost that does not grow with the size of the classes)
        groups = {}
        for c, coord in enumerate(coords):
            groups.setdefault(tuple(classes[i] for i in coord), []).append(c)
        for c, coord in enumerate(coords):
            idxs = [ i for i in groups[tuple(classes[i] for i in coord)] if i != c ]
            if idxs:
                eqcoords_all.append(np.c_[ np.ones(len(idxs), dtype=int) * c, idxs ])
    else:
        for c, coord in enumerate(coords):

            transl = [ eqs[i] for i in coord ]

            # do all possible partial substitutions
            subs = np.asarray([ list(i) for i in product(*transl) ])

            # check if any of the generated coordinates is in the initial set
            mask = (coords[:,None] == subs).all(axis=-1).any(axis=-1)

            # get their indices in the set of coordinates
            idxs = np.where(mask)[0]

            # combine them with the index of current coordinate being checked
            eqcoords = np.c_[ np.ones_like(idxs) * c, idxs ]

            # remove self-equivalent coordinates
            eqcoords = eqcoords[eqcoords[:,0] != eqcoords[:,1]]
            eqcoords_all.append(eqcoords)

    try:
        equivs = np.vstack(eqcoords_all)
    except ValueError:
        equivs = np.empty((0, 2), dtype=int)

    # smaller coordinate first, and ordered by first coord
    # deal with no equivalences found
    try:
        equivs = flip_bigger(equivs)
        equivs = equivs[equivs[:,1].argsort()]
        equivs = equivs[equivs[:,0].argsort(kind='mergesort')]
    except:
        equivs = np.empty((1, 2))

    return equivs


def main():
    '''Entry point of the `make_top` console script.'''
    Opts = options()

    # Names of the output files
    if Opts['OutFile']:
        out_stem = '.'.join(Opts['OutFile'].split(".")[:-1])
        outfile = Opts['OutFile']
    else:
        src = Opts['TopFile'] or Opts['MolFile']
        out_stem = '.'.join(src.split(".")[:-1])
        # a topology made from the geometry has no older one to tell it from
        outfile = out_stem + ('.new.top' if Opts['TopFile'] else '.top')
    outcsv = out_stem + '.csv'
    outdeps = out_stem + '_deps.dat'

    # Read in topology, or perceive the atom types from the geometry: the
    # starting topology is then a by-product, only its report is kept
    topfile = Opts['TopFile']
    tmpdir = None
    assign = []
    type_comments = {}
    if topfile is None:
        tmpdir = tempfile.TemporaryDirectory()
        topfile, assign = perceive_to_top(
                Opts['MolFile'],
                Opts['ff'],
                Opts['gmx_top'],
                out=os.path.join(tmpdir.name, 'types.top'),
                class_name=not Opts['ff_type_names'],
                resname=Opts['resname'],
                report=out_stem + '.types',
                numeric_placeholders=True,
                marked_out=out_stem + '_types.top'
            )
        topfile = str(topfile)
        type_comments = atomtype_comments(topfile)
        print('Atom types perceived: %d atoms, %d by analogy, %d to check (see %s)' % (
            len(assign), sum(a.analogue for a in assign),
            sum(bool(a.flags or a.alternatives) for a in assign), out_stem + '.types'))
        missing = sorted({a.element for a in assign if a.source == 'placeholder'})
        if missing:
            print('No force-field parameters for %s: %s lists them as FILL_ME; fill in sigma and '
                  'epsilon there and run make_top again with -p %s to get the pairs.' % (
                      ', '.join(missing), outfile,
                      out_stem + '_types.top'))

    # Get internal coordinates and other information from geometry
    data = list_intcoords(Opts['MolFile'])
    bds, angles, stiff, impdiheds, flex, LJs, excls, rings, eq = data

    # Treat each rotatable bond as done by hand: methyl rotors take literature
    # parameters, single bonds between sp2 atoms get one scanned dihedral,
    # double bonds get stiff dihedrals
    u = mda.Universe(Opts['MolFile'])
    feats = Features([ str(t).capitalize() for t in u.atoms.types ], u.atoms.positions, bds)
    classes = topological_classes(feats.el, bds)
    same = { i : np.where(classes == classes[i])[0] for i in range(feats.n) }
    plan = classify_flex(feats, flex, same)
    stiff = np.asarray(stiff).reshape(-1, 4)
    stiff = np.vstack([ stiff, plan.double ])
    flex = plan.flex

    # Group bonds, angles and stiff dihedrals by ring system, so that the
    # dependencies refer to the reordered coordinates
    grouper = RingGrouper(feats)
    order, bd_labels = grouper.group(bds)
    bds = bds[order]
    order, ang_labels = grouper.group(angles)
    angles = angles[order]
    order, stiff_labels = grouper.group(stiff)
    stiff = stiff[order]
    imp_labels = [ 'starlike' ] * len(impdiheds)
    labels = bd_labels + ang_labels + stiff_labels + imp_labels + plan.labels
    headers = { n + 1 : label for n, label in enumerate(labels) }

    # NEW: Get ring-based stiff dihedral equivalencies
    enhanced_eq, ring_dih_info = integrate_ring_dihedral_equivalencies(eq, stiff, rings)

    # Get specific stiff dihedral dependencies for ring dihedrals
    stiff_ring_deps = find_stiff_dihedral_deps(stiff, rings)

    # Create Topology (unchanged)
    topobj = add_terms(
            topfile,
            bds,
            angles,
            stiff,
            impdiheds,
            flex,
            LJs,
            excls,
            mixing=Opts["MixingRule"],
            flex_params=plan.params,
            pair_selection=(feats.graph, feats.pos, plan.rotors),
            pair_ratio=Opts['PairRatio']
        )

    if tmpdir is not None:
        tmpdir.cleanup()

    # Write topology (unchanged)
    topobj.write(outfile)
    insert_headers(outfile, headers)
    commented = topobj.commented_pairs
    reason = [
            '; kept: 1,5 or farther apart, with a rotatable bond between the two atoms, and closer than',
            '; %.2f sigma of the pair at the closest approach along that rotation (--pair-ratio); the others are commented below' % Opts['PairRatio']
        ]
    append_commented_pairs(outfile, [ format_commented_pair(ai, aj, eij, sij, ratio, k) for ratio, ai, aj, eij, sij, k in commented ], reason)
    if type_comments:
        add_atomtype_comments(outfile, type_comments)
    missing = {a.element for a in assign if a.source == 'placeholder'}
    if missing:
        mark_missing_parameters(outfile, missing)

    # Create summary dataframe (unchanged)
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
                topobj.pair_counts['1,4'],
                topobj.pair_counts['1,5'],
                topobj.pair_counts['1,6'],
                topobj.pair_counts['1,7'],
                topobj.pair_counts['other'],
            ]
        })

    df['End'] = df['Number'].cumsum()
    df['Start'] = [ 1 ] + (df['End'] + 1).tolist()[:-1]
    df = df[['Type', 'Number', 'Start', 'End']]

    df.to_csv(outcsv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Find equivalent coordinates using enhanced equivalencies
    # For atoms, use the original eq dictionary
    bds_eqs = find_deps(bds, eq)  # Original function for bonds
    ang_eqs = find_deps(angles, eq)  # Original function for angles

    # For stiff dihedrals, combine the original symmetry-based equivalencies
    # with the new ring-based equivalencies
    stiff_eqs = find_deps(stiff, eq)  # Original symmetry-based equivalencies

    # Add ring-based stiff dihedral dependencies
    if stiff_ring_deps.size > 0:
        # Convert stiff_ring_deps to the same format as other equivalencies
        # (add 1 to convert from 0-based to 1-based indexing if needed)
        if stiff_eqs.size > 0:
            stiff_eqs = np.vstack([stiff_eqs, stiff_ring_deps + 1])
        else:
            stiff_eqs = stiff_ring_deps + 1

    imp_eqs = find_deps(impdiheds, eq)  # Original function for impropers
    flex_eqs = find_deps(flex, eq)      # Original function for flexible dihedrals

    # Rest of the function remains the same...
    # Renumber each set of coordinate to follow global numbering
    bds_eqs += 1
    ang_eqs += 1 + (df[df["Type"] == "Bonds"]["End"]).values
    stiff_eqs += 1 + (df[df["Type"] == "Angles"]["End"]).values
    imp_eqs += 1 + (df[df["Type"] == "Stiff_dihedrals"]["End"]).values
    flex_eqs += 1 + (df[df["Type"] == "Impropers"]["End"]).values

    # Merge dependencies
    arrs = [
        bds_eqs,
        ang_eqs,
        stiff_eqs,
        imp_eqs,
        flex_eqs
    ]

    for arr in arrs:
        if arr.shape[0] > 1:
            try:
                deps = np.r_[ deps, arr ]
            except:
                deps = arr

    # Filter out redundancies (unchanged)
    deps_dict = {}
    for dep in deps:
        done = False
        for k, v in deps_dict.items():
            if (dep[1] in v or dep[0] in v) or (dep[0] == k or dep[1] == k):
                deps_dict[k].extend(dep)
                done = True

        if not done:
            deps_dict[dep[0]] = [ dep[1] ]

    # Remove self-equivalence
    for k, v in deps_dict.items():
        v = np.asarray(v, dtype=int)
        v = v[v != k]
        deps_dict[k] = np.sort(np.unique(v))

    # Write dependencies in Joyce format
    with open(outdeps, "w") as f:
        f.write("$dependence 1.2\n")
        for k, v in deps_dict.items():
            for vv in v:
                f.write("%5d = %5d*1.d0 ;\n" % (vv, k))
        f.write("$end")

    return


if __name__ == '__main__':
    main()
