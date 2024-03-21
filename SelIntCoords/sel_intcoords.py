#!/usr/bin/env python

import sys
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import MDAnalysis as mda
from MDAnalysis.lib.util import unique_rows

try:
    import psi4
    from pyscf.symm.geom import detect_symm, symm_identical_atoms
except:
    import psi4
    from pyscf.symm.geom import detect_symm, symm_identical_atoms
finally:
    import psi4
    from pyscf.symm.geom import detect_symm, symm_identical_atoms

import warnings
warnings.filterwarnings("ignore")

au2ang = 0.5291771


def flatten(lst):
    '''
    Recursive function to flatten a nested list.

    Parameters
    ----------
    lst: list.
        Nested list to be flattened.

    Returns
    -------
    flattened: list.
        Flattened list.
    '''

    flattened = sum( ([x] if not isinstance(x, list)
                     else flatten(x) for x in lst), [] )

    return flattened


def get_sp2(u, alkyl=True, ether=False):
    '''
    Function to get indices of sp2 and sp3 atoms.

    Parameters
    ----------
    u: object.
        MDAnalysis Universe to be cropped.
    alkyl: bool.
        Whether side chains to crop are purely alkylic.
    ether: bool.
        Whether side chains to crop have an oxygen atom connected to the


    Returns
    -------
    sp2: np.ndarray.
        Indices of sp2 atoms.
    sp3: np.ndarray.
        Indices of sp3 atoms.
    '''

    # Get bonds
    try:
        bds = u.bonds.to_indices()
    except:
        u.guess_bonds()
        bds = u.bonds.to_indices()

    # Get connectivity, use -1 as a placeholder for empty valence
    conn = np.ones((len(u.atoms), 4)) * -1
    for bond in bds:
        at1, at2 = bond
        for j in np.arange(conn[at1].shape[0]):
            if conn[at1,j] == -1:
                conn[at1,j] = at2
                break

        for j in np.arange(conn[at2].shape[0]):
            if conn[at2,j] == -1:
                conn[at2,j] = at1
                break

    # Get heavy atoms
    heavy = np.where(u.atoms.types != "H")[0]

    # Get sp3 atoms
    sp3 = np.where(np.all(conn > -1, axis=1))[0]

    # Alkyls or ether chain
    if alkyl:
        allcheck = sp3
    elif ether:
        allcheck = np.concatenate([sp3, oxy])
    else:
        allcheck = sp3

    # Check all sp3 atoms
    keep = []
    delete = []
    for satat in allcheck:

        # check connectivity
        iconn = conn[satat]

        # filter H out from connected
        iconnheavy = iconn[np.in1d(iconn, heavy)]

        delete.append(satat)
        delete.extend(iconn[~np.in1d(iconn, heavy)])

    # Convert to int arrays
    keep = np.asarray(keep).astype(int)
    delete = np.asarray(delete).astype(int)

    # Get non sp3 atoms
    unsat = ~np.all(conn > -1, axis=1)

    # Set which saturated atoms to keep or delete
    unsat[keep] = True
    unsat[delete] = False
    tokeep = np.where(unsat)[0]
    sp2 = np.intersect1d(tokeep, heavy)

    return sp2, sp3


def dihedral(A, B, C, D):
    '''
    Function to compute the dihedral angle between the planes containing
    segments AB and CD.

    Parameters
    ----------
    A: np.array (N).
        First point.
    B: np.array (N).
        Second point.
    C: np.array (N).
        Third point.
    D: np.array (N).
        Fourth point.

    Returns
    -------
    dihedral: float.
        Dihedral angle defined by AB and CD (in degrees).
    '''

    ab = B - A
    bc = C - B
    cd = D - C
    
    c1 = np.cross(ab, bc)
    n1 = c1 / np.linalg.norm(c1)
    c2 = np.cross(bc, cd)
    n2 = c2 / np.linalg.norm(c2)
    m1 = np.cross(n1, bc)
    x = np.dot(n1, n2) 
    y = np.dot(m1, n2)
    dihedral = np.degrees(np.arctan2(y, x))

    return dihedral


def findPaths(G, u, n):
    '''
    Function to get n-order paths in graph G, starting from node u.

    Parameters
    ----------
    G: NetworkX Graph object.
        Graph for path search.
    u: int.
        Index of the starting node for the path search.
    n: int.
        Order of the paths to be found, that is number of bonds separating
        atoms involved in the interaction (1 for bonds, 2 for angles, 3 for
        dihedrals or 1,4 interactions, 4 for 1,5 interactions etc).

    Returns
    -------
    paths: list.
        List of n-order paths found in Graph G starting from node u.
    '''

    if n == 0:
        return [[u]]
    paths = [
        [u] + path for neighbor in G.neighbors(u) for path 
        in findPaths(G,neighbor,n-1) if u not in path
    ]

    return paths


def flip_bigger(arr, col1=0, col2=-1):
    '''
    Function to flip columns of array arr based on values in columns
    identified by index col1 and col2.

    Parameters
    ----------
    arr: np.ndarray (shape: (N, M)).
        Array whose columns are to be flipped.
    col1: int.
        Index of the first column.
    col2: int.
        Index of the second column.

    Returns
    -------
    arr: np.ndarray (shape: (N, M)).
        Array with flipped columns (no duplicate rows).
    '''

    arr = arr.copy()
    flipped = arr[:,col1] > arr[:,col2]
    arr[flipped] = np.fliplr(arr[flipped])
    arr = unique_rows(arr)

    return arr


def myround(x, base=5):
    '''
    Function to round a number x to the closest multiple of an arbitrary base.

    Parameters
    ----------
    x: float.
        number to be rounded.
    base: int or float.
        Base for the closest multiple.

    Returns
    -------
    rounded: int or float.
        Rounded number to the closest multiple of base.
    '''

    rounded = base * round(x / base)

    return rounded


def get_rings(G, bds):
    '''
    Function to get n-order paths in graph G, starting from node u.

    Parameters
    ----------
    G: NetworkX Graph object.
        Graph for path search.
    bds: np.ndarray.
        Array of bonds, in terms of indices of involved atoms.

    Returns
    -------
    rings: list of list.
        List of rings, where each ring is a list ot atom indices constituting
        the ring core.
    fullrings: list of list.
        List of rings, where each ring is a list ot atom indices constituting
        the ring core and its substituents.
    '''

    rings = sorted(nx.cycle_basis(G))
    fullrings = []
    for i, ring in enumerate(rings):

        ringbds = []
        for at in ring:
            idxs = np.where((bds == at).any(axis=1))[0]
            ringbds.extend(bds[idxs])

        ringbds = np.asarray(ringbds)
        ats = np.unique(ringbds.reshape(-1))
        other_rings = [ r for j, r in enumerate(rings[:]) if j != i ]
        delats = np.asarray([
            at for at in ats if at in flatten(other_rings) and at not in ring
        ])

        try:
            cleanidxs = np.where((ringbds != delats).all(axis=1))[0]
            cleanringbds = ringbds[cleanidxs]
            ringats = np.unique(cleanringbds.reshape(-1))
        except:
            cleanringbds = ringbds.copy()
            ringats = np.unique(cleanringbds.reshape(-1))

        fullrings.append(ringats)

    return rings, fullrings


def select_dihedrals(u, diheds, sp2, eq_hs, rings):
    '''
    Function to separate stiff and flexible dihedrals from the complete set
    based on the atoms making up a specific coordinate belonging to the same
    ring. Additionally, it selects the stiff dihedrals to keep (the ones at
    0 degrees), discarding the redundant set (the ones at 180 degrees).

    Parameters
    ----------
    u: object.
        MDAnalysis Universe, to get coordinates.
    diheds: np.ndarray (shape: (N, 4))
        Array of bonds, in terms of indices of involved atoms.
    sp2: np.ndarray.
        Indices of sp2 atoms.
    eq_hs: dict.
        Dictionary of equivalent H atoms.
    rings: dict.
        Dictionary of rings, divided in core atoms, and core plus substituents
        (full).

    Returns
    -------
    stiff: np.ndarray (shape: (N, 4)).
        Array of stiff proper dihedrals, in terms of indices of involved atoms.
    flex: np.ndarray (shape: (N, 4)).
        Array of flexible proper dihedrals, in terms of indices of involved
        atoms.
    '''

    # Still have to figure out how to use equivalent hydrogens to reduce or
    # group proper dihedrals.
    # Divide dihedrals into stiff and flexible
    # if the two central atoms belong to the same ring, then stiff
    # otherwise flexible. Only need to check proper dihedrals
    flex = []
    stiff = []
    for dihed in diheds:
        ati, atj, atk, atl = dihed
        for ring in rings['core']:
            if atj in ring and atk in ring and atj in sp2 and atk in sp2:
                stiff.append(dihed)
                break

        try:
            check = (dihed == np.asarray(stiff)).all(axis=1).any()
        except:
            check = False
        if not check:
            flex.append(dihed)

    flex = np.asarray(flex)
    stiff = np.asarray(stiff)

    # Select stiff dihedrals to only keep those that are 0
    # and remove the redundant ones at 180
    stiffk = []
    for dihed in stiff:
        A, B, C, D = u.atoms.positions[dihed]
        value = dihedral(A, B, C, D)
        if value < 15 and value > -15:
            stiffk.append(dihed)

    stiff = np.asarray(stiffk)

    return stiff, flex


def get_equivalent_hydrogens(u, G):
    '''
    Function to get equivalent H atoms in a molecule.

    Parameters
    ----------
    u: object.
        MDAnalysis Universe representing the molecule.
    G: NetworkX Graph object.
        Graph for path search.

    Returns
    -------
    eq_hs: dict.
        Dictionary of equivalent H atoms.
    '''

    hs = u.select_atoms("type H").indices
    hbds = []
    for h in hs:
        hbd = findPaths(G, h, 1)
        hbds.extend(hbd)

    hbds = np.fliplr(np.asarray(hbds))
    hbds = hbds[hbds[:,0].argsort()]
    ghbds = np.split(hbds[:,1], np.unique(hbds[:,0], return_index=True)[1][1:])

    eq_hs = {}
    for group in ghbds:
        for i, h in enumerate(group):
            eq_hs[h] = np.delete(group, i)

    return eq_hs


def get_equivalent_atoms(u):
    '''
    Function to get equivalent atoms in a molecule from symmetry point group.
    For this to work the geometry of the molecule needs to be aligned to the
    cartesian axes.

    Parameters
    ----------
    u: object.
        MDAnalysis Universe representing the molecule.

    Returns
    -------
    eq_ats: dict.
        Dictionary of equivalent atoms.
    '''

    # Symmetrise molecule to detect point group
    m = psi4.core.Molecule.from_arrays(
            elem=u.atoms.types,
            geom=u.atoms.positions,
            molecular_charge=0,
            molecular_multiplicity=1
        )
    m.symmetrize(0.01)
    m.update_geometry()

    coords, _, _, _, _ = m.to_arrays()
    # pg = m.find_highest_point_group()
    # point_group = pg.full_name()

    atoms = list(zip(u.atoms.types, coords * au2ang ))

    # Use symmetry to detect equivalence between atoms
    point_group, coq, axes = detect_symm(atoms)
    eqs = symm_identical_atoms(point_group, atoms)

    eq_ats = {}
    for group in eqs:
        for i, at in enumerate(group):
            eq_ats[at] = np.delete(group, i)

    return eq_ats


def list_intcoords(coordfile):
    '''
    Function to obtain a set of internal coordinates for force field 
    parameterisation from a geometry file.

    Parameters
    ----------
    coordfile: str.
        Name of the geometry file for which internal coordinates are required.

    Returns
    -------
    bds: np.ndarray.
        Array of bonds, in terms of indices of involved atoms.
    angles: np.ndarray.
        Array of angles, in terms of indices of involved atoms.
    stiff: np.ndarray (shape: (N, 4)).
        Array of stiff proper dihedrals, in terms of indices of involved atoms.
    impdiheds: np.ndarray.
        Array of improper dihedrals, in terms of indices of involved atoms.
    flex: np.ndarray (shape: (N, 4)).
        Array of flexible proper dihedrals, in terms of indices of involved
        atoms.
    LJs: dict.
        Dictionary of Lennard-Jones interactions, divided in 1,4, 1,5, 1,6,
        1,7, and other.
    excls: np.ndarray (shape: (N, 2)).
        Array of exclusions, to tell GROMACS which pairs of atoms to ignore in
        the automatic generation of LJ terms.
    rings: dict.
        Dictionary of rings, divided in core atoms, and core plus substituents
        (full).
    '''

    u = mda.Universe(coordfile, guess_bonds=True)

    # This does not replicate GView's Symmetryze behaviour
    # # align principal axes to cartesian reference frame
    # com = u.atoms.center_of_mass()
    # I = u.atoms.principal_axes()
    # M = np.linalg.inv(I)
    # newx = M.dot((u.atoms.positions.copy() - com).T).T
    # u.atoms.positions = newx

    # Get bonds
    bds = u.bonds.to_indices()

    # Build a graph representation to easily find all other
    # internal coordinates
    nodes = np.arange(bds.min(), bds.max() + 1)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(bds)

    # Get rings
    rings, frings = get_rings(G, bds)
    rings = {
        'core' : rings,
        'full' : frings
    }

    # Get equivalent atoms and Hs
    # This contains equivalences by symmetry
    eq_ats = get_equivalent_atoms(u)

    # This contains equivalences by connectivity, i.e. local symmetry
    eq_hs = get_equivalent_hydrogens(u, G)

    # Join the two equivalences
    # I also keep self equivalence because this makes it easier to
    # code the identification of equivalent internal coordinates
    eq = {}
    for k, v in sorted(eq_ats.items()):
        try:
            all_eqs = np.r_[ k, v, eq_hs[k] ]
            for h in v:
                all_eqs = np.r_[ all_eqs, eq_hs[h] ]
            eq[k] = np.sort(all_eqs)
        except:
            eq[k] = np.sort(np.r_[ k, v ])

    # Figure out how to use equivalence to reduce internal coordinates and to
    # set up dependencies

    # Get angles
    angles = []
    for node in nodes:
        angs = findPaths(G, node, 2)
        angles.extend(angs)

    angles = np.asarray(angles)

    # make first entry always less than third
    angles = flip_bigger(angles)

    # order by central atom and then by first
    angles = angles[angles[:,0].argsort()]
    angles = angles[angles[:,1].argsort(kind='mergesort')]

    # Get proper dihedrals
    diheds = []
    for node in nodes:
        dihs = findPaths(G, node, 3)
        diheds.extend(dihs)

    diheds = np.asarray(diheds)

    # make first entry always less than last
    diheds = flip_bigger(diheds, col1=1, col2=2)

    # order by central pair and then by first
    diheds = diheds[diheds[:,0].argsort()]
    diheds = diheds[diheds[:,1].argsort(kind='mergesort')]
    diheds = diheds[diheds[:,2].argsort(kind='mergesort')]

    # Get sp2 and sp3 heavy atoms to select stiff and look for impropers
    sp2, sp3 = get_sp2(u)

    # Separate stiff from flexible and select which stiff to keep
    stiff, flex = select_dihedrals(u, diheds, sp2, eq_hs, rings)

    # Get improper dihedrals
    impdiheds = []
    for idx in sp2:
        try:
            neighs = [ i for i in nx.all_neighbors(G, idx) ]
        except:
            neighs = []
        if len(neighs) > 2:
            impdiheds.append([ idx ] + neighs)

    impdiheds = np.asarray(impdiheds)

    # order by first atom
    # exception to deal with no impropers
    try:
        impdiheds = impdiheds[impdiheds[:,0].argsort()]
    except:
        pass

    # Get LJ pairs
    LJs14 = []
    LJs15 = []
    LJs16 = []
    LJs17 = []
    LJs = []
    for node in nodes:
        l = nx.single_source_shortest_path_length(G, node, cutoff=200)
        l14 = [ [ node, k ] for k, v in l.items() if v == 3 ]
        LJs14.extend(l14)
        l15 = [ [ node, k ] for k, v in l.items() if v == 4 ]
        LJs15.extend(l15)
        l16 = [ [ node, k ] for k, v in l.items() if v == 5 ]
        LJs16.extend(l16)
        l17 = [ [ node, k ] for k, v in l.items() if v == 6 ]
        LJs17.extend(l17)
        ll = [ [ node, k ] for k, v in l.items() if v >= 7 ]
        LJs.extend(ll)

    # exceptions to deal with no LJs of each type, but we reshape anyways
    # so that the empty array can be row stacked with others to gather all
    # LJs interactions
    try:
        LJs14 = flip_bigger(np.asarray(LJs14))
        LJs14 = LJs14[LJs14[:,0].argsort()]
    except:
        LJs14 = np.asarray(LJs14).reshape(-1, 2)

    try:
        LJs15 = flip_bigger(np.asarray(LJs15))
        LJs15 = LJs15[LJs15[:,0].argsort()]
    except:
        LJs15 = np.asarray(LJs15).reshape(-1, 2)

    try:
        LJs16 = flip_bigger(np.asarray(LJs16))
        LJs16 = LJs16[LJs16[:,0].argsort()]
    except:
        LJs16 = np.asarray(LJs16).reshape(-1, 2)

    try:
        LJs17 = flip_bigger(np.asarray(LJs17))
        LJs17 = LJs17[LJs17[:,0].argsort()]
    except:
        LJs17 = np.asarray(LJs17).reshape(-1, 2)

    try:
        LJs = flip_bigger(np.asarray(LJs))
        LJs = LJs[LJs[:,0].argsort()]
    except:
        LJs = np.asarray(LJs).reshape(-1, 2)

    LJs = {
        '1,4' : LJs14,
        '1,5' : LJs15,
        '1,6' : LJs16,
        '1,7' : LJs17,
        'other' : LJs,
    }

    # Make exclusions
    e = np.meshgrid(nodes, nodes, indexing="ij")
    excls = flip_bigger(np.vstack(list(map(np.ravel, e))).T)
    excls = excls[excls[:,0] != excls[:,1]]

    return bds, angles, stiff, impdiheds, flex, LJs, excls, rings, eq


if __name__ == '__main__':
    pass
