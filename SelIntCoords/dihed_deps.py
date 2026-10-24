#!/usr/bin/env python

'''
dihed_deps - group the stiff (ring) proper dihedrals that must share a force
constant, as Joyce $dependence entries.
'''

from collections import defaultdict

import networkx as nx
import numpy as np

# out-of-plane (starlike) impropers at a ring atom are stiffer than the ring torsions
STARLIKE_FACTOR = 5


def get_ring_stiff_dihedral_equivalencies(stiff, rings):
    '''
    Function to identify equivalent stiff dihedral angles that belong to the same ring.
    Dihedrals in the same ring should have similar force constants due to structural
    constraints and can be grouped together for fitting purposes.

    Parameters
    ----------
    stiff: np.ndarray (shape: (N, 4))
        Array of stiff proper dihedrals, in terms of indices of involved atoms.
    rings: dict
        Dictionary of rings, divided in core atoms, and core plus substituents (full).

    Returns
    -------
    ring_dihedral_groups: dict
        Dictionary where keys are ring indices and values are lists of dihedral 
        indices that belong to that ring.
    stiff_dihedral_equivalencies: dict
        Dictionary where keys are dihedral indices (position in stiff array) and 
        values are lists of equivalent dihedral indices within the same ring.
    '''
    
    # Initialize output dictionaries
    ring_dihedral_groups = {}
    stiff_dihedral_equivalencies = {}
    
    # Group dihedrals by ring membership
    for ring_idx, ring in enumerate(rings['core']):
        ring_dihedrals = []
        
        # Check each stiff dihedral to see if it belongs to this ring
        for dih_idx, dihed in enumerate(stiff):
            ati, atj, atk, atl = dihed
            
            # A dihedral belongs to a ring if its central bond (j-k) is in the ring
            # and all four atoms are part of the ring system
            if (atj in ring and atk in ring and 
                ati in ring and atl in ring):
                ring_dihedrals.append(dih_idx)
        
        # Only store rings that have dihedrals
        if ring_dihedrals:
            ring_dihedral_groups[ring_idx] = ring_dihedrals
    
    # Create equivalency mappings
    for ring_idx, dihedral_indices in ring_dihedral_groups.items():
        # Each dihedral in the ring is equivalent to all others in the same ring
        for dih_idx in dihedral_indices:
            # Get all other dihedrals in the same ring (excluding self)
            equivalent_dihedrals = [idx for idx in dihedral_indices if idx != dih_idx]
            stiff_dihedral_equivalencies[dih_idx] = equivalent_dihedrals
    
    return ring_dihedral_groups, stiff_dihedral_equivalencies


def integrate_ring_dihedral_equivalencies(eq, stiff, rings):
    '''
    Function to integrate ring-based stiff dihedral equivalencies into the existing
    equivalency dictionary. This extends the current symmetry-based equivalencies
    with structural equivalencies for ring dihedrals.

    Parameters
    ----------
    eq: dict
        Existing dictionary of equivalent atoms from symmetry analysis.
    stiff: np.ndarray (shape: (N, 4))
        Array of stiff proper dihedrals, in terms of indices of involved atoms.
    rings: dict
        Dictionary of rings, divided in core atoms, and core plus substituents (full).

    Returns
    -------
    enhanced_eq: dict
        Enhanced equivalency dictionary that includes ring dihedral equivalencies.
    ring_dihedral_info: dict
        Additional information about ring dihedral groupings for reference.
    '''
    
    # Get ring dihedral equivalencies
    ring_groups, dih_equivalencies = get_ring_stiff_dihedral_equivalencies(stiff, rings)
    
    # Create enhanced equivalency dictionary
    enhanced_eq = eq.copy()
    
    # Create a special key structure for dihedral equivalencies
    # We'll use negative indices to distinguish dihedral equivalencies from atom equivalencies
    dihedral_eq_offset = -1000  # Use negative numbers to avoid conflicts
    
    for dih_idx, equiv_dihedrals in dih_equivalencies.items():
        # Create a unique key for this dihedral equivalency group
        dih_key = dihedral_eq_offset - dih_idx
        enhanced_eq[dih_key] = np.array([dihedral_eq_offset - idx for idx in equiv_dihedrals])
    
    # Prepare additional information for reference
    ring_dihedral_info = {
        'ring_groups': ring_groups,
        'dihedral_equivalencies': dih_equivalencies,
        'offset': dihedral_eq_offset,
        'explanation': {
            'ring_groups': 'Maps ring index to list of dihedral indices in that ring',
            'dihedral_equivalencies': 'Maps dihedral index to equivalent dihedral indices',
            'offset': 'Offset used to distinguish dihedral keys from atom keys in enhanced_eq'
        }
    }
    
    return enhanced_eq, ring_dihedral_info


def find_stiff_dihedral_deps(stiff, rings):
    '''
    Simplified function specifically for finding dependencies among stiff dihedrals
    that belong to the same ring. This function creates dependency relationships
    similar to the existing find_deps function but specifically for ring dihedrals.

    Parameters
    ----------
    stiff: np.ndarray (shape: (N, 4))
        Array of stiff proper dihedrals, in terms of indices of involved atoms.
    rings: dict
        Dictionary of rings, divided in core atoms, and core plus substituents (full).

    Returns
    -------
    dihedral_deps: np.ndarray (shape: (M, 2))
        Array of dihedral dependencies where first column is the dependent coordinate
        and second column is the independent coordinate (both are indices into stiff array).
    '''
    
    ring_groups, _ = get_ring_stiff_dihedral_equivalencies(stiff, rings)
    
    dihedral_deps = []
    
    # For each ring with multiple dihedrals, create dependencies
    for ring_idx, dihedral_indices in ring_groups.items():
        if len(dihedral_indices) > 1:
            # Use the first dihedral as the independent coordinate
            independent = dihedral_indices[0]
            
            # All other dihedrals in the ring depend on the first one
            for dependent in dihedral_indices[1:]:
                dihedral_deps.append([dependent, independent])
    
    return np.array(dihedral_deps) if dihedral_deps else np.empty((0, 2))


def smallest_rings(graph):
    '''Rings of the graph (a smallest set), each a sorted tuple of atoms.'''

    return [ tuple(sorted(r)) for r in nx.minimum_cycle_basis(graph) ]


def _starlike_centre(graph, atoms):
    '''The atom bonded to the three others of a four-atom coordinate, or None.'''

    for c in atoms:
        if all(graph.has_edge(c, x) for x in atoms if x != c):
            return c
    return None


def ring_of_coordinate(graph, rings, atoms):
    '''
    The ring a stiff coordinate belongs to, or None: the ring that holds the
    central bond of a proper dihedral, or the centre of a starlike improper;
    the larger ring, then the lowest atoms, where two rings share the bond.
    '''

    atoms = [ int(a) for a in atoms ]
    centre = _starlike_centre(graph, atoms)
    if centre is not None:
        holders = [ r for r in rings if centre in r ]
    else:
        holders = [ r for r in rings if atoms[1] in r and atoms[2] in r and graph.has_edge(atoms[1], atoms[2]) ]
    if not holders:
        return None
    return max(holders, key=lambda r: (len(r), [ -a for a in r ]))


def ring_class_groups(graph, elements, orbit, coords, rings=None):
    '''
    Stiff coordinates (proper dihedrals and starlike impropers) that share a
    force constant because they sit in the same ring.

    The six torsions of a planar ring are not independent at the harmonic
    level (only the collective out-of-plane stiffness is fixed by the
    Hessian), so fitted separately they split it with arbitrary signs, and
    force constants below zero appear. One parameter per ring removes this.
    A coordinate belongs to the ring that holds the central bond of a proper
    dihedral, or the centre of a starlike improper (the larger ring, then the
    lowest atoms, where two rings share the bond); rings equivalent by the
    graph symmetry share their parameter. Coordinates with hydrogen are kept
    apart from those without, and the starlike ones are tied to the proper
    dihedrals of their ring with a factor (STARLIKE_FACTOR).

    Parameters
    ----------
    graph: nx.Graph
        Bond graph, nodes are atom indices.
    elements: sequence of str
        Element of each atom.
    orbit: sequence of int
        Label of the class of equivalent atoms of each atom.
    coords: np.ndarray (shape: (N, 4))
        Proper dihedrals and starlike impropers, as atom indices.
    rings: list of tuple, optional
        Smallest rings of the graph (found if not given).

    Returns
    -------
    groups: list of list
        Indices into `coords` that share one force constant.
    links: list of tuple
        (index of a starlike coordinate, index of a proper dihedral of the
        same ring) pairs: the first class is STARLIKE_FACTOR times the second.
    '''

    if rings is None:
        rings = smallest_rings(graph)
    proper = defaultdict(list)
    starlike = defaultdict(list)
    for n, atoms in enumerate(np.asarray(coords).reshape(-1, 4)):
        ring = ring_of_coordinate(graph, rings, atoms)
        if ring is None:
            continue
        atoms = [ int(a) for a in atoms ]
        key = (tuple(sorted(orbit[a] for a in ring)), any(elements[a] == 'H' for a in atoms))
        (proper if _starlike_centre(graph, atoms) is None else starlike)[key].append(n)

    groups = list(proper.values()) + list(starlike.values())
    links = [ (idxs[0], proper[key][0]) for key, idxs in starlike.items() if key in proper ]
    return groups, links


def tie_dependencies(symmetry_pairs, ring_pairs, links, first_stiff):
    '''
    Dependent coordinates, each tied once to the lowest index of its class.

    Parameters
    ----------
    symmetry_pairs: np.ndarray (shape: (N, 2))
        Pairs of coordinate numbers (global, 1-based) that are equivalent by symmetry.
    ring_pairs: np.ndarray (shape: (M, 2))
        Pairs of coordinate numbers that share a force constant because they sit in the same ring.
    links: list of tuple
        (starlike index, proper dihedral index) from `ring_class_groups`, relative to `first_stiff`.
    first_stiff: int
        Global number of coordinate 0 of the stiff and starlike set.

    Returns
    -------
    ties: list of tuple
        (dependent, reference, factor, reason), sorted by reference, with reason 'symmetry' (the two are
        equivalent by symmetry), 'ring' (same ring, or equivalent rings) or 'starlike' (factor times the
        torsions of the ring).
    '''

    def union_find(pairs):
        parent = {}

        def root(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            a, b = root(int(a)), root(int(b))
            if a != b:
                parent[max(a, b)] = min(a, b)
        return root, parent

    symmetry_root, _ = union_find(symmetry_pairs)
    root, parent = union_find(list(map(tuple, symmetry_pairs)) + list(map(tuple, ring_pairs)))
    for star, _ in links:
        root(star + first_stiff)                       # a starlike coordinate alone in its class is still tied
    classes = {}
    for x in sorted(parent):
        classes.setdefault(root(x), []).append(x)

    # a starlike class is tied to the proper dihedrals of its ring in a fixed ratio
    linked = {}
    for star, proper in links:
        star, proper = root(star + first_stiff), root(proper + first_stiff)
        for x in classes.get(star, [ star ]):
            linked[x] = proper

    ties = []
    for ref, members in classes.items():
        for x in members:
            if x in linked:
                ties.append((x, linked[x], STARLIKE_FACTOR, 'starlike'))
            elif x != ref:
                ties.append((x, ref, 1, 'symmetry' if symmetry_root(x) == symmetry_root(ref) else 'ring'))
    return sorted(ties, key=lambda tie: (tie[1], tie[0]))


if __name__ == '__main__':
    pass
