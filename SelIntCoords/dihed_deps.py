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


def _starlike_centre(graph, atoms):
    '''The atom bonded to the three others of a four-atom coordinate, or None.'''

    for c in atoms:
        if all(graph.has_edge(c, x) for x in atoms if x != c):
            return c
    return None


def ring_class_groups(graph, elements, orbit, coords):
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

    Returns
    -------
    groups: list of list
        Indices into `coords` that share one force constant.
    links: list of tuple
        (index of a starlike coordinate, index of a proper dihedral of the
        same ring) pairs: the first class is STARLIKE_FACTOR times the second.
    '''

    rings = [ tuple(sorted(r)) for r in nx.minimum_cycle_basis(graph) ]
    proper = defaultdict(list)
    starlike = defaultdict(list)
    for n, atoms in enumerate(np.asarray(coords).reshape(-1, 4)):
        atoms = [ int(a) for a in atoms ]
        centre = _starlike_centre(graph, atoms)
        if centre is not None:
            holders = [ r for r in rings if centre in r ]
        else:
            holders = [ r for r in rings if atoms[1] in r and atoms[2] in r and graph.has_edge(atoms[1], atoms[2]) ]
        if not holders:
            continue
        ring = max(holders, key=lambda r: (len(r), [ -a for a in r ]))
        key = (tuple(sorted(orbit[a] for a in ring)), any(elements[a] == 'H' for a in atoms))
        (proper if centre is None else starlike)[key].append(n)

    groups = list(proper.values()) + list(starlike.values())
    links = [ (idxs[0], proper[key][0]) for key, idxs in starlike.items() if key in proper ]
    return groups, links


if __name__ == '__main__':
    pass
