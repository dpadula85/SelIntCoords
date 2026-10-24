#!/usr/bin/env python

'''
symmetry - atoms equivalent by symmetry: of a geometry (point group) or of its bond graph.

A geometry obtained from an optimisation is symmetric only to within the
convergence of the optimiser, so the symmetry operations are searched
with a tolerance (0.01 Angstrom by default, the tolerance GaussView uses to
symmetrise). An operation is an orthogonal map O about the centroid such
that every atom goes onto an atom of the same element. Every such map is
fixed by where it sends two atoms that are not collinear with the centroid,
and by whether it is proper or improper, so the candidates are enumerated
from the images of two reference atoms and each is checked on the whole
molecule. Two atoms are equivalent when some operation maps one onto the
other: the equivalence classes are the orbits of the group of the operations
found, which needs no point-group table and no external program.

The bond graph has symmetries of its own (`graph_equivalences`): the two methyls of a gem-dimethyl or the two
halves of a molecule are equivalent whatever their conformations, which is how equivalent atoms are told by
hand. The geometry plays no part, so no tolerance is needed.
'''

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

# the sine of the angle between the two reference atoms, seen from the
# centroid, below which they count as collinear
COLLINEAR = 1e-3


def _permutation(x: np.ndarray, labels: np.ndarray, op: np.ndarray, trees, tol: float):
    '''
    Permutation of the atoms induced by `op` (y = x @ op.T), or None if some
    atom does not land on an atom of its own element within `tol`.
    '''

    y = x @ op.T
    perm = np.full(len(x), -1)
    for lab, (idx, tree) in trees.items():
        d, k = tree.query(y[idx])
        if d.max() > tol:
            return None
        perm[idx] = idx[k]
    # a map onto the atoms must be one-to-one
    if len(np.unique(perm)) != len(perm):
        return None
    return perm


def symmetry_permutations(elements: List[str], coords: np.ndarray, tol: float = 0.01) -> List[np.ndarray]:
    '''
    Atom permutations induced by the symmetry operations of a geometry.

    Parameters
    ----------
    elements: list of str.
        Element (or any label) of each atom; only atoms of equal label are
        exchanged.
    coords: np.ndarray (shape: (N, 3)).
        Coordinates in Angstrom.
    tol: float.
        Largest distance, in Angstrom, between an atom moved by an operation
        and the atom it is taken to.

    Returns
    -------
    perms: list of np.ndarray.
        One permutation per operation found, the identity included.
    '''

    labels = np.asarray(elements)
    n = len(labels)
    x = np.asarray(coords, dtype=float)
    x = x - x.mean(axis=0)
    ident = np.arange(n)
    if n < 2:
        return [ident]

    trees = {}
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        trees[lab] = (idx, cKDTree(x[idx]))

    r = np.linalg.norm(x, axis=1)
    perms = [ident]
    seen = {tuple(ident)}

    def add(perm):
        if perm is None:
            return
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            perms.append(perm)

    # reference atoms: the farthest from the centroid, and the one making the
    # largest angle with it
    a = int(np.argmax(r))
    if r[a] < tol:
        return perms
    sin = np.linalg.norm(np.cross(x[a], x), axis=1) / (r[a] * np.maximum(r, 1e-12))
    b = int(np.argmax(sin))

    if sin[b] < COLLINEAR:
        # linear molecule: rotations about the axis map every atom onto itself,
        # what is left is the inversion and the mirror plane perpendicular to the axis
        axis = x[a] / r[a]
        for op in (-np.eye(3), np.eye(3) - 2.0 * np.outer(axis, axis)):
            add(_permutation(x, labels, op, trees, tol))
        return perms

    ref = np.column_stack([x[a], x[b], np.cross(x[a], x[b])])
    ref_inv = np.linalg.inv(ref)
    dot_ab = x[a] @ x[b]

    cand_a = np.where((labels == labels[a]) & (np.abs(r - r[a]) < tol))[0]
    cand_b = np.where((labels == labels[b]) & (np.abs(r - r[b]) < tol))[0]
    for k in cand_a:
        for l in cand_b:
            if k == l or abs(x[k] @ x[l] - dot_ab) > tol * (r[a] + r[b]):
                continue
            cross = np.cross(x[k], x[l])
            for sign in (1.0, -1.0):
                op = np.column_stack([x[k], x[l], sign * cross]) @ ref_inv
                if np.abs(op @ op.T - np.eye(3)).max() > 5.0 * tol / min(r[a], r[b]):
                    continue
                add(_permutation(x, labels, op, trees, tol))
    return perms


def equivalent_atoms(elements: List[str], coords: np.ndarray, tol: float = 0.01) -> Dict[int, np.ndarray]:
    '''
    Atoms equivalent by symmetry.

    Returns
    -------
    eq_ats: dict.
        Atom index (0-based) -> sorted array of the other atoms it is
        equivalent to (empty when it has none). Every atom is a key.
    '''

    n = len(elements)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for perm in symmetry_permutations(elements, coords, tol):
        for i, j in enumerate(perm):
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    classes: Dict[int, List[int]] = {}
    for i in range(n):
        classes.setdefault(find(i), []).append(i)

    eq_ats = {}
    for members in classes.values():
        for i in members:
            eq_ats[i] = np.array([j for j in members if j != i], dtype=int)
    return eq_ats


def topological_classes(elements: List[str], bonds: np.ndarray) -> np.ndarray:
    '''
    Class of every atom by its bonded environment alone.

    Atoms start in the class of their element and are split by the classes of
    their neighbours until nothing changes (Morgan refinement), so two atoms
    share a class when they cannot be told apart by the molecular graph. This
    is the equivalence a user applies by chemical sense to a geometry that is
    not symmetric to within the tolerance of `symmetry_permutations`.

    Returns
    -------
    classes: np.ndarray (shape: (N,)).
        Integer class label of each atom.
    '''

    n = len(elements)
    nbrs: List[List[int]] = [[] for _ in range(n)]
    for i, j in np.asarray(bonds, dtype=int).reshape(-1, 2):
        nbrs[i].append(int(j))
        nbrs[j].append(int(i))

    labels = {el: k for k, el in enumerate(sorted(set(elements)))}
    classes = np.array([labels[el] for el in elements])
    n_classes = len(set(classes.tolist()))
    for _ in range(n):
        sigs = [(int(classes[i]), tuple(sorted(int(classes[j]) for j in nbrs[i]))) for i in range(n)]
        ids = {sig: k for k, sig in enumerate(sorted(set(sigs)))}
        classes = np.array([ids[sig] for sig in sigs])
        if len(ids) == n_classes:
            break
        n_classes = len(ids)
    return classes


# ----------------------------------------------------------------------
# symmetry of the bond graph (individualisation and refinement)
#
# Two atoms are equivalent when an automorphism of the bond graph that keeps the atom colours (element, or element
# and atom type) sends one onto the other. This is what is done by hand (the two methyls of a gem-dimethyl, the
# two halves of a molecule, whatever the conformation) and does not depend on a geometry being symmetric within a
# tolerance. Colour refinement splits the atoms by the colours of their neighbours; an automorphism is then
# searched by singling out atoms one at a time, without listing the group.
# ----------------------------------------------------------------------
def _refine(adj: Sequence[Sequence[int]], c1: List[int], c2: List[int]):
    '''
    Refine two colourings of one graph together until stable (a colour is split by the colours of the neighbours).
    Returns the two colourings, or None when their histograms differ at some stage.
    '''

    n = len(c1)
    k = len(set(c1))
    while True:
        s1 = [ (c1[v], tuple(sorted(c1[w] for w in adj[v]))) for v in range(n) ]
        s2 = [ (c2[v], tuple(sorted(c2[w] for w in adj[v]))) for v in range(n) ]
        if sorted(s1) != sorted(s2):
            return None
        ids = { s : i for i, s in enumerate(sorted(set(s1))) }
        c1, c2 = [ ids[s] for s in s1 ], [ ids[s] for s in s2 ]
        if len(ids) == k:
            return c1, c2
        k = len(ids)


def _extend(adj, c1: List[int], c2: List[int]) -> Optional[Dict[int, int]]:
    '''The automorphism of the graph that maps the colouring c1 onto c2 (atom -> image), or None.'''

    refined = _refine(adj, c1, c2)
    if refined is None:
        return None
    c1, c2 = refined
    n = len(c1)
    by1, by2 = {}, {}
    for v in range(n):
        by1.setdefault(c1[v], []).append(v)
        by2.setdefault(c2[v], []).append(v)
    open_labels = [ k for k, v in by1.items() if len(v) > 1 ]
    if not open_labels:
        image = { by1[k][0] : by2[k][0] for k in by1 }
        if all(sorted(image[w] for w in adj[v]) == sorted(adj[image[v]]) for v in range(n)):
            return image
        return None
    label = min(open_labels, key=lambda k: (len(by1[k]), k))
    new = max(c1) + 1
    x = by1[label][0]
    for y in by2[label]:
        d1, d2 = list(c1), list(c2)
        d1[x], d2[y] = new, new
        image = _extend(adj, d1, d2)
        if image is not None:
            return image
    return None


def automorphism_map(adj, colours: Sequence[int], sources: Sequence[int], targets: Sequence[int]) -> Optional[Dict[int, int]]:
    '''An automorphism that keeps the colours and sends the atoms `sources` onto `targets`, in order (atom -> image), or None.'''

    c1, c2 = list(colours), list(colours)
    top = max(colours) + 1
    for k, (s, t) in enumerate(zip(sources, targets)):
        c1[s], c2[t] = top + k, top + k
    return _extend(adj, c1, c2)


def automorphism_exists(adj, colours: Sequence[int], sources: Sequence[int], targets: Sequence[int]) -> bool:
    '''Whether an automorphism that keeps the colours sends the atoms `sources` onto `targets`, in order.'''

    return automorphism_map(adj, colours, sources, targets) is not None


def atom_orbits(graph: nx.Graph, colours: Sequence[int], atoms: Optional[Set[int]] = None) -> List[np.ndarray]:
    '''
    Orbits of the atoms under the automorphisms of the bond graph that keep the colours; one array per atom.
    With `atoms`, only the orbits of those atoms are found (the others are returned as they are).
    '''

    n = graph.number_of_nodes()
    adj = [ sorted(graph.neighbors(v)) for v in range(n) ]
    stable = _refine(adj, list(colours), list(colours))[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    wanted = set(range(n)) if atoms is None else set(atoms)
    for i in sorted(wanted):
        for j in range(n):
            if j != i and stable[i] == stable[j] and find(i) != find(j) and automorphism_exists(adj, colours, [i], [j]):
                parent[find(j)] = find(i)
    roots = [ find(v) for v in range(n) ]
    return [ np.array([ w for w in range(n) if roots[w] == roots[v] ]) for v in range(n) ]


def pair_orbit_closure(graph: nx.Graph, colours: Sequence[int], pairs) -> Set[Tuple[int, int]]:
    '''
    The given pairs and every pair that a symmetry of the bond graph (colours kept) maps one of them onto.
    Pairs are (i, j) with i < j.
    '''

    pairs = sorted(pairs)
    if not pairs:
        return set()
    adj = [ sorted(graph.neighbors(v)) for v in range(graph.number_of_nodes()) ]
    orbit = atom_orbits(graph, colours, { atom for pair in pairs for atom in pair })
    out: Set[Tuple[int, int]] = set()
    for i, j in pairs:
        if (i, j) in out:
            continue
        out.add((i, j))
        for k in orbit[i]:
            for l in orbit[j]:
                key = (int(min(k, l)), int(max(k, l)))
                if k == l or key in out:
                    continue
                if automorphism_exists(adj, colours, [i, j], [k, l]) or automorphism_exists(adj, colours, [i, j], [l, k]):
                    out.add(key)
    return out



def graph_equivalences(graph: nx.Graph, colours: Sequence) -> Dict[int, np.ndarray]:
    '''
    Orbits of the atoms under the automorphisms of the bond graph that keep the colours.

    Parameters
    ----------
    graph: nx.Graph.
        Bond graph with the atoms as nodes 0 ... N - 1.
    colours: sequence.
        Hashable label of each atom; atoms with different labels are never equivalent.

    Returns
    -------
    eq: dict.
        Atom index -> sorted array of the atoms equivalent to it, itself included.
    '''

    ids = { label : k for k, label in enumerate(sorted(set(colours), key=repr)) }
    orbit = atom_orbits(graph, [ ids[label] for label in colours ])
    return { i : np.sort(orbit[i]) for i in range(len(colours)) }
