#!/usr/bin/env python

'''
symmetry - atoms equivalent by the point-group symmetry of a geometry.

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
'''

import logging
from typing import Dict, List, Optional

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
