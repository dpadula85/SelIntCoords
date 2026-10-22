#!/usr/bin/env python

'''
pair_selection - which 1-n Lennard-Jones pairs a topology needs.

Every pair of atoms is excluded from the non-bonded interaction in the
topologies made for Joyce, and only the pairs that matter are given back as
explicit [ pairs ]. A pair matters when its distance changes with a torsion:
if the two atoms sit in the same rigid unit, their distance is fixed by the
bonds, angles and stiff dihedrals, and a pair term would only duplicate them.
A pair across a rotatable bond is kept when, along the rotation of that bond,
the atoms come closer than a fraction of the Lennard-Jones sigma of the pair
(the closest approach in units of sigma, `r_min / sigma`): this is the
steric and chalcogen contact (S...S, S...H) that shapes the torsional
profile.

The rotation about a bond turns the atoms on one side rigidly about the bond
axis, the rest of the molecule being kept at the geometry given. A pair that
crosses several rotors takes the smallest distance of the individual rotations.
'''

import logging
import re
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import networkx as nx

log = logging.getLogger(__name__)

# angular step of the rotation scan (degrees)
STEP = 10.0

# pairs separated by fewer bonds are described by the dihedral itself
MIN_SEPARATION = ('1,5', '1,6', '1,7', 'other')


def rotor_sides(graph: nx.Graph, bond: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    '''Atoms on each side of a rotatable bond, as boolean masks over the atoms.'''

    j, k = bond
    cut = graph.copy()
    cut.remove_edge(j, k)
    n = graph.number_of_nodes()
    side_j = np.zeros(n, dtype=bool)
    side_j[list(nx.node_connected_component(cut, j))] = True
    side_k = np.zeros(n, dtype=bool)
    side_k[list(nx.node_connected_component(cut, k))] = True
    return side_j, side_k


def closest_approach(graph: nx.Graph, coords: np.ndarray, rotors: Iterable[Tuple[int, int]],
                     pairs: np.ndarray, step: float = STEP) -> np.ndarray:
    '''
    Closest approach of atom pairs along the rotation of the bonds between them.

    Parameters
    ----------
    graph: nx.Graph.
        Bond graph.
    coords: np.ndarray (shape: (N, 3)).
        Coordinates in Angstrom.
    rotors: iterable of (int, int).
        Rotatable bonds (0-based atom indices). A bond in a ring does not
        split the molecule and is ignored.
    pairs: np.ndarray (shape: (M, 2)).
        Atom pairs to examine.
    step: float.
        Angular step of the scan, degrees.

    Returns
    -------
    r_min: np.ndarray (shape: (M,)).
        Smallest distance (Angstrom) reached by each pair along the rotation of
        a bond between its atoms; NaN for a pair that no rotor separates.
    '''

    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    r_min = np.full(len(pairs), np.nan)
    angles = np.radians(np.arange(0.0, 360.0, step))
    cos, sin = np.cos(angles)[:, None, None], np.sin(angles)[:, None, None]

    for bond in rotors:
        j, k = bond
        if not graph.has_edge(j, k):
            continue
        side_j, side_k = rotor_sides(graph, bond)
        if side_j[k]:
            continue
        # the atom of each pair that turns is the one on the side of k
        forward = side_j[pairs[:, 0]] & side_k[pairs[:, 1]]
        backward = side_k[pairs[:, 0]] & side_j[pairs[:, 1]]
        across = forward | backward
        if not across.any():
            continue
        fixed = np.where(forward, pairs[:, 0], pairs[:, 1])[across]
        moving = np.where(forward, pairs[:, 1], pairs[:, 0])[across]

        axis = coords[k] - coords[j]
        axis = axis / np.linalg.norm(axis)
        v = coords[moving] - coords[j]                                   # (n, 3)
        along = (v @ axis)[None, :, None] * axis[None, None, :]          # (1, n, 3)
        # Rodrigues rotation of every moving atom for every angle: (angles, n, 3)
        turned = coords[j] + v[None] * cos + np.cross(axis, v)[None] * sin + along * (1.0 - cos)
        dist = np.linalg.norm(turned - coords[fixed][None], axis=2).min(axis=0)
        idx = np.where(across)[0]
        r_min[idx] = np.fmin(r_min[idx], dist)
    return r_min


def pair_ratios(graph: nx.Graph, coords: np.ndarray, rotors: Sequence[Tuple[int, int]],
                LJs: Dict[str, np.ndarray], sigma: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    '''
    `r_min / sigma` of the pairs separated by a rotor.

    Parameters
    ----------
    LJs: dict.
        Lennard-Jones pairs by separation, from `sel_intcoords.list_intcoords`.
    sigma: dict.
        Pair (smaller index first) -> sigma of the pair in nm.

    Returns
    -------
    ratios: dict.
        Pair (smaller index first) -> closest approach over sigma; pairs no
        rotor separates, or closer than 1,5 apart, are missing.
    '''

    pairs = [np.asarray(LJs[key], dtype=int).reshape(-1, 2) for key in MIN_SEPARATION if key in LJs]
    pairs = np.vstack(pairs) if pairs else np.empty((0, 2), dtype=int)
    r_min = closest_approach(graph, coords, rotors, pairs)
    out = {}
    for (i, j), r in zip(pairs, r_min):
        if not np.isnan(r):
            key = (int(min(i, j)), int(max(i, j)))
            out[key] = r / 10.0 / sigma[key]
    return out


def format_commented_pair(i: int, j: int, eps: float, sig: float, ratio: float, label: str) -> str:
    '''A pair line of the [ pairs ] section, commented out.'''

    note = 'no rotor between them' if ratio is None else 'r_min/sigma %.2f' % ratio
    return ';%4d %4d   2  1.000  0.000  0.000  %7.4f  %9.6f   ; %s, %s' % (i + 1, j + 1, sig, eps, label, note)


def append_commented_pairs(path: str, lines: Sequence[str], reason: Sequence[str] = ()) -> None:
    '''
    Put `lines` right after the last pair of the [ pairs ] section of a
    topology, with no blank line in between. `reason` goes as comment lines
    under the `; N pairs` line that opens the section.
    '''

    text = open(path).read().split('\n')
    start = next(n for n, line in enumerate(text) if line.split()[:2] == ['[', 'pairs'])
    end = next((n for n in range(start + 1, len(text)) if text[n].lstrip().startswith('[')), len(text))
    # last line of the section that is not blank
    last = max(n for n in range(start, end) if text[n].strip())
    first = next((n for n in range(start + 1, end) if re.match(r';\s*\d+ pairs', text[n])), start)
    text[last + 1:last + 1] = ['; pairs left out, in the order of the 1,n lists: uncomment those you need', *lines]
    text[first + 1:first + 1] = list(reason)
    open(path, 'w').write('\n'.join(text))
