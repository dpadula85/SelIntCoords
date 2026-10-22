#!/usr/bin/env python

'''
mol_features - chemical environment of every atom from connectivity and
geometry alone (no bond orders, no external cheminformatics toolkit).

Bond orders of large conjugated heteroaromatics cannot be recovered reliably
from a geometry: a generic valence-assignment algorithm turns the sulfur of
a thiophene into a hypervalent one in DPP and PNDI2OD. What the force-field
atom types depend on is much coarser, and available directly:

* hybridisation: sp3 (4 neighbours), sp2 (3 neighbours, planar: the three
  bond angles add up to ~360 degrees; pyramidal amines add up to ~330), sp
  (2 neighbours, collinear);
* a terminal O or S (one neighbour) is doubly bonded to its neighbour
  (carbonyl, sulfonyl, nitro O);
* a ring of 5 or 6 atoms is aromatic when it is planar, every member is
  pi-capable, and it has the Hueckel count: no donor atom in a six-membered
  ring, exactly one in a five-membered ring (a donor is O, S, or N with
  three neighbours, which contributes its lone pair). A ring with a carbonyl
  carbon (lactam, imide, pyridone) is not aromatic: the carbonyl takes the
  pi electrons.
'''

import logging
from functools import cached_property
from typing import List, Tuple

import numpy as np
import networkx as nx
import MDAnalysis as mda

log = logging.getLogger(__name__)

# angle sum (deg) around a three-coordinate atom above which it is planar
PLANAR_ANGLE_SUM = 350.0

# a ring atom further than this from the least-squares plane (Angstrom)
# makes the ring non-planar; benzene rings of MD frames reach 0.17, and
# every ring atom is also tested for planar bonds (PLANAR_ANGLE_SUM)
RING_PLANARITY = 0.30

# collinear (sp) when the bond angle is above this (deg)
LINEAR_ANGLE = 160.0

HALOGENS = ('F', 'Cl', 'Br', 'I')


def read_xyz(path):
    '''
    Read elements and coordinates of a single-frame `.xyz` file.

    Returns
    -------
    elements: list of str.
    coords: np.ndarray (shape: (N, 3)), Angstrom.
    '''

    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    elements, coords = [], []
    for line in lines[2:2 + n]:
        tok = line.split()
        elements.append(tok[0].capitalize())
        coords.append([float(x) for x in tok[1:4]])
    if len(elements) != n:
        raise ValueError('%s: header declares %d atoms, found %d' % (path, n, len(elements)))
    return elements, np.asarray(coords)


class Features:
    '''
    Per-atom chemical environment of a molecule.

    Parameters
    ----------
    elements: list of str.
        Element symbols.
    coords: np.ndarray (shape: (N, 3)).
        Coordinates in Angstrom.
    bonds: np.ndarray (shape: (M, 2)).
        0-based atom index pairs.

    Notes
    -----
    Everything is indexed 0-based, as in `sel_intcoords`.
    '''

    def __init__(self, elements: List[str], coords: np.ndarray, bonds: np.ndarray):
        self.el = list(elements)
        self.pos = np.asarray(coords, dtype=float)
        self.n = len(self.el)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n))
        self.graph.add_edges_from((int(i), int(j)) for i, j in bonds)
        self.nbrs: List[List[int]] = [sorted(self.graph.neighbors(i)) for i in range(self.n)]

    @classmethod
    def from_xyz(cls, path):
        '''Features of an `.xyz` geometry, bonds guessed as in `make_top`.'''

        elements, coords = read_xyz(path)
        u = mda.Universe(str(path), guess_bonds=True)
        bonds = u.bonds.to_indices() if len(u.bonds) else np.empty((0, 2), dtype=int)
        return cls(elements, coords, np.asarray(bonds, dtype=int).reshape(-1, 2))

    # ------------------------------------------------------------------
    # atom-level helpers
    # ------------------------------------------------------------------
    def deg(self, i: int) -> int:
        return len(self.nbrs[i])

    def nbr_el(self, i: int) -> List[str]:
        '''Sorted element symbols of the neighbours of `i`.'''
        return sorted(self.el[j] for j in self.nbrs[i])

    def n_h(self, i: int) -> int:
        '''Number of hydrogen neighbours.'''
        return sum(self.el[j] == 'H' for j in self.nbrs[i])

    def heavy_nbrs(self, i: int) -> List[int]:
        return [j for j in self.nbrs[i] if self.el[j] != 'H']

    def angle_sum(self, i: int) -> float:
        '''Sum of the bond angles around a three-coordinate atom (deg).'''

        a, b, c = self.nbrs[i][:3]
        total = 0.0
        for j, k in ((a, b), (a, c), (b, c)):
            u, v = self.pos[j] - self.pos[i], self.pos[k] - self.pos[i]
            cosang = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
            total += np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        return total

    def planar(self, i: int) -> bool:
        '''Three-coordinate atom with its neighbours in a plane (sp2-like).'''
        return self.deg(i) == 3 and self.angle_sum(i) >= PLANAR_ANGLE_SUM

    def linear(self, i: int) -> bool:
        '''Two-coordinate atom with collinear bonds (sp).'''

        if self.deg(i) != 2:
            return False
        a, b = self.nbrs[i]
        u, v = self.pos[a] - self.pos[i], self.pos[b] - self.pos[i]
        cosang = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) >= LINEAR_ANGLE

    def terminal(self, i: int) -> bool:
        '''O or S with a single neighbour: doubly bonded to it.'''
        return self.el[i] in ('O', 'S') and self.deg(i) == 1

    def is_carbonyl(self, i: int) -> bool:
        '''Three-coordinate C with a terminal O or S neighbour (C=O, C=S).'''

        return (self.el[i] == 'C' and self.deg(i) == 3
                and any(self.terminal(j) for j in self.nbrs[i]))

    def is_c_double_o(self, i: int) -> bool:
        '''Three-coordinate C with a terminal O neighbour (C=O only).'''

        return (self.el[i] == 'C' and self.deg(i) == 3
                and any(self.terminal(j) and self.el[j] == 'O' for j in self.nbrs[i]))

    # ------------------------------------------------------------------
    # rings
    # ------------------------------------------------------------------
    @cached_property
    def rings(self) -> List[Tuple[int, ...]]:
        '''Smallest set of smallest rings, as sorted tuples of atom indices.'''

        # a bond that is a bridge belongs to no ring: the cycle basis is only
        # needed on what is left (the search is cubic in the number of bonds)
        core = self.graph.copy()
        core.remove_edges_from(list(nx.bridges(self.graph)))
        rings = []
        for comp in nx.connected_components(core):
            if len(comp) < 3:
                continue
            for cycle in nx.minimum_cycle_basis(core.subgraph(comp)):
                if len(cycle) <= 8:
                    rings.append(tuple(sorted(cycle)))
        return sorted(rings)

    def rings_of(self, i: int) -> List[Tuple[int, ...]]:
        return [r for r in self.rings if i in r]

    def _planar_ring(self, ring) -> bool:
        pts = self.pos[list(ring)]
        pts = pts - pts.mean(axis=0)
        # smallest singular value direction is the plane normal
        normal = np.linalg.svd(pts)[2][-1]
        return bool(np.abs(pts @ normal).max() <= RING_PLANARITY)

    def pi_role(self, i: int) -> str:
        '''
        Role of a ring atom in a pi system: 'c' (one electron), 'n' (pyridine
        N, one electron), 'd' (donor, two electrons), or '' (not pi-capable).
        '''

        e, d = self.el[i], self.deg(i)
        if e == 'C':
            return 'c' if (d == 3 and self.planar(i) and not self.is_carbonyl(i)) else ''
        if e == 'N':
            if d == 2:
                return 'n'
            return 'd' if (d == 3 and self.planar(i)) else ''
        if e in ('O', 'S', 'Se'):
            return 'd' if d == 2 else ''
        return ''

    @cached_property
    def aromatic_rings(self) -> List[Tuple[int, ...]]:
        '''Rings of 5 or 6 atoms that are planar and have the Hueckel count.'''

        out = []
        for ring in self.rings:
            if len(ring) not in (5, 6):
                continue
            roles = [self.pi_role(i) for i in ring]
            if '' in roles or not self._planar_ring(ring):
                continue
            n_donor = roles.count('d')
            if (len(ring) == 6 and n_donor == 0) or (len(ring) == 5 and n_donor == 1):
                out.append(ring)
        return out

    @cached_property
    def aromatic(self) -> np.ndarray:
        '''Boolean per atom: member of an aromatic ring.'''

        arom = np.zeros(self.n, dtype=bool)
        for ring in self.aromatic_rings:
            arom[list(ring)] = True
        return arom

    def aromatic_ring_sizes(self, i: int) -> List[int]:
        return [len(r) for r in self.aromatic_rings if i in r]

    def ring_sizes(self, i: int) -> List[int]:
        return [len(r) for r in self.rings_of(i)]

    def is_sp2_alkene(self, i: int) -> bool:
        '''Non-aromatic planar C that is not a carbonyl: C=C (or C=N).'''

        return (self.el[i] == 'C' and not self.aromatic[i] and self.planar(i)
                and not self.is_carbonyl(i))

    def summary(self, i: int) -> str:
        '''One-line description of an atom, for logging.'''

        return '%d %s nbrs=%s%s' % (
            i + 1, self.el[i], ''.join(self.nbr_el(i)),
            ' aromatic' if self.aromatic[i] else '')
