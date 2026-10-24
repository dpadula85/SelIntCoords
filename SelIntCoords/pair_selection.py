#!/usr/bin/env python

'''
pair_selection - which 1-n Lennard-Jones pairs a topology needs, from the geometry alone.

Every pair of atoms is excluded from the non-bonded interaction in the topologies made for Joyce, and only the
pairs that matter are given back as explicit [ pairs ]. Without a pair, two atoms pass through each other, so a
pair matters when some conformer of the molecule brings it closer than `BUMP_RATIO` times its sigma. The
conformers are those of a rigid-rotor model (Ramachandran): bond lengths, angles, rings and double bonds are
fixed and the rotatable bonds turn. The selection mimics what is done by hand, twist the molecule, see what
collides, add the pair, twist again:

1. Start from an empty set S of pairs.
2. Sample conformers that respect S only: single-rotor Metropolis moves, accepted when no pair of S bumps.
3. A pair outside S that bumps in at least `fraction` of the samples joins S.
4. Ring blocks: when any pair between two ring systems (fused rings count as one) bumps in at least one sample,
   every heavy-atom 1,5-or-farther pair between the two ring systems joins S (a stacking contact).
5. Repeat until S stops growing.
6. Pairs that a symmetry of the bond graph (atom colours kept) maps onto each other are written together.

Only heavy-atom pairs and H...H pairs are written. Pairs of a hydrogen and a heavy atom are hard spheres in the
sampling like all the others but are not written, and 1,4 pairs are not candidates (the dihedral covers them).
The sampling runs many independent chains in parallel, stepped together as NumPy arrays.
'''

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx
from scipy import constants as _c

log = logging.getLogger(__name__)

# a pair bumps when it is closer than this fraction of its sigma
BUMP_RATIO = 0.8

# a pair joins the list when it bumps in at least this fraction of the sampled conformers
FRACTION = 0.01

# pairs closer than this many bonds are not candidates (1,4 pairs are covered by the dihedral)
MIN_SEPARATION = 4

# nm (topology sigma) to Angstrom (coordinates)
NM_TO_A = _c.nano / _c.angstrom

# sampling: independent chains, recorded conformers per chain, rotor moves between records and before the first
CHAINS = 400
RECORDS = 50
GAP = 30
BURN = 300
MAX_ROUNDS = 12

# largest number of pair coordinates gathered at once when recording a conformer (keeps the memory low for large molecules)
BLOCK = 4_000_000

# a pair already closer than the threshold in the input geometry is held at this fraction of its distance
HOLD = 0.98


def find_rotors(graph: nx.Graph, double: Set[Tuple[int, int]]) -> List[Tuple[int, int, np.ndarray]]:
    '''
    Rotatable bonds: single bonds that split the molecule and do not end on a terminal atom.

    Parameters
    ----------
    graph: nx.Graph.
        Bond graph.
    double: set of (int, int).
        Double bonds, smaller index first (`flex_classes.double_bonds`).

    Returns
    -------
    rotors: list of (int, int, np.ndarray).
        Atom `a` on the side that stays, atom `b` on the side that turns, and the sorted indices of
        the atoms that turn (the smaller side of the bond, including `b`).
    '''

    rotors = []
    for a, b in sorted(nx.bridges(graph)):
        a, b = min(a, b), max(a, b)
        if (a, b) in double or graph.degree(a) == 1 or graph.degree(b) == 1:
            continue
        cut = graph.copy()
        cut.remove_edge(a, b)
        side_a = sorted(nx.node_connected_component(cut, a))
        side_b = sorted(nx.node_connected_component(cut, b))
        if len(side_b) <= len(side_a):
            rotors.append((a, b, np.array(side_b, dtype=int)))
        else:
            rotors.append((b, a, np.array(side_a, dtype=int)))
    return rotors


def ring_systems(graph: nx.Graph) -> Dict[int, int]:
    '''Ring system of every ring atom (fused rings share one); atoms in no ring are missing.'''

    systems = []
    for ring in nx.minimum_cycle_basis(graph):
        ring = set(ring)
        for system in systems:
            if len(system & ring) >= 2:
                system |= ring
                break
        else:
            systems.append(ring)
    merged = True
    while merged:
        merged = False
        for a in range(len(systems)):
            for b in range(a + 1, len(systems)):
                if len(systems[a] & systems[b]) >= 2:
                    systems[a] |= systems.pop(b)
                    merged = True
                    break
            if merged:
                break
    return { atom : k for k, system in enumerate(systems) for atom in system }


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    '''Cross product along the last axis of broadcastable arrays (faster than `np.cross` for small arrays).'''

    out = np.empty(np.broadcast(u, v).shape)
    out[..., 0] = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    out[..., 1] = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    out[..., 2] = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return out


class Ensemble:
    '''
    Rigid-rotor conformers of a molecule, sampled by many Metropolis chains stepped together.

    Parameters
    ----------
    coords: np.ndarray (shape: (N, 3)).
        Coordinates in Angstrom.
    rotors: list of (int, int, np.ndarray).
        Rotatable bonds from `find_rotors`.
    pairs: np.ndarray (shape: (M, 2)).
        Candidate atom pairs.
    sigma: np.ndarray (shape: (M,)).
        Lennard-Jones sigma of each pair in Angstrom.
    ratio: float.
        A pair bumps below `ratio * sigma`.
    '''

    def __init__(self, coords: np.ndarray, rotors, pairs: np.ndarray, sigma: np.ndarray, ratio: float = BUMP_RATIO):
        self.x0 = np.asarray(coords, dtype=float)
        self.n = len(self.x0)
        self.rotors = rotors
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.sigma = np.asarray(sigma, dtype=float)
        threshold = ratio * self.sigma
        d0 = np.linalg.norm(self.x0[self.pairs[:, 0]] - self.x0[self.pairs[:, 1]], axis=1)
        self.d0 = d0
        # a pair that is in contact in the input geometry is held there instead of being forbidden
        self.limit = np.where(d0 < threshold, HOLD * d0, threshold)
        self.cross, self.moving_pos, self.fixed_atoms = [], [], []
        for a, b, moving in rotors:
            mask = np.zeros(self.n, dtype=bool)
            mask[moving] = True
            in0, in1 = mask[self.pairs[:, 0]], mask[self.pairs[:, 1]]
            idx = np.where(in0 != in1)[0]
            turning = np.where(in0[idx], self.pairs[idx, 0], self.pairs[idx, 1])
            fixed = np.where(in0[idx], self.pairs[idx, 1], self.pairs[idx, 0])
            self.cross.append(idx)
            self.moving_pos.append(np.searchsorted(moving, turning))
            self.fixed_atoms.append(fixed)

    def sample(self, active: np.ndarray, rng: np.random.Generator, chains: int = CHAINS, records: int = RECORDS,
               gap: int = GAP, burn: int = BURN) -> Tuple[np.ndarray, np.ndarray, int]:
        '''
        Sample conformers that respect the `active` pairs only.

        Returns
        -------
        squeezed: np.ndarray (shape: (M,)).
            Number of sampled conformers in which each pair outside `active` is below its threshold.
        closest: np.ndarray (shape: (M,)).
            Smallest distance over sigma of each pair in the sampled conformers.
        total: int.
            Number of sampled conformers.
        '''

        m = len(self.pairs)
        nr = len(self.rotors)
        if nr == 0:
            return np.zeros(m, dtype=int), self.d0 / self.sigma, 1
        gap, burn = max(gap, 2 * nr), max(burn, 20 * nr)
        y = np.broadcast_to(self.x0, (chains,) + self.x0.shape).copy()
        squeezed = np.zeros(m, dtype=int)
        closest = np.full(m, np.inf)
        free = ~active
        block = max(1, BLOCK // (3 * m))
        # only the pairs that are already selected can reject a move
        pos, fixed, limit = [], [], []
        for r in range(nr):
            idx = self.cross[r]
            keep = active[idx]
            pos.append(self.moving_pos[r][keep])
            fixed.append(self.fixed_atoms[r][keep])
            limit.append(self.limit[idx][keep])
        for step in range(burn + records * gap):
            r = int(rng.integers(nr))
            a, b, moving = self.rotors[r]
            origin = y[:, a]
            axis = y[:, b] - origin
            axis /= np.linalg.norm(axis, axis=1)[:, None]
            angle = rng.uniform(-np.pi, np.pi, chains)
            cos, sin = np.cos(angle)[:, None, None], np.sin(angle)[:, None, None]
            u = axis[:, None, :]
            v = y[:, moving] - origin[:, None, :]
            turned = origin[:, None, :] + v * cos + _cross(u, v) * sin + u * (v * u).sum(axis=2, keepdims=True) * (1.0 - cos)
            if len(pos[r]):
                d = np.linalg.norm(turned[:, pos[r]] - y[:, fixed[r]], axis=2)
                ok = (d >= limit[r]).all(axis=1)
                keep = np.where(ok)[0]
                y[keep[:, None], moving[None, :]] = turned[keep]
            else:
                y[:, moving] = turned
            if step >= burn and (step - burn) % gap == 0:
                for c0 in range(0, chains, block):
                    yc = y[c0:c0 + block]
                    d = np.linalg.norm(yc[:, self.pairs[:, 0]] - yc[:, self.pairs[:, 1]], axis=2)
                    squeezed += ((d < self.limit) & free).sum(axis=0)
                    closest = np.minimum(closest, (d / self.sigma).min(axis=0))
        return squeezed, closest, chains * records


# ----------------------------------------------------------------------
# symmetry of the bond graph (individualisation and refinement)
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


def _extend(adj, c1: List[int], c2: List[int]) -> bool:
    '''Whether the colourings c1 -> c2 are mapped onto each other by an automorphism of the graph.'''

    refined = _refine(adj, c1, c2)
    if refined is None:
        return False
    c1, c2 = refined
    n = len(c1)
    by1, by2 = {}, {}
    for v in range(n):
        by1.setdefault(c1[v], []).append(v)
        by2.setdefault(c2[v], []).append(v)
    open_labels = [ k for k, v in by1.items() if len(v) > 1 ]
    if not open_labels:
        image = { by1[k][0] : by2[k][0] for k in by1 }
        return all(sorted(image[w] for w in adj[v]) == sorted(adj[image[v]]) for v in range(n))
    label = min(open_labels, key=lambda k: (len(by1[k]), k))
    new = max(c1) + 1
    x = by1[label][0]
    for y in by2[label]:
        d1, d2 = list(c1), list(c2)
        d1[x], d2[y] = new, new
        if _extend(adj, d1, d2):
            return True
    return False


def automorphism_exists(adj, colours: Sequence[int], sources: Sequence[int], targets: Sequence[int]) -> bool:
    '''Whether an automorphism that keeps the colours sends the atoms `sources` onto `targets`, in order.'''

    c1, c2 = list(colours), list(colours)
    top = max(colours) + 1
    for k, (s, t) in enumerate(zip(sources, targets)):
        c1[s], c2[t] = top + k, top + k
    return _extend(adj, c1, c2)


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


# ----------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------
@dataclass
class PairSelection:
    '''
    Result of `select_pairs`.

    written: pairs to write, smaller index first.
    closest: closest approach over sigma of every candidate in conformers that respect the pairs selected, so a
        written pair sits at the threshold and a pair left out shows how near it comes.
    history: pairs written after each round.
    '''

    written: List[Tuple[int, int]]
    closest: Dict[Tuple[int, int], float]
    history: List[int]


def select_pairs(graph: nx.Graph, coords: np.ndarray, rotors, candidates: np.ndarray, sigma: np.ndarray,
                 hydrogen: Sequence[bool], colours: Sequence[int], ratio: float = BUMP_RATIO,
                 fraction: float = FRACTION, seed: int = 0) -> PairSelection:
    '''
    The 1-n pairs a molecule needs (see the module docstring).

    Parameters
    ----------
    graph: nx.Graph.
        Bond graph.
    coords: np.ndarray (shape: (N, 3)).
        Coordinates in Angstrom.
    rotors: list of (int, int, np.ndarray).
        Rotatable bonds, `find_rotors`.
    candidates: np.ndarray (shape: (M, 2)).
        Pairs 1,5 or farther apart.
    sigma: np.ndarray (shape: (M,)).
        Lennard-Jones sigma of every candidate in nm.
    hydrogen: sequence of bool.
        Whether each atom is a hydrogen.
    colours: sequence of int.
        Atoms with the same colour may be exchanged by a symmetry (element and atom type).
    ratio, fraction: float.
        A pair bumps below `ratio * sigma`; it is written when it bumps in `fraction` of the sampled conformers.
    seed: int.
        Seed of the sampling; the result is reproducible for a given seed.
    '''

    pairs = np.sort(np.asarray(candidates, dtype=int).reshape(-1, 2), axis=1)
    m = len(pairs)
    hydrogen = np.asarray(hydrogen, dtype=bool)
    ih, jh = hydrogen[pairs[:, 0]], hydrogen[pairs[:, 1]]
    written = ih == jh
    heavy = ~ih & ~jh
    system = ring_systems(graph)
    ring_pair = np.array([ heavy[k] and a in system and b in system and system[a] != system[b]
                           for k, (a, b) in enumerate(pairs) ], dtype=bool)
    ensemble = Ensemble(coords, rotors, pairs, np.asarray(sigma, dtype=float) * NM_TO_A, ratio)
    rng = np.random.default_rng(seed)
    active = np.zeros(m, dtype=bool)
    history = []
    closest = np.full(m, np.inf)
    for _ in range(MAX_ROUNDS):
        squeezed, closest, total = ensemble.sample(active, rng)
        new = (squeezed >= max(fraction * total, 1)) & ~active
        # ring blocks: one bump between two ring systems brings in all their heavy pairs
        touched = { (min(system[a], system[b]), max(system[a], system[b])) for a, b in pairs[(squeezed > 0) & ring_pair] }
        for k in np.where(ring_pair & ~active)[0]:
            a, b = pairs[k]
            new[k] |= (min(system[a], system[b]), max(system[a], system[b])) in touched
        if not new.any():
            break
        active |= new
        history.append(int((active & written).sum()))
        log.debug('pair selection round %d: %d pairs written', len(history), history[-1])
    chosen = { (int(a), int(b)) for a, b in pairs[active & written] }
    candidate_set = { (int(a), int(b)) for a, b in pairs[written] }
    chosen = sorted(pair_orbit_closure(graph, colours, chosen) & candidate_set)
    ratios = { (int(a), int(b)) : float(r) for (a, b), r in zip(pairs, closest) }
    return PairSelection(chosen, ratios, history)


def format_commented_pair(i: int, j: int, eps: float, sig: float, ratio: Optional[float], label: str) -> str:
    '''A pair line of the [ pairs ] section, commented out.'''

    if ratio is None:
        note = 'covered by the dihedral'
    elif np.isfinite(ratio):
        note = 'closest approach %.2f sigma' % ratio
    else:
        note = 'sigma not known'
    return ';%4d %4d   2  1.000  0.000  0.000  %7.4f  %9.6f   ; %s, %s' % (i + 1, j + 1, sig, eps, label, note)


def append_commented_pairs(path: str, lines: Sequence[str], reason: Sequence[str] = ()) -> None:
    '''
    Put `lines` right after the last pair of the [ pairs ] section of a topology, with no blank line in between
    (GROMACS stops reading a section at one). `reason` goes as comment lines under the `; N pairs` line that opens
    the section.
    '''

    text = open(path).read().split('\n')
    start = next(n for n, line in enumerate(text) if line.split()[:2] == ['[', 'pairs'])
    end = next((n for n in range(start + 1, len(text)) if text[n].lstrip().startswith('[')), len(text))
    last = max(n for n in range(start, end) if text[n].strip())
    first = next((n for n in range(start + 1, end) if re.match(r';\s*\d+ pairs', text[n])), start)
    text[last + 1:last + 1] = ['; pairs left out, in the order of the 1,n lists: uncomment those you need', *lines]
    text[first + 1:first + 1] = list(reason)
    open(path, 'w').write('\n'.join(text))
