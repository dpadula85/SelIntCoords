#!/usr/bin/env python

'''
ring_groups - internal coordinates grouped by the ring system they belong to.

When the dependencies of a Joyce run are written by hand it helps to find the
coordinates of one ring together, under a line that names the ring. A ring
system is a set of rings fused by sharing a bond (naphthalene, benzothiophene,
...); rings joined by a single bond, or by a vinylene, are separate systems.
Hydrogens count as members of the system of the atom they are bonded to, so
the C-H bonds and H-C-C angles of a ring stay with the ring.

A coordinate is assigned to

* a ring system, when all its atoms belong to it;
* "<ring> substituents", when it also includes heavy atoms outside any ring;
* "<ring> to <ring>", when it spans two ring systems (the link between them);
* "methyls", when it involves a methyl group;
* "substituents", when none of its atoms belongs to a ring system.
'''

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from .mol_features import Features

log = logging.getLogger(__name__)

# fused pairs of rings that have a name of their own
FUSED_PAIRS = {
    ('benzene', 'thiophene'): 'benzothiophene',
    ('benzene', 'furan'): 'benzofuran',
    ('benzene', 'selenophene'): 'benzoselenophene',
    ('benzene', 'pyrrole'): 'indole',
    ('benzene', 'thiazole'): 'benzothiazole',
    ('benzene', 'oxazole'): 'benzoxazole',
    ('benzene', 'imidazole'): 'benzimidazole',
    ('benzene', 'thiadiazole'): 'benzothiadiazole',
    ('benzene', 'oxadiazole'): 'benzoxadiazole',
    ('benzene', 'triazole'): 'benzotriazole',
    ('benzene', 'pyridine'): 'quinoline',
    ('thiophene', 'thiophene'): 'thienothiophene',
    ('pyrrole', 'thiophene'): 'thienopyrrole',
}

ACENES = {2: 'naphthalene', 3: 'anthracene', 4: 'tetracene', 5: 'pentacene',
          6: 'hexacene', 7: 'heptacene'}

SATURATED = {3: 'cyclopropane', 4: 'cyclobutane', 5: 'cyclopentane',
             6: 'cyclohexane', 7: 'cycloheptane', 8: 'cyclooctane'}

# heteroatoms of an aromatic five-membered ring (sorted), and whether they
# are adjacent, to the name of the ring
FIVE_RING = {
    (('S',), None): 'thiophene',
    (('O',), None): 'furan',
    (('Se',), None): 'selenophene',
    (('N',), None): 'pyrrole',
    (('N', 'S'), False): 'thiazole',
    (('N', 'S'), True): 'isothiazole',
    (('N', 'O'), False): 'oxazole',
    (('N', 'O'), True): 'isoxazole',
    (('N', 'N'), False): 'imidazole',
    (('N', 'N'), True): 'pyrazole',
}

SIX_RING_N = {1: 'pyridine', 3: 'triazine'}
SIX_RING_NN = {1: 'pyridazine', 2: 'pyrimidine', 3: 'pyrazine'}


def _cycle(f: Features, ring: Sequence[int]) -> List[int]:
    '''Atoms of a ring in the order they are bonded.'''

    sub = f.graph.subgraph(ring)
    start = ring[0]
    order = [start]
    prev, cur = None, start
    while True:
        nxt = [j for j in sub.neighbors(cur) if j != prev and j not in order]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        order.append(cur)
    return order


def ring_name(f: Features, ring: Sequence[int]) -> str:
    '''Name of a single ring (thiophene, benzene, ...), or a description.'''

    ring = tuple(ring)
    order = _cycle(f, ring)
    n = len(ring)
    hetero = [i for i in order if f.el[i] != 'C']
    names = [f.el[i] for i in hetero]

    if ring in f.aromatic_rings:
        if n == 6:
            if not hetero:
                return 'benzene'
            if set(names) == {'N'}:
                if len(hetero) in (1, 3):
                    return SIX_RING_N[len(hetero)]
                if len(hetero) == 2:
                    k = order.index(hetero[1]) - order.index(hetero[0])
                    return SIX_RING_NN[min(abs(k), n - abs(k))]
        elif n == 5:
            if len(hetero) <= 2:
                adjacent = None
                if len(hetero) == 2:
                    k = abs(order.index(hetero[1]) - order.index(hetero[0]))
                    adjacent = k in (1, n - 1)
                name = FIVE_RING.get((tuple(sorted(names)), adjacent))
                if name:
                    return name
            elif len(hetero) == 3:
                counts = Counter(names)
                if counts['N'] == 3:
                    return 'triazole'
                for el, label in (('S', 'thiadiazole'), ('O', 'oxadiazole'), ('Se', 'selenadiazole')):
                    if counts[el] == 1 and counts['N'] == 2:
                        return label

    # a thiazolidine-type ring with a thione or oxo group on its carbon
    if n == 5 and sorted(f.el[i] for i in ring) == ['C', 'C', 'C', 'N', 'S']:
        if any(f.terminal(j) and f.el[j] == 'S' for i in ring for j in f.nbrs[i]):
            return 'rhodanine'
        if any(f.terminal(j) and f.el[j] == 'O' for i in ring for j in f.nbrs[i]):
            return 'thiazolidinone'

    if not hetero and n in SATURATED:
        n_sp2 = sum(f.planar(i) for i in ring)
        if n_sp2 == 0:
            return SATURATED[n]
        if n_sp2 == 2:
            return SATURATED[n][:-3] + 'ene'

    formula = ''.join('%s%s' % (el, c if c > 1 else '') for el, c in sorted(Counter(f.el[i] for i in ring).items()))
    return '%d-membered %s ring' % (n, formula)


class RingSystem:
    '''A set of rings fused by shared bonds.'''

    def __init__(self, f: Features, rings: List[Tuple[int, ...]]):
        self.rings = rings
        self.atoms = sorted(set(i for r in rings for i in r))
        self.name = self._name(f)

    def _name(self, f: Features) -> str:
        names = [ring_name(f, r) for r in self.rings]
        if len(self.rings) == 1:
            return names[0]

        if set(names) == {'benzene'}:
            n = len(names)
            if self._linear(f):
                return ACENES.get(n, '%d benzene (fused)' % n)
            if n == 3:
                return 'phenanthrene'
            return '%d benzene (fused)' % n

        if len(names) == 2:
            pair = tuple(sorted(names))
            if pair in FUSED_PAIRS:
                return FUSED_PAIRS[pair]
        counts = Counter(names)
        parts = ['%s' % n if c == 1 else '%d %s' % (c, n) for n, c in sorted(counts.items())]
        return ' + '.join(parts) + ' (fused)'

    def _linear(self, f: Features) -> bool:
        '''All the middle rings of a chain of rings are fused on opposite bonds.'''

        for ring in self.rings:
            shared = [set(ring) & set(other) for other in self.rings if other != ring]
            shared = [s for s in shared if len(s) >= 2]
            if len(shared) > 2:
                return False
            if len(shared) == 2:
                order = _cycle(f, ring)
                a = sorted(order.index(i) for i in shared[0])
                b = sorted(order.index(i) for i in shared[1])
                # opposite bonds of a hexagon are three bonds apart
                d = min(abs(a[0] - b[0]), len(order) - abs(a[0] - b[0]))
                if d != 3:
                    return False
        return True


def ring_systems(f: Features) -> List[RingSystem]:
    '''Ring systems, ordered by their lowest atom index.'''

    g = nx.Graph()
    g.add_nodes_from(range(len(f.rings)))
    for a in range(len(f.rings)):
        for b in range(a + 1, len(f.rings)):
            if len(set(f.rings[a]) & set(f.rings[b])) >= 2:
                g.add_edge(a, b)
    systems = [RingSystem(f, [f.rings[k] for k in sorted(comp)]) for comp in nx.connected_components(g)]
    systems.sort(key=lambda s: s.atoms[0])

    # the same name on several systems is told apart by a number
    counts = Counter(s.name for s in systems)
    seen: Counter = Counter()
    for s in systems:
        if counts[s.name] > 1:
            seen[s.name] += 1
            s.name = '%s %d' % (s.name, seen[s.name])
    return systems


def methyl_atoms(f: Features) -> set:
    '''Methyl carbons (three H, one heavy neighbour) and their hydrogens.'''

    out = set()
    for i in range(f.n):
        if f.el[i] == 'C' and f.deg(i) == 4 and f.n_h(i) == 3:
            out.add(i)
            out.update(j for j in f.nbrs[i] if f.el[j] == 'H')
    return out


class RingGrouper:
    '''
    Assigns internal coordinates to ring-system groups.

    Atoms outside the ring systems are gathered into chains (connected sets
    of heavy atoms with no ring atom among them, methyl groups apart). A
    chain bonded to two or more ring systems is a link between them; a chain
    bonded to one system is a substituent of it.

    Parameters
    ----------
    f: Features.
        Environment of the molecule, for its rings and connectivity.
    '''

    def __init__(self, f: Features):
        self.systems = ring_systems(f)
        self.owner: Dict[int, int] = {}
        for k, s in enumerate(self.systems):
            for i in s.atoms:
                self.owner[i] = k
        # hydrogens follow the atom they are bonded to
        for i in range(f.n):
            if f.el[i] == 'H' and i not in self.owner:
                parent = f.nbrs[i][0] if f.nbrs[i] else None
                if parent in self.owner:
                    self.owner[i] = self.owner[parent]
        self.methyls = methyl_atoms(f)

        # chains of the heavy atoms in no ring system and in no methyl
        loose = [i for i in range(f.n) if i not in self.owner and i not in self.methyls
                 and f.el[i] != 'H']
        sub = f.graph.subgraph(loose)
        self.chain: Dict[int, int] = {}
        self.chain_systems: List[tuple] = []
        for comp in nx.connected_components(sub):
            attached = sorted({self.owner[j] for i in comp for j in f.nbrs[i] if j in self.owner})
            for i in comp:
                self.chain[i] = len(self.chain_systems)
            self.chain_systems.append(tuple(attached))
        # hydrogens of a chain follow it
        for i in range(f.n):
            if f.el[i] == 'H' and i not in self.owner and i not in self.methyls:
                parent = f.nbrs[i][0] if f.nbrs[i] else None
                if parent in self.chain:
                    self.chain[i] = self.chain[parent]

    def label(self, coord: Sequence[int]) -> Tuple[tuple, str]:
        '''
        Sort key and header of the group of a coordinate.

        The key puts every ring system before its substituents, the links
        between two systems after the second of them, and methyls and loose
        substituents last.
        '''

        coord = [int(i) for i in coord]
        if any(i in self.methyls for i in coord):
            return (len(self.systems), 0), 'methyls'

        ring_part = {self.owner[i] for i in coord if i in self.owner}
        chains = [self.chain_systems[self.chain[i]] for i in coord if i in self.chain]
        linked = set(ring_part)
        for attached in chains:
            if len(attached) >= 2:
                linked.update(attached)

        if len(linked) >= 2:
            owners = sorted(linked)
            names = ' to '.join(self.systems[k].name for k in owners)
            return (owners[-1], 2, *owners[:-1]), names

        involved = sorted(ring_part | {k for attached in chains for k in attached})
        if not involved:
            return (len(self.systems), 1), 'substituents'
        k = involved[0]
        if chains:
            return (k, 1), '%s substituents' % self.systems[k].name
        return (k, 0), self.systems[k].name

    def group(self, coords: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        '''
        Reorder coordinates by group.

        Returns
        -------
        order: np.ndarray.
            Indices into `coords` in the new order (the original order is
            kept inside a group).
        labels: list of str.
            Header of the group of each coordinate, in the new order.
        '''

        if len(coords) == 0:
            return np.empty(0, dtype=int), []
        keyed = [self.label(c) for c in coords]
        order = sorted(range(len(coords)), key=lambda k: (keyed[k][0], k))
        return np.asarray(order, dtype=int), [keyed[k][1] for k in order]


def insert_headers(path: str, labels: Dict[int, str]) -> None:
    '''
    Write a `; header` line before each run of coordinates of one group in
    the [ bonds ], [ angles ] and [ dihedrals ] sections of a topology.

    Parameters
    ----------
    path: str.
        Topology written by `make_top`; every coordinate line ends with
        `; <global coordinate index>`.
    labels: dict.
        Global coordinate index (1-based) -> header.
    '''

    with open(path) as fh:
        lines = fh.read().split('\n')

    out = []
    section, last = None, None
    for line in lines:
        m = re.match(r'\s*\[\s*(\w+)\s*\]', line)
        if m:
            section, last = m.group(1), None
            out.append(line)
            continue
        if section in ('bonds', 'angles', 'dihedrals'):
            if line.lstrip().startswith(';'):
                last = None
            m = re.search(r';\s*(\d+)\s*$', line)
            if m and not line.lstrip().startswith(';'):
                label = labels.get(int(m.group(1)))
                if label is not None and label != last:
                    out.append('; %s' % label)
                    last = label
        out.append(line)

    with open(path, 'w') as fh:
        fh.write('\n'.join(out))
