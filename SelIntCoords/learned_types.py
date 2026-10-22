#!/usr/bin/env python

'''
learned_types - atom types learned from the residues of a GROMACS force field.

Every `.rtp` file of a force field lists, for each residue, the bonded graph
and the type of every atom. The type of an atom is a function of its chemical
environment, so the environment of each atom in a residue can be tabulated
against its type and looked up for an atom of a new molecule.

The environment of radius r of an atom is the Morgan label: radius 0 is the
element, and the label of radius r combines the label of radius r-1 of the
atom with the sorted labels of radius r-1 of its neighbours. Radius 1 is
therefore the element plus the elements of the neighbours (hence the
degree); radius 2 adds the neighbours of the neighbours, and so on. Lookup
starts at the largest radius and backs off to smaller ones until the label
has been seen in the residues: the type is that of the largest-radius label
known, the most frequent one when a label belongs to several types.

Only the bond graph with element symbols is used, so the same table works for
any force field with residue templates. It can only return types that occur
in the residues of the force field: no thiophene, imide or other group missing
from the biomolecular templates (see `oplsaa_rules` for those).
'''

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ffdb import ForceField, SYMBOLS
from .mol_features import Features

log = logging.getLogger(__name__)

MAX_RADIUS = 3


@dataclass(frozen=True)
class Prediction:
    '''Learned type of one atom: the radius of the label it was found at.'''
    type: str
    radius: int
    candidates: Dict[str, int]  # type -> number of training atoms with this label

    @property
    def unique(self) -> bool:
        return len(self.candidates) == 1


def _morgan(elements: List[str], nbrs: List[List[int]], ids: List[dict], grow: bool):
    '''
    Morgan labels of radius 0 .. MAX_RADIUS as integers.

    `ids[r]` maps the label tuple of radius r to an integer. With `grow` the
    map is extended (training); otherwise a label not in the map is given -1,
    and so is every label of larger radius that depends on it.
    '''

    def intern(r, key):
        table = ids[r]
        if key in table:
            return table[key]
        if grow:
            table[key] = len(table)
            return table[key]
        return -1

    labels = [[intern(0, (e,)) for e in elements]]
    for r in range(1, MAX_RADIUS + 1):
        prev = labels[-1]
        cur = []
        for v in range(len(elements)):
            if prev[v] < 0 or any(prev[u] < 0 for u in nbrs[v]):
                cur.append(-1)
            else:
                cur.append(intern(r, (prev[v], tuple(sorted(prev[u] for u in nbrs[v])))))
        labels.append(cur)
    return labels


class LearnedTyper:
    '''
    Atom types looked up from the residue templates of a force field.

    Parameters
    ----------
    ff: ForceField.
        The force field whose `.rtp` residues are the training set.
    skip: iterable of str, optional.
        Residue names left out of the training set (leave-one-out tests).
    '''

    def __init__(self, ff: ForceField, skip=()):
        self.ff = ff
        self.ids: List[dict] = [dict() for _ in range(MAX_RADIUS + 1)]
        # table[r][label id] -> Counter of types
        self.table: List[Dict[int, Counter]] = [defaultdict(Counter) for _ in range(MAX_RADIUS + 1)]
        self.n_train = 0
        skip = set(skip)
        for res in ff.residues:
            if res.name not in skip:
                self._train(res)
        log.debug('%s: %d training atoms, labels per radius %s',
                  ff.name, self.n_train, [len(d) for d in self.ids])

    def _train(self, res):
        # atoms of the residue plus the +X / -X atoms of its neighbours
        names = list(res.atoms)
        index = {n: k for k, n in enumerate(names)}
        elements = []
        for n in names:
            elements.append(SYMBOLS.get(self.ff.atomtypes[res.atoms[n]].z, '?'))
        for a, b in res.bonds:
            for n in (a, b):
                if n not in index:  # '-C', '+N': element from the letter
                    index[n] = len(elements)
                    elements.append(n.lstrip('+-')[:1].upper())
        nbrs = [[] for _ in elements]
        for a, b in res.bonds:
            nbrs[index[a]].append(index[b])
            nbrs[index[b]].append(index[a])

        labels = _morgan(elements, nbrs, self.ids, grow=True)
        for n in names:
            v = index[n]
            for r in range(MAX_RADIUS + 1):
                self.table[r][labels[r][v]][res.atoms[n]] += 1
            self.n_train += 1

    def predict(self, f: Features, atoms: Optional[List[int]] = None) -> Dict[int, Optional[Prediction]]:
        '''
        Look up the type of atoms of a molecule.

        Parameters
        ----------
        f: Features.
            Molecule (elements and bond graph are used).
        atoms: list of int, optional.
            0-based atom indices; all atoms by default.

        Returns
        -------
        pred: dict.
            Atom index -> `Prediction`, or None when not even the element is
            in the training set.
        '''

        labels = _morgan(f.el, f.nbrs, self.ids, grow=False)
        out = {}
        for i in (range(f.n) if atoms is None else atoms):
            out[i] = None
            for r in range(MAX_RADIUS, -1, -1):
                lab = labels[r][i]
                if lab >= 0 and lab in self.table[r]:
                    cands = self.table[r][lab]
                    # most frequent, name as tie-break so the choice is stable
                    best = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                    out[i] = Prediction(best, r, dict(cands))
                    break
        return out


def residue_features(res, ff: ForceField) -> Tuple[Features, List[str]]:
    '''
    Bonded graph of one residue as `Features` (no geometry), for testing a
    typer on the residues of the force field itself.

    The atoms of the neighbouring residues (`-C`, `+N`) are included, as
    they are in training, after the residue's own atoms.

    Returns
    -------
    f: Features.
    names: list of str.
        Names of the residue's own atoms, which are atoms 0..len(names)-1 of `f`.
    '''

    import numpy as np

    names = list(res.atoms)
    index = {n: k for k, n in enumerate(names)}
    elements = [SYMBOLS.get(ff.atomtypes[res.atoms[n]].z, '?') for n in names]
    for a, b in res.bonds:
        for n in (a, b):
            if n not in index:
                index[n] = len(elements)
                elements.append(n.lstrip('+-')[:1].upper())
    bonds = [(index[a], index[b]) for a, b in res.bonds]
    f = Features(elements, np.zeros((len(elements), 3)), np.asarray(bonds, dtype=int).reshape(-1, 2))
    return f, names
