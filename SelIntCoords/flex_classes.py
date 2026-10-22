#!/usr/bin/env python

'''
flex_classes - what to do with each rotatable bond, as done by hand.

The dihedrals around a bond that is not part of a ring are listed by
`sel_intcoords` in every combination of the substituents of its two atoms. In
a topology prepared for Joyce they are treated by kind of bond:

* methyl rotor (CH3 on a C, N, O, S or Se): parameters taken from the
  literature, not fitted. The three H-C-X-Y dihedrals are kept for one
  heavy neighbour Y of the atom X when X is sp2, N, O or S, and for every
  neighbour of X when X is an sp3 carbon;
* single bond outside a ring that joins two rigid units (rings, sp2 atoms,
  azo N), directly or through a chain of sp3 atoms (an ether, an ethylene
  linker): fitted to a scan of the rotation. One dihedral is kept, the one over the carbon
  skeleton when there is a choice, and starts with a single `n = 2` term
  (Joyce extends it);
* double bond outside a ring (C=C, C=N) with no atom in a ring (the chain
  of a retinal): fitted to a scan like a single bond (`phi<N>` instead of
  `delta<N>`);
* double bond with an atom in a ring (exocyclic, as the C=C joining two rigid
  units in ZR1 and Y6): every dihedral about it is a stiff one, kept at its
  equilibrium value by a harmonic improper term;
* bond in a ring that is not part of the planar framework (the sp3 carbons of
  a ring-locking cycle): literature parameters, like the methyl rotors;
* any other rotor (a side chain whose far end has no ring or sp2 atom, as the
  CH2 of an ethyl, Si methyls): left as it is, with null parameters, for the
  chain merged by `oligomer_builder` or for the user.
'''

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .mol_features import Features
from .pair_selection import rotor_sides

log = logging.getLogger(__name__)

# methyl rotation and sp3 ring torsions, funct 1: V = k (1 + cos(n phi - delta)),
# kJ/mol. 0.651 kJ/mol = 0.1556 kcal/mol, the GAFF barrier of X-c3-c3-X
# (Wang et al., J. Comput. Chem. 2004, 25, 1157, doi:10.1002/jcc.20035)
METHYL_PARAMS = {'delta': 0.0, 'kchi': 0.651, 'n': 3}

# a rotor fitted to a scan starts as one cosine of period 2
SCAN_PARAMS = {'delta': 0.0, 'kchi': 0.0, 'n': 2}

# a rotor left for the user
OTHER_PARAMS = {'delta': 0.0, 'kchi': 0.0, 'n': 3}

# covalent radii (Angstrom) of Cordero et al., Dalton Trans. 2008, 2832
# (doi:10.1039/B801115J), sp3 carbon; a bond shorter than DOUBLE_RATIO times
# the sum of the radii is double (C=C 1.34-1.38 gives 0.88-0.91, an sp2-sp2
# single bond 1.43-1.47 gives 0.94-0.97)
COVALENT_RADII = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'Se': 1.20, 'Si': 1.11}
DOUBLE_RATIO = 0.92

METHYL_HOSTS = ('C', 'N', 'O', 'S', 'Se')

OTHER_LABEL = 'flexible (merge or by hand)'
RING_LABEL = 'ring dihedrals (literature)'


@dataclass
class FlexPlan:
    '''
    Treatment of the flexible dihedrals.

    Attributes
    ----------
    double: np.ndarray (shape: (N, 4)).
        Dihedrals about double bonds, to be added to the stiff ones.
    flex: np.ndarray (shape: (M, 4)).
        Flexible dihedrals kept: methyl rotors first, then the scanned
        rotors, then the others.
    labels: list of str.
        Header of the group of each flexible dihedral.
    params: list of dict.
        Parameters of each flexible dihedral (funct 1).
    counts: dict.
        Number of dihedrals by kind.
    rotors: list of (int, int).
        Rotatable bonds whose rotation moves atoms relative to each other:
        the single and double bonds that are scanned and the others left
        for the user (methyl and ring torsions are not).
    '''

    double: np.ndarray
    flex: np.ndarray
    labels: List[str] = field(default_factory=list)
    params: List[dict] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    rotors: List[tuple] = field(default_factory=list)


def double_bonds(f: Features) -> set:
    '''
    Bonds that are double: between two sp2-like atoms, shorter than
    DOUBLE_RATIO times the sum of the covalent radii, and the shortest of the
    candidates at each atom (an atom has one double bond; the single bonds of
    a conjugated chain are shortened too, so several candidates can meet at one atom).
    '''

    found = []
    for i, j in f.graph.edges():
        if is_double_bond(f, i, j):
            ratio = np.linalg.norm(f.pos[i] - f.pos[j]) / (COVALENT_RADII[f.el[i]] + COVALENT_RADII[f.el[j]])
            found.append((ratio, min(i, j), max(i, j)))
    used, out = set(), set()
    for ratio, i, j in sorted(found):
        if i not in used and j not in used:
            used.update((i, j))
            out.add((i, j))
    return out


def is_double_bond(f: Features, i: int, j: int) -> bool:
    '''Bond between two sp2-like atoms shorter than a single bond.'''

    def sp2_like(a):
        return f.planar(a) or (f.el[a] == 'N' and f.deg(a) == 2)

    if f.el[i] not in COVALENT_RADII or f.el[j] not in COVALENT_RADII:
        return False
    if not (sp2_like(i) and sp2_like(j)):
        return False
    d = np.linalg.norm(f.pos[i] - f.pos[j])
    return bool(d < DOUBLE_RATIO * (COVALENT_RADII[f.el[i]] + COVALENT_RADII[f.el[j]]))


def _equivalent_bonds(eq: Dict[int, np.ndarray], a: Sequence[int], b: Sequence[int]) -> bool:
    '''The central bonds of two dihedrals map onto each other by symmetry.'''

    def same(i, j):
        return i == j or (i in eq and j in set(eq[i].tolist()))

    return (same(a[1], b[1]) and same(a[2], b[2])) or (same(a[1], b[2]) and same(a[2], b[1]))


def _in_ring(f: Features, i: int, j: int) -> bool:
    return any(i in r and j in r for r in f.rings)


def _is_methyl(f: Features, i: int) -> bool:
    return f.el[i] == 'C' and f.deg(i) == 4 and f.n_h(i) == 3


def _methyl_ends(f: Features, m: int, a: int) -> List[int]:
    '''Atoms at the far end of the kept dihedrals of methyl `m` on atom `a`.'''

    others = [x for x in f.nbrs[a] if x != m]
    if f.el[a] == 'C' and f.deg(a) == 4:
        return others
    heavy = [x for x in others if f.el[x] != 'H']
    return heavy[:1] if heavy else others[:1]


def _rigid_atom(f: Features, i: int) -> bool:
    '''Atom of a rigid unit: in a ring, sp2 (planar), sp, or a two-coordinate N (azo, imine).'''

    return bool(f.rings_of(i)) or f.planar(i) or f.linear(i) or (f.el[i] == 'N' and f.deg(i) == 2)


def _joins_rigid_units(f: Features, j: int, k: int) -> bool:
    '''Both sides of the bond hold a rigid unit: it links two parts, it is not a side chain.'''

    return all(any(_rigid_atom(f, i) for i in np.where(side)[0]) for side in rotor_sides(f.graph, (j, k)))


def _numbered(diheds, prefix, eq):
    '''Order rotors by bond and give equivalent bonds the same `<prefix>N` header.'''

    diheds = sorted(diheds, key=lambda d: min(d[1], d[2]))
    reps: List[Sequence[int]] = []
    numbers = []
    for d in diheds:
        for n, rep in enumerate(reps):
            if _equivalent_bonds(eq, d, rep):
                break
        else:
            reps.append(d)
            n = len(reps) - 1
        numbers.append(n + 1)
    ranked = sorted(range(len(diheds)), key=lambda n: (numbers[n], n))
    return [diheds[n] for n in ranked], ['%s%d' % (prefix, numbers[n]) for n in ranked]


def _scan_choice(f: Features, dihedrals: Sequence[Sequence[int]]) -> Sequence[int]:
    '''The dihedral over the sp2 carbon skeleton, ties broken by atom index.'''

    def cost(atom):
        if f.el[atom] == 'H':
            return 3
        return (0 if f.el[atom] == 'C' else 1) + (0 if f.planar(atom) else 0.5)

    return min(dihedrals, key=lambda d: (cost(d[0]) + cost(d[3]), d[0], d[3]))


def classify_flex(f: Features, flex: np.ndarray, eq: Optional[Dict[int, np.ndarray]] = None) -> FlexPlan:
    '''
    Sort the flexible dihedrals of a molecule by kind of rotor.

    Parameters
    ----------
    f: Features.
        Environment of the molecule.
    flex: np.ndarray (shape: (N, 4)).
        Flexible proper dihedrals from `sel_intcoords.list_intcoords`.
    eq: dict.
        Maps each atom index to the array of atoms equivalent to it by
        symmetry; scanned rotors on equivalent bonds share a header.
    '''

    eq = eq or {}
    flex = np.asarray(flex).reshape(-1, 4)
    by_bond: Dict[tuple, List[Sequence[int]]] = {}
    for d in flex:
        d = [int(x) for x in d]
        by_bond.setdefault(tuple(sorted((d[1], d[2]))), []).append(d)

    doubles = double_bonds(f)
    methyls, rings, scans, phis, others, double = [], [], [], [], [], []
    for (j, k), diheds in by_bond.items():
        ring = _in_ring(f, j, k)
        methyl = None
        if not ring:
            for m, a in ((j, k), (k, j)):
                if _is_methyl(f, m) and f.el[a] in METHYL_HOSTS:
                    methyl = (m, a)
        if methyl:
            m, a = methyl
            ends = set(_methyl_ends(f, m, a))
            hs = {x for x in f.nbrs[m] if f.el[x] == 'H'}
            keep = [d for d in diheds if ({d[0], d[3]} & ends) and ({d[0], d[3]} & hs)]
            methyls.extend(keep)
        elif ring and not (f.planar(j) and f.planar(k)):
            rings.extend(diheds)
        elif not ring and (min(j, k), max(j, k)) in doubles:
            if f.rings_of(j) or f.rings_of(k):
                double.extend(diheds)
            else:
                phis.append(_scan_choice(f, diheds))
        elif not ring and _joins_rigid_units(f, j, k):
            scans.append(_scan_choice(f, diheds))
        else:
            others.extend(diheds)

    scans, scan_labels = _numbered(scans, 'delta', eq)
    phis, phi_labels = _numbered(phis, 'phi', eq)
    rows = methyls + rings + scans + phis + others
    labels = ['methyls'] * len(methyls) + [RING_LABEL] * len(rings) + scan_labels + phi_labels \
        + [OTHER_LABEL] * len(others)
    params = [dict(METHYL_PARAMS) for _ in methyls + rings] + [dict(SCAN_PARAMS) for _ in scans + phis] \
        + [dict(OTHER_PARAMS) for _ in others]
    log.debug('flexible dihedrals: %d methyl, %d ring, %d single-bond scans, %d double-bond scans, '
              '%d double-bond (stiff), %d other', len(methyls), len(rings), len(scans), len(phis),
              len(double), len(others))
    return FlexPlan(
            double=np.asarray(double, dtype=int).reshape(-1, 4),
            flex=np.asarray(rows, dtype=int).reshape(-1, 4),
            labels=labels,
            params=params,
            rotors=sorted({tuple(sorted((int(d[1]), int(d[2])))) for d in scans + phis + others}),
            counts={'methyl': len(methyls), 'ring': len(rings), 'scan': len(scans), 'phi': len(phis),
                    'double': len(double), 'other': len(others)}
        )
