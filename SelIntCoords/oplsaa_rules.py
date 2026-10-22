#!/usr/bin/env python

'''
oplsaa_rules - curated OPLS-AA atom-type rules from the local chemical
environment of an atom (see `mol_features` for how that environment is read
off a geometry).

Every rule names an `opls_NNN` type of the GROMACS `oplsaa.ff` database and
tests element, neighbours, hybridisation and ring membership. Rules are tried
in order and every rule that matches is recorded: the first one types the
atom, the others are reported as alternatives. Descriptions of the types are
those of `oplsaa.ff/atomtypes.atp`.

What the downstream step needs from a type is the Lennard-Jones class (sigma,
epsilon) and the mass: charges come from the QM fit and bonded terms from
Joyce. The rules therefore separate every Lennard-Jones class of the
force field that occurs in organic molecules (alkane / aromatic / alkene /
carbonyl carbon, carbonyl / ether / ester-alkoxy / hydroxyl oxygen, the
hydrogen classes, ...) and, within one class, pick the number whose
description fits. A carbon in an ether or an alcohol is typed as an alkane
carbon, for instance: OPLS-AA has dozens of functional-group-specific sp3
carbon types (opls_157, opls_181, opls_906, ...) that differ from
opls_135-139 in charge only.

Analogues. OPLS-AA has no thiophene, furan or thiadiazole types. Rules
flagged `analogue` borrow the closest published parameters: the sulfur of
an aromatic ring is typed as thiazole S (opls_633), the carbons of
heteroaromatic rings as benzene C (opls_145), a two-coordinate aromatic N of
a five-membered ring as imidazole N (opls_557). These are the choices of the
reference examples in `examples/`; the report lists every atom typed this
way.

Conjugated carbons. In the acceptor end groups (indanone with a C=C(CN)2
arm, as in IDIC and Y6), the quinone of PM6 and the fullerene of PCBM,
every sp2 carbon is typed as benzene C (opls_145) whether or not a ring is
aromatic by the electron count, and so is an sp3 carbon between two
conjugated carbons. Four analogue rules follow that convention: a ring
ketone C between two conjugated carbons, an alkene C without H and with two
conjugated carbon neighbours (the =CH- of the same units stays an alkene C), a pyramidalised sp2 C of a cage (fullerene), and a
quaternary sp3 C with two conjugated carbon neighbours. A conjugated
carbon is a three-coordinate C (any planarity) or a two-coordinate linear
one (nitrile, alkyne). Isolated alkenes and acyclic ketones keep their
OPLS-AA types.

Rhodanine-type rings (a five-membered ring with one S, one amide N and a
ring carbon bearing a terminal O or S: rhodanine, thiazolidinone), the
acceptor end group of ZR1: every ring carbon, the C=O and C=S carbons
included, is benzene C (opls_145), the ring S and the exocyclic thione S are
thiazole S (opls_633), the N is an amide N (opls_239).
'''

from dataclasses import dataclass
from typing import Callable, List

from .mol_features import Features


@dataclass(frozen=True)
class Rule:
    '''One typing rule: `match(f, i)` is True when atom `i` has the type.'''
    type: str
    name: str
    match: Callable[[Features, int], bool]
    analogue: bool = False


def _pi_c(f, j):
    '''Conjugated carbon: three-coordinate C, or two-coordinate linear C.'''

    return f.el[j] == 'C' and (f.deg(j) == 3 or f.linear(j))


def _n_pi_c_nbrs(f, i):
    return sum(_pi_c(f, j) for j in f.nbrs[i])


def _c_ring_ketone_conjugated(f, i):
    '''C=O carbon in a ring, bonded to two conjugated carbons and nothing else.'''

    if not f.is_c_double_o(i) or _n_terminal_o_on(f, i) != 1 or not f.ring_sizes(i):
        return False
    others = [j for j in f.nbrs[i] if not f.terminal(j)]
    return len(others) == 2 and all(_pi_c(f, j) for j in others)


def _c_alkene_conjugated(f, i):
    '''Alkene C without H and with two conjugated C neighbours (=CH- stays an alkene C).'''

    return f.is_sp2_alkene(i) and f.n_h(i) == 0 and _n_pi_c_nbrs(f, i) >= 2


def _c_cage_sp2(f, i):
    '''Pyramidalised sp2 C (angle sum 335-350 degrees) with three C neighbours, in rings: fullerene.'''

    return (f.el[i] == 'C' and f.deg(i) == 3 and not f.aromatic[i]
            and all(f.el[j] == 'C' for j in f.nbrs[i]) and bool(f.ring_sizes(i))
            and 335.0 <= f.angle_sum(i) < 350.0)


def _c_quaternary_between_conjugated(f, i):
    return (f.el[i] == 'C' and f.deg(i) == 4 and all(f.el[j] == 'C' for j in f.nbrs[i])
            and _n_pi_c_nbrs(f, i) >= 2)


def _rhodanine_ring(f, i):
    '''Rhodanine-type five-membered ring through atom `i`, or None.'''

    for ring in f.rings_of(i):
        if len(ring) != 5:
            continue
        s = [j for j in ring if f.el[j] == 'S' and f.deg(j) == 2]
        n = [j for j in ring if f.el[j] == 'N' and f.deg(j) == 3 and f.planar(j)]
        c = [j for j in ring if f.el[j] == 'C']
        if len(s) == 1 and len(n) == 1 and len(c) == 3 and any(
                f.terminal(k) for j in c for k in f.nbrs[j]):
            return ring
    return None


def _s_rhodanine(f, i):
    '''Ring S, or exocyclic thione S, of a rhodanine-type ring.'''

    if f.el[i] == 'S' and f.deg(i) == 2:
        return _rhodanine_ring(f, i) is not None
    return f.el[i] == 'S' and f.deg(i) == 1 and _rhodanine_ring(f, f.nbrs[i][0]) is not None


def _carbonyl_nbr(f, i):
    '''C=O carbon bonded to atom `i`, or None.'''

    for j in f.nbrs[i]:
        if f.is_c_double_o(j):
            return j
    return None


def _n_terminal_o_on(f, c):
    return sum(f.terminal(j) and f.el[j] == 'O' for j in f.nbrs[c])


def _terminal_o_on_carbonyl(f, i):
    return f.el[i] == 'O' and f.deg(i) == 1 and f.is_c_double_o(f.nbrs[i][0]) \
        and _n_terminal_o_on(f, f.nbrs[i][0]) == 1


def _n_terminal_o(f, i):
    return sum(f.terminal(j) and f.el[j] == 'O' for j in f.nbrs[i])


def _is_amide_n(f, i):
    return f.el[i] == 'N' and f.deg(i) in (2, 3) and _carbonyl_nbr(f, i) is not None \
        and any(f.el[j] == 'O' and f.terminal(j) for j in f.nbrs[_carbonyl_nbr(f, i)])


def _is_nitro_n(f, i):
    return f.el[i] == 'N' and f.deg(i) == 3 and _n_terminal_o(f, i) == 2


def _is_nitrile_c(f, i):
    return f.el[i] == 'C' and f.deg(i) == 2 and any(
        f.el[j] == 'N' and f.deg(j) == 1 for j in f.nbrs[i])


def _lactam_ring(f, i):
    '''
    Five-membered ring through atom `i` that holds an amide N (a lactam or
    imide ring: not aromatic because its carbonyl takes the pi electrons),
    or None.
    '''

    for ring in f.rings_of(i):
        if len(ring) == 5 and any(_is_amide_n(f, j) for j in ring):
            return ring
    return None


def _aromatic_pyrrole_ring(f, i):
    '''Aromatic five-membered ring through `i` whose donor is an N, or None.'''

    for ring in f.aromatic_rings:
        if i in ring and len(ring) == 5 and any(f.el[j] == 'N' and f.deg(j) == 3 for j in ring):
            return ring
    return None


# ----------------------------------------------------------------------
# carbon
# ----------------------------------------------------------------------
def _c_alkane(n_h):
    return lambda f, i: f.el[i] == 'C' and f.deg(i) == 4 and f.n_h(i) == n_h \
        and f.nbr_el(i).count('F') == 0


def _c_fluoroalkyl(n_f, n_h=None):
    def match(f, i):
        if not (f.el[i] == 'C' and f.deg(i) == 4 and f.nbr_el(i).count('F') == n_f):
            return False
        return n_h is None or f.n_h(i) == n_h
    return match


def _c_aromatic_with(f, i, element):
    return f.el[i] == 'C' and f.aromatic[i] and any(
        f.el[j] == element and not f.aromatic[j] for j in f.nbrs[i])


def _c_aromatic_n_amine(f, i):
    return f.el[i] == 'C' and f.aromatic[i] and any(
        f.el[j] == 'N' and not f.aromatic[j] and f.deg(j) == 3 and not _is_amide_n(f, j)
        and not _is_nitro_n(f, j) for j in f.nbrs[i])


def _c_lactam_alpha(f, i):
    ring = _lactam_ring(f, i)
    return (f.el[i] == 'C' and not f.is_carbonyl(i) and f.planar(i) and ring is not None
            and any(j in ring and _is_amide_n(f, j) for j in f.nbrs[i]))


def _c_lactam_beta(f, i):
    ring = _lactam_ring(f, i)
    return (f.el[i] == 'C' and not f.is_carbonyl(i) and f.planar(i) and ring is not None
            and not any(j in ring and _is_amide_n(f, j) for j in f.nbrs[i]))


def _c_pyrrole_alpha(f, i):
    ring = _aromatic_pyrrole_ring(f, i)
    return (f.el[i] == 'C' and ring is not None
            and any(j in ring and f.el[j] == 'N' for j in f.nbrs[i]))


def _c_pyrrole_beta(f, i):
    ring = _aromatic_pyrrole_ring(f, i)
    return (f.el[i] == 'C' and ring is not None
            and not any(j in ring and f.el[j] == 'N' for j in f.nbrs[i]))


def _c_heteroaromatic(f, i):
    '''Aromatic C in a ring that also holds a heteroatom.'''

    return f.el[i] == 'C' and f.aromatic[i] and any(
        f.el[j] != 'C' for r in f.aromatic_rings if i in r for j in r)


def _c_carbonyl(kind):
    def match(f, i):
        if not f.is_c_double_o(i) or _n_terminal_o_on(f, i) != 1:
            return False
        nbr = [j for j in f.nbrs[i] if not f.terminal(j)]
        if kind == 'amide':
            return any(f.el[j] == 'N' for j in nbr)
        if kind == 'acid':
            return any(f.el[j] == 'O' and f.n_h(j) == 1 for j in nbr)
        if kind == 'ester':
            return any(f.el[j] == 'O' for j in nbr)
        if kind == 'aldehyde':
            return f.n_h(i) == 1
        return not any(f.el[j] in ('N', 'O') for j in nbr)  # ketone
    return match


def _c_alkene(n_h):
    return lambda f, i: f.is_sp2_alkene(i) and f.n_h(i) == n_h


def _c_alkyne(terminal):
    def match(f, i):
        if not (f.el[i] == 'C' and f.linear(i)) or any(f.el[j] not in ('C', 'H', 'Si') for j in f.nbrs[i]):
            return False
        return (f.n_h(i) == 1) == terminal
    return match


# ----------------------------------------------------------------------
# heteroatoms
# ----------------------------------------------------------------------
def _n_amide(n_h):
    return lambda f, i: _is_amide_n(f, i) and f.n_h(i) == n_h


def _n_aromatic_six(f, i):
    return f.el[i] == 'N' and f.deg(i) == 2 and f.aromatic[i] and 6 in f.aromatic_ring_sizes(i)


def _n_aromatic_five(f, i):
    return f.el[i] == 'N' and f.deg(i) == 2 and f.aromatic[i] and 5 in f.aromatic_ring_sizes(i)


def _n_pyrrole(f, i):
    return f.el[i] == 'N' and f.deg(i) == 3 and f.aromatic[i] and not _is_amide_n(f, i)


def _n_amine(n_h):
    return lambda f, i: (f.el[i] == 'N' and f.deg(i) == 3 and f.n_h(i) == n_h
                         and not f.aromatic[i] and not _is_amide_n(f, i) and not _is_nitro_n(f, i)
                         and not any(f.is_c_double_o(j) for j in f.nbrs[i]))


def _n_nitro(aryl):
    def match(f, i):
        if not _is_nitro_n(f, i):
            return False
        has_aryl = any(f.el[j] == 'C' and f.aromatic[j] for j in f.nbrs[i])
        return has_aryl == aryl
    return match


def _o_carbonyl(kind):
    def match(f, i):
        if not _terminal_o_on_carbonyl(f, i):
            return False
        c = f.nbrs[i][0]
        nbr = [j for j in f.nbrs[c] if j != i]
        if kind == 'amide':
            return any(f.el[j] == 'N' for j in nbr)
        if kind == 'acid':
            return any(f.el[j] == 'O' and f.n_h(j) == 1 for j in nbr)
        if kind == 'ester':
            return any(f.el[j] == 'O' for j in nbr)
        return not any(f.el[j] in ('N', 'O') for j in nbr)
    return match


def _o_hydroxyl(kind):
    def match(f, i):
        if not (f.el[i] == 'O' and f.deg(i) == 2 and f.n_h(i) == 1):
            return False
        heavy = f.heavy_nbrs(i)
        if not heavy:
            return False
        c = heavy[0]
        if kind == 'acid':
            return f.is_c_double_o(c)
        if kind == 'phenol':
            return f.el[c] == 'C' and f.aromatic[c]
        return not f.is_c_double_o(c) and not (f.el[c] == 'C' and f.aromatic[c])
    return match


def _o_ether(kind):
    def match(f, i):
        if not (f.el[i] == 'O' and f.deg(i) == 2 and f.n_h(i) == 0) or f.aromatic[i]:
            return False
        nbr = f.nbrs[i]
        if kind == 'ester':
            return any(f.is_c_double_o(j) for j in nbr)
        if kind == 'aryl':
            return any(f.el[j] == 'C' and f.aromatic[j] for j in nbr)
        return True
    return match


def _o_aromatic(f, i):
    return f.el[i] == 'O' and f.deg(i) == 2 and f.aromatic[i]


def _s_sulfonyl(f, i):
    return f.el[i] == 'S' and sum(f.terminal(j) for j in f.nbrs[i]) >= 2


def _s_sulfoxide(f, i):
    return f.el[i] == 'S' and f.deg(i) == 3 and sum(f.terminal(j) for j in f.nbrs[i]) == 1


def _s_thioether(aryl):
    def match(f, i):
        if not (f.el[i] == 'S' and f.deg(i) == 2 and f.n_h(i) == 0 and not f.aromatic[i]
                and not any(f.el[j] == 'S' for j in f.nbrs[i])):
            return False
        return any(f.el[j] == 'C' and f.aromatic[j] for j in f.nbrs[i]) == aryl
    return match


# ----------------------------------------------------------------------
# hydrogen: typed from the atom it is bonded to
# ----------------------------------------------------------------------
def _h_on(test):
    def match(f, i):
        if f.el[i] != 'H' or f.deg(i) != 1:
            return False
        return test(f, f.nbrs[i][0])
    return match


def _c_sp3_alpha_to_ester_o(f, p):
    return f.el[p] == 'C' and f.deg(p) == 4 and any(
        f.el[j] == 'O' and any(f.is_c_double_o(k) for k in f.nbrs[j]) for j in f.nbrs[p])


def _c_sp3_alpha_to_amine_n(f, p):
    return f.el[p] == 'C' and f.deg(p) == 4 and any(
        f.el[j] == 'N' and f.deg(j) == 3 and not f.aromatic[j] and not _is_amide_n(f, j)
        and not _is_nitro_n(f, j) for j in f.nbrs[p])


RULES: List[Rule] = [
    # ---- halogens
    Rule('opls_719', 'F on aromatic C', lambda f, i: f.el[i] == 'F' and f.nbrs[i]
         and f.el[f.nbrs[i][0]] == 'C' and f.aromatic[f.nbrs[i][0]]),
    Rule('opls_965', 'F on perfluoroalkyl C', lambda f, i: f.el[i] == 'F' and f.nbrs[i]
         and f.el[f.nbrs[i][0]] == 'C' and f.nbr_el(f.nbrs[i][0]).count('F') >= 2),
    Rule('opls_956', 'F on alkyl C', lambda f, i: f.el[i] == 'F' and f.nbrs[i]
         and f.el[f.nbrs[i][0]] == 'C'),
    Rule('opls_264', 'Cl on aromatic C', lambda f, i: f.el[i] == 'Cl' and f.nbrs[i]
         and f.aromatic[f.nbrs[i][0]]),
    Rule('opls_151', 'Cl on alkyl C', lambda f, i: f.el[i] == 'Cl'),
    Rule('opls_730', 'Br on aromatic C', lambda f, i: f.el[i] == 'Br' and f.nbrs[i]
         and f.aromatic[f.nbrs[i][0]]),
    Rule('opls_722', 'Br on alkyl C', lambda f, i: f.el[i] == 'Br'),
    Rule('opls_732', 'I on aromatic C', lambda f, i: f.el[i] == 'I' and f.nbrs[i]
         and f.aromatic[f.nbrs[i][0]]),

    # ---- sulfur
    Rule('opls_633', 'aromatic S', lambda f, i: f.el[i] == 'S' and f.aromatic[i], analogue=True),
    Rule('opls_633', 'S of a rhodanine-type ring (ring S, thione S)', _s_rhodanine, analogue=True),
    Rule('opls_493', 'sulfonyl S', _s_sulfonyl),
    Rule('opls_496', 'sulfoxide S', _s_sulfoxide),
    Rule('opls_203', 'disulfide S', lambda f, i: f.el[i] == 'S' and f.deg(i) == 2
         and any(f.el[j] == 'S' for j in f.nbrs[i])),
    Rule('opls_200', 'thiol S', lambda f, i: f.el[i] == 'S' and f.deg(i) == 2 and f.n_h(i) == 1),
    Rule('opls_222', 'aryl thioether S', _s_thioether(True)),
    Rule('opls_202', 'thioether S', _s_thioether(False)),
    Rule('opls_202', 'thiocarbonyl S', lambda f, i: f.terminal(i) and f.el[i] == 'S'),

    # ---- oxygen
    Rule('opls_761', 'nitro O', lambda f, i: f.el[i] == 'O' and f.deg(i) == 1
         and _is_nitro_n(f, f.nbrs[i][0])),
    Rule('opls_497', 'sulfoxide O', lambda f, i: f.el[i] == 'O' and f.deg(i) == 1
         and _s_sulfoxide(f, f.nbrs[i][0])),
    Rule('opls_494', 'sulfonyl O', lambda f, i: f.el[i] == 'O' and f.deg(i) == 1
         and f.el[f.nbrs[i][0]] == 'S'),
    Rule('opls_272', 'carboxylate O', lambda f, i: f.el[i] == 'O' and f.deg(i) == 1
         and _n_terminal_o_on(f, f.nbrs[i][0]) == 2 and f.el[f.nbrs[i][0]] == 'C'),
    Rule('opls_236', 'amide / imide C=O oxygen', _o_carbonyl('amide')),
    Rule('opls_269', 'carboxylic acid C=O oxygen', _o_carbonyl('acid')),
    Rule('opls_466', 'ester C=O oxygen', _o_carbonyl('ester')),
    Rule('opls_281', 'ketone / aldehyde oxygen', _o_carbonyl('ketone')),
    Rule('opls_566', 'aromatic ring O (furan)', _o_aromatic),
    Rule('opls_268', 'carboxylic acid OH', _o_hydroxyl('acid')),
    Rule('opls_167', 'phenol OH', _o_hydroxyl('phenol')),
    Rule('opls_154', 'alcohol OH', _o_hydroxyl('alcohol')),
    Rule('opls_467', 'ester alkoxy O', _o_ether('ester')),
    Rule('opls_179', 'aryl ether O', _o_ether('aryl')),
    Rule('opls_180', 'dialkyl ether O', _o_ether('alkyl')),

    # ---- nitrogen
    Rule('opls_760', 'alkyl nitro N', _n_nitro(False)),
    Rule('opls_767', 'aryl nitro N', _n_nitro(True)),
    Rule('opls_753', 'nitrile N', lambda f, i: f.el[i] == 'N' and f.deg(i) == 1
         and _is_nitrile_c(f, f.nbrs[i][0])),
    Rule('opls_239', 'N of a rhodanine-type ring', lambda f, i: f.el[i] == 'N' and f.deg(i) == 3
         and _rhodanine_ring(f, i) is not None),
    Rule('opls_237', 'primary amide N', _n_amide(2)),
    Rule('opls_238', 'secondary amide N', _n_amide(1)),
    Rule('opls_239', 'tertiary amide / imide N', _n_amide(0)),
    Rule('opls_520', 'aromatic N, six-membered ring (pyridine)', _n_aromatic_six),
    Rule('opls_520', 'azo N (N=N), typed as pyridine N', lambda f, i: f.el[i] == 'N' and f.deg(i) == 2
         and not f.aromatic[i] and any(f.el[j] == 'N' and f.deg(j) == 2 for j in f.nbrs[i]), analogue=True),
    Rule('opls_557', 'aromatic N, five-membered ring', _n_aromatic_five, analogue=True),
    Rule('opls_542', 'pyrrole-type N', _n_pyrrole),
    Rule('opls_900', 'primary amine N', _n_amine(2)),
    Rule('opls_901', 'secondary amine N', _n_amine(1)),
    Rule('opls_902', 'tertiary amine N', _n_amine(0)),

    # ---- carbon, functional groups first
    Rule('opls_145', 'C of a rhodanine-type ring (C=O, C=S and ring C=C alike)',
         lambda f, i: f.el[i] == 'C' and _rhodanine_ring(f, i) is not None, analogue=True),
    Rule('opls_754', 'nitrile C', _is_nitrile_c),
    Rule('opls_925', 'terminal alkyne C', _c_alkyne(True)),
    Rule('opls_931', 'internal alkyne C', _c_alkyne(False)),
    Rule('opls_234', 'amide / imide C=O carbon bonded to an aromatic C',
         lambda f, i: _c_carbonyl('amide')(f, i)
         and any(f.el[j] == 'C' and f.aromatic[j] for j in f.nbrs[i])),
    Rule('opls_271', 'carboxylate C', lambda f, i: f.el[i] == 'C' and f.deg(i) == 3
         and _n_terminal_o_on(f, i) == 2),
    Rule('opls_235', 'amide / imide C=O carbon', _c_carbonyl('amide')),
    Rule('opls_267', 'carboxylic acid C', _c_carbonyl('acid')),
    Rule('opls_465', 'ester C', _c_carbonyl('ester')),
    Rule('opls_145', 'ring ketone C between two conjugated C', _c_ring_ketone_conjugated,
         analogue=True),
    Rule('opls_277', 'aldehyde C', _c_carbonyl('aldehyde')),
    Rule('opls_280', 'ketone C', _c_carbonyl('ketone')),
    Rule('opls_543', 'sp2 C of a lactam ring bonded to the ring N (as pyrrole C2)', _c_lactam_alpha,
         analogue=True),
    Rule('opls_544', 'sp2 C of a lactam ring not bonded to the ring N (as pyrrole C3)', _c_lactam_beta,
         analogue=True),
    Rule('opls_543', 'pyrrole C2', _c_pyrrole_alpha),
    Rule('opls_544', 'pyrrole C3', _c_pyrrole_beta),
    # aromatic carbons, specific substituent first
    Rule('opls_718', 'aromatic C bonded to F', lambda f, i: _c_aromatic_with(f, i, 'F')),
    Rule('opls_263', 'aromatic C bonded to Cl', lambda f, i: _c_aromatic_with(f, i, 'Cl')),
    Rule('opls_729', 'aromatic C bonded to Br', lambda f, i: _c_aromatic_with(f, i, 'Br')),
    Rule('opls_731', 'aromatic C bonded to I', lambda f, i: _c_aromatic_with(f, i, 'I')),
    Rule('opls_166', 'aromatic C bonded to OH', lambda f, i: f.el[i] == 'C' and f.aromatic[i]
         and any(f.el[j] == 'O' and f.n_h(j) == 1 and not f.aromatic[j] for j in f.nbrs[i])),
    Rule('opls_199', 'aromatic C bonded to ether O', lambda f, i: f.el[i] == 'C' and f.aromatic[i]
         and any(f.el[j] == 'O' and f.deg(j) == 2 and f.n_h(j) == 0 and not f.aromatic[j]
                 for j in f.nbrs[i])),
    Rule('opls_916', 'aromatic C bonded to amine N', _c_aromatic_n_amine),
    Rule('opls_768', 'aromatic C bonded to nitro N', lambda f, i: f.el[i] == 'C' and f.aromatic[i]
         and any(_is_nitro_n(f, j) for j in f.nbrs[i])),
    Rule('opls_145', 'aromatic C in a heteroaromatic ring', _c_heteroaromatic, analogue=True),
    Rule('opls_145', 'aromatic C', lambda f, i: f.el[i] == 'C' and f.aromatic[i]),
    Rule('opls_145', 'alkene C without H and with two conjugated C neighbours', _c_alkene_conjugated,
         analogue=True),
    Rule('opls_145', 'pyramidalised sp2 C of a cage (fullerene)', _c_cage_sp2, analogue=True),
    Rule('opls_141', 'alkene C (R2C=)', _c_alkene(0)),
    Rule('opls_142', 'alkene C (RHC=)', _c_alkene(1)),
    Rule('opls_143', 'alkene C (H2C=)', _c_alkene(2)),
    # sp3 carbon: alkane numbers, perfluoro / fluoroalkyl variants
    Rule('opls_145', 'quaternary sp3 C between two conjugated C', _c_quaternary_between_conjugated,
         analogue=True),
    Rule('opls_964', 'CF4', _c_fluoroalkyl(4)),
    Rule('opls_961', 'CF3', _c_fluoroalkyl(3)),
    Rule('opls_962', 'CF2', _c_fluoroalkyl(2)),
    Rule('opls_957', 'CH2F', _c_fluoroalkyl(1, 2)),
    Rule('opls_959', 'CHF', _c_fluoroalkyl(1, 1)),
    Rule('opls_960', 'CF', _c_fluoroalkyl(1, 0)),
    Rule('opls_138', 'alkane CH4', _c_alkane(4)),
    Rule('opls_135', 'alkane CH3', _c_alkane(3)),
    Rule('opls_136', 'alkane CH2', _c_alkane(2)),
    Rule('opls_137', 'alkane CH', _c_alkane(1)),
    Rule('opls_139', 'alkane C', _c_alkane(0)),

    # ---- hydrogen
    Rule('opls_270', 'H on carboxylic acid O', _h_on(lambda f, p: f.el[p] == 'O' and f.deg(p) == 2
         and any(f.is_c_double_o(j) for j in f.nbrs[p]))),
    Rule('opls_168', 'H on phenol O', _h_on(lambda f, p: f.el[p] == 'O' and f.deg(p) == 2
         and any(f.el[j] == 'C' and f.aromatic[j] for j in f.nbrs[p]))),
    Rule('opls_155', 'H on alcohol O', _h_on(lambda f, p: f.el[p] == 'O' and f.deg(p) == 2)),
    Rule('opls_240', 'H on primary amide N', _h_on(lambda f, p: _is_amide_n(f, p) and f.n_h(p) == 2)),
    Rule('opls_241', 'H on secondary amide N', _h_on(lambda f, p: _is_amide_n(f, p))),
    Rule('opls_545', 'H on pyrrole-type N', _h_on(lambda f, p: f.el[p] == 'N' and f.aromatic[p])),
    Rule('opls_909', 'H on primary amine N', _h_on(lambda f, p: f.el[p] == 'N' and f.deg(p) == 3
         and f.n_h(p) == 2)),
    Rule('opls_910', 'H on secondary amine N', _h_on(lambda f, p: f.el[p] == 'N' and f.deg(p) == 3)),
    Rule('opls_204', 'H on thiol S', _h_on(lambda f, p: f.el[p] == 'S' and f.deg(p) == 2)),
    Rule('opls_146', 'H on aromatic C', _h_on(lambda f, p: f.el[p] == 'C' and f.aromatic[p])),
    Rule('opls_279', 'H on carbonyl C', _h_on(lambda f, p: f.is_c_double_o(p))),
    Rule('opls_144', 'H on alkene C', _h_on(lambda f, p: f.is_sp2_alkene(p))),
    Rule('opls_926', 'H on alkyne C', _h_on(lambda f, p: f.el[p] == 'C' and f.linear(p))),
    Rule('opls_469', 'H on C bonded to an ester O', _h_on(_c_sp3_alpha_to_ester_o)),
    Rule('opls_911', 'H on C bonded to an amine N', _h_on(_c_sp3_alpha_to_amine_n)),
    Rule('opls_140', 'alkane H', _h_on(lambda f, p: f.el[p] == 'C')),
]


def _cationic_aromatic_n(f, i):
    '''Planar three-coordinate N in a six-ring of otherwise pi atoms (pyridinium, N-oxide).'''

    if not (f.el[i] == 'N' and f.deg(i) == 3 and f.planar(i)):
        return False
    for ring in f.rings_of(i):
        if len(ring) == 6:
            roles = [f.pi_role(j) for j in ring if j != i]
            if '' not in roles and 'd' not in roles:
                return True
    return False


# atoms whose chemistry the rules do not cover and would mistype silently
REFUSE = [
    ('three-coordinate planar N in a six-membered ring of sp2 atoms '
     '(pyridinium-type cation): charged species are not supported', _cationic_aromatic_n),
]
