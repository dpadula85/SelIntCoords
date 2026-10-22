#!/usr/bin/env python

'''
perceive_types - force-field atom types from a bare geometry.

Takes a `.xyz`, guesses the bonds as `make_top` does, and assigns every atom
a type of a GROMACS force field in `<gromacs>/share/gromacs/top`. Writes the
minimal topology `make_top` starts from (`[ defaults ]`, `[ atomtypes ]` with
mass and Lennard-Jones parameters from the force field, `[ atoms ]` with zero
charges: charges come from the QM fit) and a report of how each atom was
typed and where the choice was not clear-cut.

Two engines, in this order:

1. curated rules (`oplsaa_rules`), written for the chemistry of organic
   semiconductors and checked against the types the reference examples in
   `examples/` were built with: available for OPLS-AA;
2. types learned from the residue templates of the force field
   (`learned_types`), for atoms no rule covers and, for every atom, as a
   cross-check: a disagreement in Lennard-Jones class is reported.

An atom neither engine can type is an error, never a guess.
'''

import sys
import logging
import re
import argparse as arg
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ffdb import ForceField
from .mol_features import Features
from .learned_types import LearnedTyper, Prediction
from .oplsaa_rules import RULES as OPLSAA_RULES, REFUSE as OPLSAA_REFUSE, Rule

log = logging.getLogger(__name__)

# force field -> (rules, refusals)
RULESETS = {'oplsaa': (OPLSAA_RULES, OPLSAA_REFUSE)}

# learned labels of radius 1 say little more than "an atom of this element
# with these neighbours": used, but reported; radius 0 is not used at all
MIN_TRUSTED_RADIUS = 2

# alternatives listed per atom in the report
MAX_ALTERNATIVES = 4

# written in place of a parameter the force field does not have
MARKER = 'FILL_ME'


@dataclass
class Assignment:
    '''Type of one atom and how it was obtained.'''
    index: int                       # 0-based
    element: str
    type: str
    source: str                      # 'rule' or 'learned'
    rule: str = ''                   # rule name, or 'learned (radius r)'
    analogue: bool = False
    alternatives: List[Tuple[str, str]] = field(default_factory=list)  # (type, origin)
    flags: List[str] = field(default_factory=list)   # reasons to check the atom


def perceive(f: Features, ff: ForceField, rules: Optional[List[Rule]] = None,
             learned: Optional[LearnedTyper] = None, refuse=()) -> List[Assignment]:
    '''
    Assign a force-field type to every atom.

    Parameters
    ----------
    f: Features.
        The molecule.
    ff: ForceField.
        Source of the types, masses and Lennard-Jones parameters.
    rules: list of Rule, optional.
        Curated rules, tried in order; the first match types the atom, later
        matches of a different Lennard-Jones class are listed as alternatives.
    learned: LearnedTyper, optional.
        Types learned from the residue templates: used for atoms no rule
        types and as a cross-check on all others.
    refuse: list of (reason, predicate).
        Environments the rules do not cover and would mistype (charged
        species); an atom matching one raises instead of being typed.

    Returns
    -------
    out: list of Assignment.
        One per atom, in atom order.

    Raises
    ------
    ValueError
        If an atom gets no type from either engine, or is in a refused
        environment.
    '''

    refused = [(reason, i) for i in range(f.n) for reason, test in refuse if test(f, i)]
    if refused:
        raise ValueError('; '.join('atom %s: %s' % (f.summary(i), reason) for reason, i in refused[:10]))

    pred: Dict[int, Optional[Prediction]] = learned.predict(f) if learned is not None else {}
    out, untyped = [], []

    for i in range(f.n):
        matched = [r for r in (rules or []) if r.match(f, i)]
        p = pred.get(i)
        # a type of another element, or a label that is only the element (a
        # draw among every type of that element), is no answer
        if p is not None and (ff.atomtypes[p.type].element != f.el[i] or p.radius == 0):
            p = None

        if matched:
            first = matched[0]
            if first.type not in ff.atomtypes:
                raise ValueError('rule %r names type %s, absent from %s' % (first.name, first.type, ff.name))
            a = Assignment(i, f.el[i], first.type, 'rule', first.name, first.analogue)
        elif p is not None:
            a = Assignment(i, f.el[i], p.type, 'learned', 'learned (radius %d)' % p.radius)
            if p.radius < MIN_TRUSTED_RADIUS:
                a.flags.append('low confidence: only the element and the elements of its %d neighbours matched'
                               % f.deg(i))
            for t in p.candidates:
                if t != p.type:
                    a.alternatives.append((t, 'learned'))
        elif f.el[i] not in ff.elements or f.el[i] in ff.extra or f.el[i] in ff.placeholders:
            # an element the force field does not cover (Se): a type named after
            # the element, with the parameters left to fill in
            at = ff.atomtypes.get(f.el[i]) or ff.add_missing_element(f.el[i])
            if f.el[i] in ff.extra:
                a = Assignment(i, f.el[i], at.name, 'extra', 'parameters outside %s' % ff.name, analogue=True)
            else:
                a = Assignment(i, f.el[i], at.name, 'placeholder', 'no %s type for %s' % (ff.name, f.el[i]))
                a.flags.append(at.description)
            out.append(a)
            continue
        else:
            untyped.append(i)
            continue

        if p is not None and p.radius >= MIN_TRUSTED_RADIUS and ff.lj_key(p.type) != ff.lj_key(a.type):
            a.alternatives.append((p.type, 'learned (radius %d)' % p.radius))
            a.flags.append('rules and learned types disagree on the Lennard-Jones class')
        out.append(a)

    if untyped:
        raise ValueError(
            'no type for atom(s): ' + '; '.join(f.summary(i) for i in untyped[:10])
            + (' ...' if len(untyped) > 10 else ''))
    out.sort(key=lambda a: a.index)
    return out


def class_names(assign: List[Assignment], ff: ForceField, f: Features) -> List[str]:
    '''
    One short name per local environment (type + elements of the neighbours),
    from the force field's own `bond_type` (CA, HA, S, ...), numbered when a
    molecule has several environments under the same name: two aromatic
    carbons, one bonded to S and one not, become CA1 and CA2. The `.top`
    keeps the force-field type in a comment.
    '''

    envs: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    per_base: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}
    for a in assign:
        key = (a.type, tuple(f.nbr_el(a.index)))
        if key not in envs:
            envs[key] = len(envs)
            per_base.setdefault(ff.atomtypes[a.type].bond_type, []).append(key)

    names = {}
    for base, keys in per_base.items():
        for k, key in enumerate(keys):
            names[key] = base if len(keys) == 1 else '%s%d' % (base, k + 1)
    return [names[(a.type, tuple(f.nbr_el(a.index)))] for a in assign]


def write_top(path, assign: List[Assignment], names: List[str], ff: ForceField,
              resname: str = 'MOL', source: str = '', marker: Optional[str] = MARKER):
    '''
    Write the minimal topology `make_top` takes as input.

    Parameters
    ----------
    path: str or Path.
    assign: list of Assignment.
    names: list of str.
        Atomtype name of each atom (the force-field type, or `class_names`).
    ff: ForceField.
    resname: str.
        Residue and molecule name.
    source: str.
        Geometry file, for the header.
    marker: str or None.
        Text written for a missing sigma or epsilon (see `ForceField.add_placeholder`);
        None writes NaN, which `make_top` reads and replaces by `MARKER` in its output.
    '''

    seen = {}
    for a, n in zip(assign, names):
        prev = seen.setdefault(n, a.type)
        if prev != a.type:
            raise ValueError('name %s given to both %s and %s' % (n, prev, a.type))

    lines = [
        '; Atom types perceived from %s' % source,
        '; force field: %s (GROMACS database); charges are zero, to be set from QM.' % ff.name,
        '',
        '[ defaults ]',
        '; nbfunc   comb-rule   gen-pairs  fudgeLJ  fudgeQQ',
        '     1       3           no         1.0       0.0',
        '',
        '[ atomtypes ]',
        '; name     mass    charge ptype    sigma       epsilon',
    ]
    for n, t in seen.items():
        at = ff.atomtypes[t]
        desc = ('  ' + at.description) if at.description else ''
        if t in ff.placeholders and marker is not None:
            lines.append('%-7s %9.4f %7.3f   A  %12s %12s  ; %s%s'
                         % (n, at.mass, 0.0, marker, marker, t, desc))
        else:
            lines.append('%-7s %9.4f %7.3f   A  %12.5e %12.5e  ; %s%s'
                         % (n, at.mass, 0.0, at.sigma, at.epsilon, t, desc))
    lines += [
        '',
        '[ moleculetype ]',
        '; Name     nrexcl',
        '%-10s 3' % resname,
        '',
        '[ atoms ]',
        '; nr  type  resnr residue  atom  cgnr    charge     mass',
    ]
    for a, n in zip(assign, names):
        lines.append('%5d %-7s %3d %-6s %-7s %5d %10.6f %9.4f'
                     % (a.index + 1, n, 1, resname, '%s%d' % (a.element, a.index + 1), a.index + 1,
                        0.0, ff.atomtypes[a.type].mass))
    lines += ['', '[ system ]', '; Name', resname, '', '[ molecules ]', '; Compound   #mols',
              '%-10s 1' % resname, '']
    Path(path).write_text('\n'.join(lines))


def write_report(path, assign: List[Assignment], names: List[str], ff: ForceField,
                 source: str = ''):
    '''
    Write the per-atom typing report: the table, the atoms typed by analogy
    (one line per rule) and the atoms to check.
    '''

    to_check = [a for a in assign if a.flags or a.alternatives]
    analogues: Dict[Tuple[str, str], List[int]] = {}
    for a in assign:
        if a.analogue:
            analogues.setdefault((a.type, a.rule), []).append(a.index + 1)

    lines = ['Atom types perceived from %s (%s)' % (source, ff.name),
             '%d atoms: %d typed by rule, %d learned from residue templates, '
             '%d by analogy, %d to check'
             % (len(assign), sum(a.source == 'rule' for a in assign),
                sum(a.source == 'learned' for a in assign),
                sum(a.analogue for a in assign), len(to_check)), '']

    lines.append('%5s %-3s %-9s %-7s %s' % ('atom', 'el', 'type', 'name', 'rule'))
    for a, n in zip(assign, names):
        lines.append('%5d %-3s %-9s %-7s %s' % (a.index + 1, a.element, a.type, n, a.rule))

    if analogues:
        lines += ['', 'Typed by analogy (no dedicated type in %s, the closest published one used):' % ff.name]
        for (t, rule), atoms in analogues.items():
            at = ff.atomtypes[t]
            lines.append('  %s  %s  <- %s: atoms %s'
                         % (t, at.description or at.bond_type, rule, ' '.join(map(str, atoms))))

    if to_check:
        lines += ['', 'Atoms to check:']
        for a in to_check:
            lines.append('%5d %-3s %s (%s)' % (a.index + 1, a.element, a.type, a.rule))
            for flag in a.flags:
                lines.append('        - %s' % flag)
            for t, origin in a.alternatives[:MAX_ALTERNATIVES]:
                at = ff.atomtypes[t]
                lines.append('        - alternative %s [%s]: %s' % (t, origin, at.description or at.bond_type))
            if len(a.alternatives) > MAX_ALTERNATIVES:
                lines.append('        - ... and %d more' % (len(a.alternatives) - MAX_ALTERNATIVES))
    Path(path).write_text('\n'.join(lines) + '\n')


def perceive_to_top(xyz, ff_name: str = 'oplsaa', gmx_top=None, out=None,
                    class_name: bool = True, resname: str = 'MOL',
                    report=None, numeric_placeholders: bool = False,
                    marked_out=None) -> Tuple[Path, List[Assignment]]:
    '''
    Perceive the types of `xyz` and write the minimal `.top` and the report.

    Parameters
    ----------
    xyz: str or Path.
        Geometry.
    ff_name: str.
        Force field, e.g. `oplsaa` (`<name>.ff` in the GROMACS database).
    gmx_top: str or Path, optional.
        GROMACS `top` directory (see `ffdb.find_gmx_top`).
    out: str or Path, optional.
        Output `.top`; `<xyz stem>_types.top` by default. The report goes next
        to it, with the extension `.types`.
    class_name: bool.
        Atomtype names from `class_names` (CA1, HC, ...), the default, instead
        of the force-field types (opls_145); the type stays in the comment of
        the atomtype line. Atom names are the element and the atom number.
    resname: str.
        Residue and molecule name.
    report: str or Path, optional.
        Report file; the extension `.types` next to `out` by default.
    numeric_placeholders: bool.
        Write NaN instead of `MARKER` for missing parameters, so that the file
        can be read as a topology (what `make_top` does internally).
    marked_out: str or Path, optional.
        If an element has no type in the force field, also write the topology
        with `MARKER` to this path, for the user to fill in.

    Returns
    -------
    top: Path.
        The topology written.
    assign: list of Assignment.
    '''

    xyz = Path(xyz)
    top = Path(out) if out else xyz.with_name(xyz.stem + '_types.top')
    ff = ForceField(ff_name, gmx_top)
    f = Features.from_xyz(xyz)
    rules, refuse = RULESETS.get(ff.name, (None, ()))
    if rules is None:
        log.warning('no curated rules for %s: types come from the residue templates only', ff.name)
    learned = LearnedTyper(ff) if ff.residues else None

    assign = perceive(f, ff, rules, learned, refuse)
    names = class_names(assign, ff, f) if class_name else [a.type for a in assign]
    write_top(top, assign, names, ff, resname, xyz.name, marker=None if numeric_placeholders else MARKER)
    if marked_out is not None and ff.placeholders:
        write_top(marked_out, assign, names, ff, resname, xyz.name, marker=MARKER)
    write_report(Path(report) if report else top.with_suffix('.types'), assign, names, ff, xyz.name)
    return top, assign


def atomtype_comments(path) -> Dict[str, str]:
    '''Comment (force-field type and description) of each atomtype line of a `.top`.'''

    out, section = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\s*\[\s*(\w+)\s*\]', line)
        if m:
            section = m.group(1)
        elif section == 'atomtypes' and ';' in line and not line.lstrip().startswith(';'):
            body, comment = line.split(';', 1)
            out[body.split()[0]] = comment.strip()
    return out


def add_atomtype_comments(path, comments: Dict[str, str]):
    '''Append `; <comment>` to the atomtype lines of a `.top` that have none.'''

    lines, section = [], None
    for line in Path(path).read_text().split('\n'):
        m = re.match(r'\s*\[\s*(\w+)\s*\]', line)
        if m:
            section = m.group(1)
        elif section == 'atomtypes' and line.strip() and not line.lstrip().startswith(';') \
                and ';' not in line and line.split()[0] in comments:
            line = '%s  ; %s' % (line.rstrip(), comments[line.split()[0]])
        lines.append(line)
    Path(path).write_text('\n'.join(lines))


def mark_missing_parameters(path, elements):
    '''
    Replace the NaN that stands for a missing Lennard-Jones parameter in a
    topology written by `make_top` by `MARKER`, and say what is to be filled in.

    Parameters
    ----------
    path: str or Path.
        Topology to edit in place.
    elements: iterable of str.
        Elements without parameters in the force field.
    '''

    import re

    text = Path(path).read_text()
    note = ('; %s: no force-field parameters for %s. Fill in sigma and epsilon of its atomtype below,\n'
            '; and the 1-4 pair parameters of every pair that involves it (the pairs written as %s).\n'
            % (MARKER, ', '.join(sorted(elements)), MARKER))
    text = re.sub(r'\bnan\b', MARKER, text)
    head, sep, tail = text.partition('[ atomtypes ]')
    Path(path).write_text(head + note + sep + tail if sep else note + text)


def add_perception_args(parser):
    '''Command-line options shared by `perceive_types` and `make_top`.'''

    grp = parser.add_argument_group('Atom type perception')
    grp.add_argument('--ff', default='oplsaa',
                     help='Force field of the GROMACS database (default: oplsaa).')
    grp.add_argument('--gmx-top', default=None,
                     help='GROMACS share/gromacs/top directory (default: $GMXLIB, $GMXDATA/top, '
                          'or next to the gmx executable).')
    grp.add_argument('--ff-type-names', action='store_true',
                     help='Name atomtypes by force-field type (opls_145) instead of by local '
                          'environment (CA1, CA2, ...); the type is in the comment either way.')
    grp.add_argument('--resname', default='MOL', help='Residue and molecule name (default: MOL).')


def main(argv: Optional[List[str]] = None) -> int:
    '''Entry point of the `perceive_types` console script.'''

    parser = arg.ArgumentParser(
        description='Assign force-field atom types from a geometry and write the minimal .top.')
    parser.add_argument('-m', '--mol', required=True, help='Molecule coordinates (.xyz).')
    parser.add_argument('-o', '--out', default=None,
                        help='Output .top (default: <mol>_types.top; the report is its .types).')
    parser.add_argument('-v', '--verbose', action='store_true')
    add_perception_args(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format='%(levelname)s %(name)s: %(message)s')
    top, assign = perceive_to_top(args.mol, args.ff, args.gmx_top, args.out,
                                  not args.ff_type_names, args.resname)
    n_flag = sum(bool(a.flags or a.alternatives) for a in assign)
    print('Topology saved to: %s (%d atoms, %d to check, see %s)'
          % (top, len(assign), n_flag, top.with_suffix('.types')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
