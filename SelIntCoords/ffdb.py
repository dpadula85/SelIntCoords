#!/usr/bin/env python

'''
ffdb - read a GROMACS force-field directory (`<top>/<name>.ff`).

The database gives, per atom type, the element, the mass and the Lennard-Jones
parameters (`ffnonbonded.itp`), a one-line free-text description
(`atomtypes.atp`, OPLS-AA only) and, per residue, the bonded graph with the
type of every atom (`*.rtp`). SI units as in GROMACS: sigma in nm, epsilon in
kJ/mol. A force field written with combination rule 1 (C6, C12) is converted
to (sigma, epsilon) on reading.
'''

import os
import re
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Element symbols by atomic number, H to Xe
SYMBOLS = dict(enumerate((
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
    'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe').split(), start=1))


@dataclass(frozen=True)
class AtomType:
    '''One `[ atomtypes ]` line of `ffnonbonded.itp`.'''
    name: str
    bond_type: str
    z: int
    mass: float
    charge: float
    ptype: str
    sigma: float
    epsilon: float
    description: str = ''

    @property
    def element(self) -> str:
        return SYMBOLS.get(self.z, str(self.z))


@dataclass
class Residue:
    '''One `[ name ]` block of an `.rtp` file.

    `atoms` maps atom name -> type name; `bonds` is a list of atom-name pairs,
    where a leading `-` or `+` marks an atom of the previous/next residue.
    '''
    name: str
    atoms: Dict[str, str] = field(default_factory=dict)
    bonds: List[Tuple[str, str]] = field(default_factory=list)
    source: str = ''


# force fields shipped with the package (see data/top/README.md)
BUNDLED_TOP = Path(__file__).resolve().parent / 'data' / 'top'


def find_gmx_top(gmx_top=None, ff_name: Optional[str] = None) -> Path:
    '''
    Locate the directory that holds the `<name>.ff` force-field folders.

    Parameters
    ----------
    gmx_top: str or Path, optional.
        Explicit directory. If None, the force fields shipped with the
        package are used when they include `ff_name`; otherwise `$GMXLIB`,
        then `$GMXDATA/top`, then the `share/gromacs/top` next to the `gmx`
        executable are tried.
    ff_name: str, optional.
        Force field wanted (`oplsaa`); only directories that contain it are
        accepted.

    Returns
    -------
    top: Path.
        The directory.
    '''

    name = None if ff_name is None else (ff_name[:-3] if ff_name.endswith('.ff') else ff_name)

    cands = []
    if gmx_top is not None:
        cands.append(Path(gmx_top))
    else:
        cands.append(BUNDLED_TOP)
        if os.environ.get('GMXLIB'):
            cands.append(Path(os.environ['GMXLIB']))
        if os.environ.get('GMXDATA'):
            cands.append(Path(os.environ['GMXDATA']) / 'top')
        for exe in ('gmx', 'gmx_d', 'gmx_mpi'):
            path = shutil.which(exe)
            if path:
                cands.append(Path(path).resolve().parent.parent / 'share' / 'gromacs' / 'top')

    for cand in cands:
        if not cand.is_dir():
            continue
        # an explicit directory is returned as it is, so that a missing force
        # field is reported with the list of those it has
        if gmx_top is not None or name is None or (cand / (name + '.ff')).is_dir():
            return cand

    raise FileNotFoundError(
        'no topology database with %s found (tried: %s); pass the GROMACS '
        'share/gromacs/top directory (--gmx-top) or set $GMXLIB'
        % ('force field %r' % name if name else 'a force field', ', '.join(map(str, cands))))


def _strip(line: str) -> str:
    return line.split(';')[0].strip()


def _sections(path: Path):
    '''Yield (section name, list of non-empty, comment-free lines).'''

    sec = None
    lines = []
    for raw in path.read_text(errors='replace').splitlines():
        line = _strip(raw)
        if not line or line.startswith('#'):
            continue
        m = re.match(r'\[\s*([^\]]+?)\s*\]$', line)
        if m:
            if sec is not None:
                yield sec, lines
            sec, lines = m.group(1), []
        else:
            lines.append(line)
    if sec is not None:
        yield sec, lines


class ForceField:
    '''
    A GROMACS force-field directory.

    Parameters
    ----------
    name: str.
        Directory name with or without the `.ff` suffix (e.g. `oplsaa`).
    gmx_top: str or Path, optional.
        Directory with the `.ff` folders (see `find_gmx_top`); the OPLS-AA
        files shipped with the package by default.

    Attributes
    ----------
    atomtypes: dict.
        Type name -> `AtomType`.
    residues: list of Residue.
        Every residue of every `.rtp` file, with the atom types of the
        force field it belongs to.
    comb_rule: int.
        GROMACS combination rule from `forcefield.itp`.
    '''

    def __init__(self, name: str, gmx_top=None):
        self.name = name[:-3] if name.endswith('.ff') else name
        self.top = find_gmx_top(gmx_top, self.name)
        self.path = self.top / (self.name + '.ff')
        if not self.path.is_dir():
            avail = sorted(p.name[:-3] for p in self.top.glob('*.ff'))
            raise FileNotFoundError(
                'force field %r not found in %s (available: %s)'
                % (self.name, self.top, ', '.join(avail)))

        self.comb_rule = self._read_comb_rule()
        self.atomtypes = self._read_atomtypes()
        self._read_descriptions()
        self._residues = None
        self.placeholders: set = set()    # elements added without parameters
        self.extra: set = set()           # elements added with parameters from `extra_types`

    def _read_comb_rule(self) -> int:
        for sec, lines in _sections(self.path / 'forcefield.itp'):
            if sec == 'defaults' and lines:
                return int(lines[0].split()[1])
        raise ValueError('no [ defaults ] in %s' % (self.path / 'forcefield.itp'))

    def _read_atomtypes(self) -> Dict[str, AtomType]:
        types = {}
        # ffnonbonded.itp, plus ffnanonbonded.itp (nucleic acids) where present
        files = sorted(self.path.glob('ff*nonbonded.itp'))
        for path in files:
            for sec, lines in _sections(path):
                if sec != 'atomtypes':
                    continue
                for line in lines:
                    tok = line.split()
                    # name [bond_type] at.num mass charge ptype v w
                    has_bond_type = not tok[1].lstrip('-').isdigit()
                    i = 2 if has_bond_type else 1
                    bond_type = tok[1] if has_bond_type else tok[0]
                    z = int(tok[i])
                    mass, charge, ptype = float(tok[i + 1]), float(tok[i + 2]), tok[i + 3]
                    v, w = float(tok[i + 4]), float(tok[i + 5])
                    if self.comb_rule == 1:
                        # C6 = 4 eps sigma^6, C12 = 4 eps sigma^12
                        sigma, eps = ((w / v) ** (1. / 6.), v * v / (4. * w)) if v > 0 else (0., 0.)
                    else:
                        sigma, eps = v, w
                    types[tok[0]] = AtomType(tok[0], bond_type, z, mass, charge, ptype, sigma, eps)
        if not types:
            raise ValueError('no [ atomtypes ] found in %s' % self.path)
        return types

    def _read_descriptions(self):
        '''Attach the free-text comment of `atomtypes.atp`, where present.'''

        path = self.path / 'atomtypes.atp'
        if not path.exists():
            return
        for raw in path.read_text(errors='replace').splitlines():
            if ';' not in raw:
                continue
            head, _, desc = raw.partition(';')
            tok = head.split()
            if tok and tok[0] in self.atomtypes and desc.strip():
                old = self.atomtypes[tok[0]]
                self.atomtypes[tok[0]] = AtomType(
                    old.name, old.bond_type, old.z, old.mass, old.charge,
                    old.ptype, old.sigma, old.epsilon, desc.strip())

    @property
    def residues(self) -> List[Residue]:
        if self._residues is None:
            self._residues = self._read_residues()
        return self._residues

    def _read_residues(self) -> List[Residue]:
        directives = ('bondedtypes', 'atoms', 'bonds', 'angles', 'dihedrals',
                      'impropers', 'exclusions', 'cmap')
        residues = []
        for path in sorted(self.path.glob('*.rtp')):
            res = None
            for sec, lines in _sections(path):
                if sec not in directives:
                    # any other bracketed name opens a new residue
                    res = Residue(sec, source=path.name)
                    residues.append(res)
                elif sec == 'atoms' and res is not None:
                    for line in lines:
                        tok = line.split()
                        res.atoms[tok[0]] = tok[1]
                elif sec == 'bonds' and res is not None:
                    for line in lines:
                        tok = line.split()
                        res.bonds.append((tok[0], tok[1]))
        known = [r for r in residues if r.atoms and all(t in self.atomtypes for t in r.atoms.values())]
        if len(known) < len([r for r in residues if r.atoms]):
            log.debug('%s: %d residues use atom types missing from the nonbonded files; skipped',
                      self.name, len([r for r in residues if r.atoms]) - len(known))
        return known

    def __repr__(self):
        return 'ForceField(%r, %d types)' % (self.name, len(self.atomtypes))

    @property
    def elements(self) -> set:
        '''Elements for which the force field has at least one atom type.'''
        return {t.element for t in self.atomtypes.values()}

    def add_missing_element(self, symbol: str) -> AtomType:
        '''
        Register an atom type for an element the force field does not cover.

        The type is named after the element and has its standard atomic mass.
        Sigma and epsilon are those of `extra_types.EXTRA_TYPES` if the element
        is listed there (`self.extra`); otherwise they are NaN, to be filled in
        by the user (`self.placeholders`, see `perceive_types.MARKER`).
        '''

        try:
            from MDAnalysis.guesser.tables import masses
        except ImportError:
            from MDAnalysis.topology.tables import masses
        from .extra_types import EXTRA_TYPES

        z = {v: k for k, v in SYMBOLS.items()}[symbol]
        mass = float(masses[symbol])
        if symbol in EXTRA_TYPES:
            par = EXTRA_TYPES[symbol]
            at = AtomType(symbol, symbol, z, par.mass, 0.0, 'A', par.sigma, par.epsilon,
                          'not an %s value, from %s' % (self.name, par.source))
            self.extra.add(symbol)
        else:
            at = AtomType(symbol, symbol, z, mass, 0.0, 'A', float('nan'), float('nan'),
                          'no %s type for %s: sigma and epsilon to be filled in' % (self.name, symbol))
            self.placeholders.add(symbol)
        self.atomtypes[symbol] = at
        return at

    def lj_key(self, typename: str) -> Tuple[int, float, float]:
        '''(z, sigma, epsilon) rounded for comparing Lennard-Jones classes.'''

        t = self.atomtypes[typename]
        return t.z, round(t.sigma, 6), round(t.epsilon, 5)
