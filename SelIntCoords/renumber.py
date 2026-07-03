#!/usr/bin/env python

'''
Renumber the atoms of a GROMACS topology (.top) file according to an
external reference numbering (e.g. an experimental crystallographic
numbering), using a two-column map such as the one produced by
`map_atoms` (see :mod:`SelIntCoords.map_atoms`).

Bonds, angles, dihedrals, impropers, pairs, exclusions and cmaps all
hold direct references to the same :class:`SelIntCoords.blocks.Atom`
objects that live in ``molecule.atoms``, and are written out from
their live ``.number`` attribute (see :mod:`SelIntCoords.top`).
Renumbering the atoms in place is therefore enough to keep the whole
topology internally consistent - there is no need to touch the bonded
sections directly.
'''

import argparse as arg
import logging
from pathlib import Path

from .top import TOP

log = logging.getLogger("renumber")


def read_map(filename):
    '''Read a two-column (reference, target) atom map into a dictionary.

    Parameters
    ----------
    filename: str.
        Map file, one "reference target" pair per line (as written e.g.
        by :meth:`SelIntCoords.map_atoms.AtomMapping.write`). Lines
        starting with "#" and blank lines are skipped.

    Returns
    -------
    renummap: dict.
        {target_number: reference_number}, i.e. the renumbering to
        apply to a topology using the *target* numbering.
    '''

    renummap = {}
    with open(filename) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    "%s, line %d: expected two integers, got: %r" % (filename, lineno, line)
                )

            ref, tgt = (int(x) for x in fields)
            if tgt in renummap:
                raise ValueError(
                    "%s, line %d: target atom %d appears more than once." % (filename, lineno, tgt)
                )

            renummap[tgt] = ref

    return renummap


def renumber_molecule(molecule, renummap, molname=None, strict=True):
    '''Renumber a Molecule's atoms in place according to a map.

    Parameters
    ----------
    molecule: SelIntCoords.blocks.Molecule.
        Molecule whose atoms are to be renumbered.
    renummap: dict.
        {old_number: new_number} mapping, e.g. as returned by :func:`read_map`.
    molname: str.
        Molecule name, used only for clearer error/log messages.
    strict: bool.
        If True (default), require `renummap` to cover every atom, and
        the resulting numbering to be a bijection (no two atoms mapped
        to the same new number). This is the safe default: applying a
        partial map on top of untouched atom numbers can silently
        create duplicate atom numbers and corrupt the topology. Set to
        False to allow atoms missing from the map to keep their
        original number, at your own risk.

    Returns
    -------
    applied: list of (int, int) tuples.
        (old_number, new_number) pairs actually applied, for logging or
        bookkeeping purposes.
    '''

    label = " for molecule '%s'" % molname if molname else ""
    atom_numbers = [atom.number for atom in molecule.atoms]

    missing = [n for n in atom_numbers if n not in renummap]
    if missing:
        msg = "%d atom(s)%s are not present in the map (e.g. %s)." % (
            len(missing), label, missing[:5]
        )
        if strict:
            raise ValueError(msg + " Use --allow-partial to keep their original numbering.")
        else:
            log.warning(msg + " Keeping their original numbering.")

    new_numbers = [renummap.get(n, n) for n in atom_numbers]
    if len(set(new_numbers)) != len(new_numbers):
        dupes = sorted({n for n in new_numbers if new_numbers.count(n) > 1})
        raise ValueError(
            "Renumbering%s is not a bijection: new number(s) %s would be assigned "
            "to more than one atom. Check your map file." % (label, dupes)
        )

    applied = []
    for atom, new_number in zip(molecule.atoms, new_numbers):
        applied.append((atom.number, new_number))
        atom.number = new_number

    molecule.atoms = sorted(molecule.atoms, key=lambda a: a.number)
    molecule._anumb_to_atom = {}   # invalidate any cached number->atom lookup

    return applied


def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(
        description="Renumber the atoms of a GROMACS .top file using an external "
                     "(e.g. crystallographic) atom map.",
        formatter_class=arg.ArgumentDefaultsHelpFormatter,
    )

    inp = parser.add_argument_group("Input Data")
    inp.add_argument(
        "-p", "--top", type=str, required=True, dest="TopFile",
        help="Topology to renumber (.top format).",
    )
    inp.add_argument(
        "-m", "--map", type=str, required=True, dest="MapFile",
        help="Two-column (reference target) atom map, e.g. as produced by map_atoms.",
    )
    inp.add_argument(
        "-n", "--molecule", type=int, default=0, dest="MolIndex",
        help="Index of the molecule (within top.molecules) to renumber.",
    )
    inp.add_argument(
        "--allow-partial", action="store_true", dest="AllowPartial",
        help="Allow atoms missing from the map to keep their original number "
             "(unsafe: can silently create duplicate atom numbers).",
    )

    out = parser.add_argument_group("Output Data")
    out.add_argument(
        "-o", "--output", type=str, default=None, dest="OutFile",
        help="Renumbered topology output file. Defaults to '<TopFile (no ext)>.renumbered.top'.",
    )
    out.add_argument(
        "-v", "--verbose", action="store_true", dest="Verbose",
        help="Print the full old -> new atom number mapping applied.",
    )

    return vars(parser.parse_args())


def main():
    Opts = options()

    logging.basicConfig(
        level=logging.INFO if Opts["Verbose"] else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    top = TOP(Opts["TopFile"])
    renummap = read_map(Opts["MapFile"])

    molecule = top.molecules[Opts["MolIndex"]]
    molname = getattr(molecule, "name", None)

    applied = renumber_molecule(
        molecule, renummap, molname=molname, strict=not Opts["AllowPartial"]
    )

    for old, new in applied:
        log.info("%5d -> %5d", old, new)

    outfile = Opts["OutFile"] or str(Path(Opts["TopFile"]).with_suffix("")) + ".renumbered.top"
    top.write(outfile)
    log.info("Renumbered topology written to %s", outfile)


if __name__ == "__main__":
    main()
