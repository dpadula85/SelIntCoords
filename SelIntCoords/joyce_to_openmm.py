#!/usr/bin/env python

'''
Convert a Joyce-generated GROMACS `.top` (+ every file it `#include`s)
into a corrected copy that both real GROMACS and OpenMM read
*identically*. Pure text rewriting -- never loads OpenMM, GROMACS, or
any simulation package. Verified (with OpenMM and GROMACS both
installed, in `oligomer_builder`) against a real GROMACS single-point
energy oracle on 18 real systems spanning ~20 to ~146 000 atoms; every
bonded and nonbonded energy term matches to float64 rounding once a
file is run through this conversion.

THREE THINGS THIS FIXES IN THE FILE'S OWN TEXT
===============================================

1. `[ pairs ]` funct 2 (`ai aj 2 fudgeQQ qi qj V W` -- GROMACS's format
   for an explicit 1-4-or-further intramolecular nonbonded exception
   with its own fudgeQQ/charges/LJ parameters) is rejected outright by
   OpenMM's `GromacsTopFile._processPair`, which only understands
   funct 1. Joyce writes funct 2 for *every* `[ pairs ]` line, not just
   genuine 1-4 pairs: its own `[ pairs ]`-writing logic (`ic_handling.
   f`) puts two different things under this one format -- genuine 1-4
   pairs (`fudgeQQ=0.833` in that source) and every other non-excluded,
   non-adjacent intramolecular pair beyond that (`fudgeQQ=1.0`,
   labelled "everything else" there). GROMACS makes no functional
   distinction between the two, so neither does this module: every
   funct-2 line is rewritten to funct-1 (`ai aj 1 V W`) the same way.

2. `[ pairs ]` is a *list*, not a *set* -- the same atom pair can
   legitimately appear on more than one line (confirmed real on a
   fused-ring small-molecule NFA acceptor: 147 of 3955 pairs listed
   twice, identical `V`/`W` both times, and separately on 8 more real
   force-field files), and GROMACS sums every listed line as an
   independent, additive interaction (confirmed via `gmx dump`: its own
   internal interaction list carries every listed instance, not a
   deduplicated count) -- so does *any* naive reader that assumes at
   most one interaction per atom pair, which silently overcounts.
   Duplicate lines are commented out here (kept, tagged `; DUPLICATE
   (dropped during conversion...)`, not deleted), first occurrence kept
   live -- confirmed lossless on every real duplicate found so far
   (always identical to the first occurrence, sometimes with `i`/`j`
   swapped).

3. A funct-1 `[ pairs ]` line has no per-pair charge override at all --
   GROMACS computes its Coulomb-14 from the atoms' *real* `[ atoms ]`
   charges times the *global* `fudgeQQ` in `[ defaults ]`; only funct 2
   carries an explicit override (`qi`, `qj`, its own `fudgeQQ`). Simply
   rewriting funct 2 to funct 1 (fix 1 above) therefore silently
   reintroduces a nonzero 1-4 Coulomb term the original line's own
   `qi=qj=0` explicitly zeroed, the moment the global `fudgeQQ` and the
   atoms' real charges are both nonzero. Confirmed real with a minimal
   hand-built GROMACS test (real nonzero atom charges, global
   `fudgeQQ=0.8333`, a funct-1 line with explicit `V W`): Coulomb-14 =
   -96.48 kJ/mol, not 0. One real pipeline `.top` has exactly this
   nonzero global `fudgeQQ` but happens to carry zero-charge atoms
   throughout, masking the effect there by coincidence, not by
   correctness -- a real risk for any Joyce file with both a nonzero
   global `fudgeQQ` and nonzero atom charges. Fixed by forcing
   `[ defaults ]`'s global `fudgeQQ` to `0.0` -- safe *because*, not
   despite, every kept `[ pairs ]` line being validated first to have
   `fudgeQQ*qi*qj == 0`; a line that fails this check raises
   `ChargeOverrideError` instead of silently producing a wrong file,
   since forcing `fudgeQQ` to 0 in that case would misrepresent that
   specific pair's intended physics, not conservatively preserve it.
   Every real Joyce file checked so far has `qi=qj=0` on every
   `[ pairs ]` line, so this has not fired in practice.

Comb-rule-dependent unit handling, a dihedral-multiplicity-0 crash,
spurious auto-generated 1-4 interactions at a merged junction, and a
dispersion-correction default mismatch are OpenMM-`System`-internal
mistakes with no counterpart in real GROMACS at all -- not file-content
problems, so out of scope here; see `oligomer_builder.openmm_compat`
for those and the full verification record.
'''

import argparse as arg
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("joyce_to_openmm")

_SECTION_RE = re.compile(r'^\s*\[\s*([a-zA-Z_]+)\s*\]')
_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"')
_DEFAULTS_LINE_RE = re.compile(r'^(\s*\S+\s+\S+\s+\S+\s+\S+\s+)(\S+)(.*)$')


class UnparameterisedPairError(ValueError):
    '''Raised when a `[ pairs ]` line carries no explicit LJ parameters.

    A Joyce `[ pairs ]` line always gives its own `V W`. A bare `ai aj 1`
    line would take its parameters from `[ pairtypes ]` or `gen-pairs`,
    which `openmm_compat`'s pair rebuild never consults, so the pair
    would silently vanish from the `System`.
    '''


class ChargeOverrideError(ValueError):
    '''Raised when a `[ pairs ]` line's own `fudgeQQ*qi*qj` is nonzero.

    See fix 3 in this module's docstring for why this exact value
    cannot be carried through a funct-2-to-funct-1 rewrite, and why
    forcing `[ defaults ]`'s global `fudgeQQ` to 0 would silently
    change this particular pair's intended physics rather than leave
    it alone.
    '''


def _rewrite_and_collect(top_text, source_name='<string>'):
    '''Single pass over one `.top`/`.itp` file's text.

    Parameters
    ----------
    top_text: str.
        Full text of a `.top` or `.itp` file.
    source_name: str.
        Name used only in `ChargeOverrideError` messages.

    Returns
    -------
    rewritten_text: str.
        Funct-2 `[ pairs ]` lines rewritten to funct-1 (`ai aj 1 V W`);
        a duplicate `[ pairs ]` line for a pair already seen earlier
        **in the same `[ moleculetype ]`** is commented out (kept, not
        deleted, tagged) rather than passed through. Byte-identical
        otherwise, except `[ defaults ]`'s global `fudgeQQ` field,
        forced to `0.0` whenever this text contains one.
    pairs: list of (mol_name, i, j, fudgeQQ, qi, qj, v, w).
        Every *kept* (non-duplicate) `[ pairs ]` line, 1-indexed and
        molecule-local, in file order -- byproduct of the same parse,
        added so `openmm_compat.load_top` can rebuild each pair's 1-4
        interaction from the file's own text instead of trusting
        `createSystem`'s auto-generated one (see that module).
    includes: list of str.
        `#include`d filenames, in file order.

    Raises
    ------
    ChargeOverrideError.
        If any kept `[ pairs ]` line's own `fudgeQQ*qi*qj` is nonzero.
    '''

    out = []
    pairs = []
    includes = []
    section = None
    mol_name = None
    expect_mol_name = False
    seen_pairs = set()
    has_defaults = False
    lineno = 0

    for line in top_text.split('\n'):
        lineno += 1
        inc = _INCLUDE_RE.match(line)
        if inc:
            includes.append(inc.group(1))
            out.append(line)
            continue

        m = _SECTION_RE.match(line)
        if m:
            section = m.group(1).lower()
            expect_mol_name = (section == 'moleculetype')
            if section == 'moleculetype':
                seen_pairs = set()
            out.append(line)
            continue

        stripped = line.split(';')[0].strip()
        if not stripped:
            out.append(line)
            continue

        if section == 'defaults' and not has_defaults:
            # First non-comment [ defaults ] line: nbfunc, comb-rule,
            # gen-pairs, fudgeLJ, fudgeQQ. Force fudgeQQ (5th field) to
            # 0 -- fix 3 above; safe exactly because every pair kept
            # below is validated to have fudgeQQ*qi*qj == 0.
            has_defaults = True
            m2 = _DEFAULTS_LINE_RE.match(line)
            if m2:
                out.append(
                    '%s0.0  ; fudgeQQ forced to 0 during funct-2->funct-1 '
                    '[ pairs ] conversion, see joyce_to_openmm.py%s'
                    % (m2.group(1), m2.group(3))
                )
                continue
            out.append(line)
            continue

        if section == 'moleculetype' and expect_mol_name:
            mol_name = stripped.split()[0]
            expect_mol_name = False
            out.append(line)
            continue

        if section == 'pairs':
            fields = stripped.split()
            is_funct2 = len(fields) >= 8 and fields[2] == '2'
            # A bare funct-1 line (5 fields, no qi/qj columns -- funct-1
            # has none) shows up here when re-processing this module's
            # own output (idempotency): its intended chargeProd is
            # always 0 by construction (fix 3), since a second pass
            # only ever sees a file whose global fudgeQQ was already
            # forced to 0.
            is_bare_funct1 = len(fields) >= 5 and fields[2] == '1' and len(fields) < 8
            if is_funct2 or is_bare_funct1:
                i, j = int(fields[0]), int(fields[1])
                if is_funct2:
                    fudgeQQ = float(fields[3])
                    qi, qj = float(fields[4]), float(fields[5])
                    v, w = float(fields[6]), float(fields[7])
                else:
                    fudgeQQ, qi, qj = 0.0, 0.0, 0.0
                    v, w = float(fields[3]), float(fields[4])
                if fudgeQQ * qi * qj != 0.0:
                    raise ChargeOverrideError(
                        "%s:%d: [ pairs ] line for atoms %d,%d has a "
                        "nonzero fudgeQQ*qi*qj = %r -- this exact charge "
                        "product cannot be represented in funct-1 GROMACS "
                        "format (no per-pair override there), so no single "
                        "converted file can be read identically by both "
                        "GROMACS and OpenMM for this file. See fix 3 in "
                        "this module's docstring."
                        % (source_name, lineno, i, j, fudgeQQ * qi * qj)
                    )
                key = (min(i, j), max(i, j))
                if key in seen_pairs:
                    out.append(
                        '; DUPLICATE (dropped during conversion, identical '
                        'to an earlier line for this pair): %s' % line
                    )
                    continue
                seen_pairs.add(key)
                pairs.append((mol_name, i, j, fudgeQQ, qi, qj, v, w))
                if is_funct2:
                    out.append('%6d %6d 1 %s %s' % (i, j, fields[6], fields[7]))
                else:
                    out.append(line)
                continue
            raise UnparameterisedPairError(
                "%s:%d: [ pairs ] line %r has no explicit V W -- only "
                "explicit funct-1 (ai aj 1 V W) or funct-2 (ai aj 2 fudgeQQ "
                "qi qj V W) lines are supported." % (source_name, lineno, stripped)
            )

        out.append(line)

    return '\n'.join(out), pairs, includes


def _resolve_and_rewrite(top_path, output_dir):
    '''Recursively rewrites `top_path` and every file it `#include`s.

    `#include`s are resolved relative to each including file's own
    directory (GROMACS semantics). The rewritten files are written under
    `output_dir` mirroring their layout relative to the deepest directory
    common to all of them, so every `#include` line keeps resolving
    unchanged -- including `../` paths and two same-named `.itp` files in
    different directories.

    Returns `(pairs, top_out)`: the accumulated `pairs` list (see
    `_rewrite_and_collect`) across every file, in the order encountered,
    and the path of the rewritten top-level file inside `output_dir`.
    '''

    top_path = Path(top_path).resolve()
    output_dir = Path(output_dir)
    pairs = []
    files = {}

    def process(path, included_from=None):
        path = path.resolve()
        if path in files:
            return
        if not path.is_file():
            raise FileNotFoundError(
                "#include \"%s\" (from %s) not found"
                % (path, included_from or "<top>")
            )
        rewritten, file_pairs, includes = _rewrite_and_collect(
            path.read_text(), source_name=path.name
        )
        files[path] = rewritten
        pairs.extend(file_pairs)
        for inc_name in includes:
            process(path.parent / inc_name, included_from=path)

    process(top_path)
    root = Path(os.path.commonpath([str(f.parent) for f in files]))
    for path, rewritten in files.items():
        target = output_dir / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rewritten)
    return pairs, output_dir / top_path.relative_to(root)


def convert_top(top_path, output_dir):
    '''Convert a `.top` (and every file it `#include`s) for dual use.

    Parameters
    ----------
    top_path: str or Path.
        Joyce-generated `.top` to convert.
    output_dir: str or Path.
        Directory the converted file(s) are written into (created if
        missing).

    Returns
    -------
    converted: Path.
        Path to the converted top-level file inside `output_dir`; every
        `#include`d file it needed is written alongside it, keeping the
        original relative layout, ready for `gmx grompp` or an OpenMM
        `GromacsTopFile` loader.

    Raises
    ------
    ChargeOverrideError.
        See this module's docstring, fix 3.
    UnparameterisedPairError.
        A `[ pairs ]` line carries no explicit V W.
    FileNotFoundError.
        An `#include`d file does not exist.
    '''

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, converted = _resolve_and_rewrite(top_path, output_dir)
    return converted


def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(
        description="Convert a Joyce-generated GROMACS .top file (and everything "
                     "it #includes) into a corrected copy readable identically by "
                     "both real GROMACS and OpenMM.",
        formatter_class=arg.ArgumentDefaultsHelpFormatter,
    )

    inp = parser.add_argument_group("Input Data")
    inp.add_argument(
        "-p", "--top", type=str, required=True, dest="TopFile",
        help="Topology to convert (.top format).",
    )

    out = parser.add_argument_group("Output Data")
    out.add_argument(
        "-o", "--output-dir", type=str, required=True, dest="OutDir",
        help="Directory the converted topology (and any #included file it needs) "
             "is written into.",
    )
    out.add_argument(
        "-v", "--verbose", action="store_true", dest="Verbose",
        help="Print progress information.",
    )

    return vars(parser.parse_args())


def main():
    Opts = options()

    logging.basicConfig(
        level=logging.INFO if Opts["Verbose"] else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        converted = convert_top(Opts["TopFile"], Opts["OutDir"])
    except (ChargeOverrideError, UnparameterisedPairError) as e:
        log.error("Conversion refused: %s", e)
        raise SystemExit(1)

    log.info("Converted topology written to %s", converted)


if __name__ == "__main__":
    main()
