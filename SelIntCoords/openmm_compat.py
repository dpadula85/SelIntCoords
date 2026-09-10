#!/usr/bin/env python

'''
Build a *correct* OpenMM `System` from a Joyce-generated GROMACS `.top`,
one that computes the same energy real GROMACS does -- not just one
OpenMM can parse without crashing.

`joyce_to_openmm.py` (this package) already fixes three things that are
wrong in the `.top` file's own *text*: funct-2 `[ pairs ]` (rejected
outright by OpenMM's parser), duplicate `[ pairs ]` lines (double-counted
by a naive reader), and the funct-1 charge-override gap (`[ defaults ]`'s
`fudgeQQ` forced to 0). That conversion is necessary but **not
sufficient**: once the file parses, `GromacsTopFile.createSystem()` still
gets the resulting `System` object wrong in four more ways that have
nothing to do with the file's text at all -- they are mistakes in how
OpenMM's own Python API turns a *correctly formatted* file into a
`System`. Converting the file makes both programs able to *read* it; it
does not make them compute the *same* energy. This module is what
closes that second gap.

Verified against a real GROMACS single-point-energy oracle (`gmx mdrun
-rerun`, `gmx_d`, double precision) on 18 real systems spanning ~20 to
~146 000 atoms, in the `oligomer_builder` pipeline package, where this
exact fix (`oligomer_builder.openmm_compat`) originates -- see that
module's own docstring and `oligomer_builder/HANDOFF.md`'s "OpenMM as a
future GROMACS alternative" for the full diagnosis and verification
record. This is a deliberate, acknowledged duplication of that module,
not an oversight: `oligomer_builder` needs its own copy regardless,
because one of the four bugs below (auto-generated 1-4 interactions at a
merged junction) only exists once two Joyce fragments have been merged
into one system -- something a single Joyce monomer, the only kind of
`.top` this package ever handles, never does. The fix applies
unconditionally and correctly either way; it is only the *bug* that
needs a merged junction to actually fire.

FOUR MORE THINGS THIS FIXES, NONE OF THEM FILE-TEXT PROBLEMS
==============================================================

1. An explicit `[ pairs ]` line's `V W` is real sigma/epsilon under
   comb-rule 2 or 3 (this pipeline's usual choice) but literal C6/C12
   under comb-rule 1 -- and depending on which internal `Force` object
   ends up holding the value `createSystem` produced, trusting its own
   bookkeeping can get this wrong silently (an LJ-14 of ~0 instead of
   the real value, no error). Fixed by reading `V`/`W` from the file's
   own text (via `joyce_to_openmm._rewrite_and_collect`'s `pairs`
   return value) and converting by comb-rule in this module's own code,
   never trusting `createSystem`'s internal representation.

2. A funct-1 `[ dihedrals ]` line with multiplicity 0 -- a real,
   valid GROMACS construct: with `mult=0` the periodic-torsion form
   `k*(1+cos(mult*phi-phi0))` collapses to the phi-independent constant
   `k*(1+cos(-phi0))`, a fixed energy offset with zero gradient, not an
   error. `createSystem` builds a `PeriodicTorsionForce` entry with
   `periodicity=0` for these without complaint; OpenMM only rejects it
   later, at `Context` construction (`"periodicity must be positive"`),
   far from the actual cause. Fixed by dropping those terms from
   `PeriodicTorsionForce` and separately returning their constant
   contribution (`dropped_dihedral_offset`) for callers that need the
   absolute potential energy to match GROMACS; irrelevant for forces or
   dynamics.

3. `createSystem` auto-generates a 1-4 exception for *every* topological
   1-4 pair it finds (dihedral list or bond graph), using generic
   atomtype combining plus `[ defaults ]`'s global `fudgeLJ`/`fudgeQQ`,
   before the file's own `[ pairs ]` entries overwrite the ones they
   cover. Harmless on a single Joyce-generated monomer -- the only kind
   of `.top` this package ever produces -- since Joyce enumerates every
   genuine 1-4 pair explicitly, so `createSystem`'s auto-generated value
   and the file's own `[ pairs ]` line agree. **Not** harmless the
   moment two such fragments are merged at a junction Joyce never saw
   (`oligomer_builder`/`top_merge.py`, downstream of this package): the
   junction's own cross-fragment 1-4 pairs are topologically real but
   have no `[ pairs ]` line, so GROMACS (which has no generic-combining
   fallback: zero interaction unless a `[ pairs ]` line reintroduces
   one) and OpenMM's auto-generated value disagree. The fix below
   applies unconditionally regardless of whether a junction exists --
   it zeroes every auto-generated 1-4 exception and rebuilds all of
   them strictly from the file's own `[ pairs ]` lines, so it is
   already correct here even though the bug it guards against cannot
   actually occur on a single monomer.

4. `createSystem`'s `useDispersionCorrection` keyword defaults to
   `True`, silently adding an analytic long-range LJ tail correction
   GROMACS's own default (`DispCorr = no`) does not include. Invisible
   on an isolated molecule in a large box (the correction scales with
   density, so it is ~0 there) -- exactly the case for every system
   this package builds, since it never assembles a dense periodic
   system itself -- but real and large on an actual condensed system
   downstream (`oligomer_builder`/`HANDOFF.md`: +183 kJ/mol out of
   -22 541 on a real 57 960-atom production system). Fixed by simply
   not asking for the correction by default
   (`use_dispersion_correction=False`, matching GROMACS's own default);
   pass `True` for a real MD run that turns `DispCorr` on.

**The fix for issues 1 and 3, in one place
(`_rebuild_onefour_interactions`):** every 1-4-shaped interaction
`createSystem` itself built (`NonbondedForce` exceptions and any
comb-rule-1/3 `CustomBondForce`) is zeroed unconditionally, then
rebuilt from scratch in one dedicated `CustomBondForce`
(`'OneFourInteractions'`) with exactly one bond added per `[ pairs ]`
*line* actually in the file -- never keyed by atom pair, so a
duplicate line (already resolved by `joyce_to_openmm`'s own dedup, but
this function does not assume that happened) can never silently
collapse -- each one's C6/C12 converted from that line's own `V`/`W`
by comb-rule and its charge product read from that line's own
`fudgeQQ*qi*qj`.

Needs `openmm` (`pip install openmm`) -- not in `requirements.txt`,
matching how `psi4` (needed only by `make_top`/`sel_intcoords`) is
already handled: documented, not declared, since this is the only
module in the package that uses it.
'''

import math
import tempfile
from pathlib import Path

import scipy.constants as const

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from SelIntCoords.joyce_to_openmm import _resolve_and_rewrite, ChargeOverrideError

# 1/(4 pi eps0) in kJ/mol . nm / e^2 -- GROMACS's own ONE_4PI_EPS0 (derived
# the same way rather than copied from either code base, per house
# convention; matches to 6 significant figures: 138.935458 kJ/mol.nm/e^2).
_ONE_4PI_EPS0 = (1.0 / (4 * math.pi * const.epsilon_0) * const.e**2
                 * const.N_A / const.kilo / const.nano)


def _strip_zero_multiplicity_torsions(system):
    '''Removes every periodicity-0 torsion from `system`'s
    `PeriodicTorsionForce` (OpenMM cannot build a `Context` with one)
    and returns their total `k*(1 + cos(-phi0))` energy in kJ/mol -- the
    position-independent constant GROMACS adds for them.
    '''
    total_offset = 0.0
    for i in reversed(range(system.getNumForces())):
        force = system.getForce(i)
        if not isinstance(force, mm.PeriodicTorsionForce):
            continue
        kept = mm.PeriodicTorsionForce()
        for k in range(force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k_const = force.getTorsionParameters(k)
            if periodicity > 0:
                kept.addTorsion(p1, p2, p3, p4, periodicity, phase, k_const)
            else:
                phi0 = phase.value_in_unit(unit.radian)
                kk = k_const.value_in_unit(unit.kilojoule_per_mole)
                total_offset += kk * (1 + math.cos(-phi0))
        system.removeForce(i)
        system.addForce(kept)
    return total_offset


def _add_unshifted_coulomb_correction(system, nb, nonbonded_cutoff):
    '''Cancels the reaction-field shift `NonbondedForce` always applies
    for `CutoffPeriodic`/`CutoffNonPeriodic` (a per-pair `-ke*qi*qj/rc`
    constant baked into its formula even at `reactionFieldDielectric=1`,
    where it reduces to plain "potential shifted to zero at the
    cutoff") -- turning its main Coulomb term into a plain truncated
    `ke*qi*qj/r`, GROMACS's `coulomb-modifier = None` convention. Adds
    one small `CustomNonbondedForce` (`+ke*qi*qj/rc` for every pair
    `NonbondedForce` itself still computes, zero beyond `rc` by the
    same hard cutoff, same exclusions) rather than trying to
    reconfigure `NonbondedForce` itself, which cannot express "no
    shift" directly.

    Has **no effect on forces or dynamics** -- adding/removing a
    per-pair constant changes only the reported potential energy, never
    its gradient. Only meaningful under `CutoffPeriodic`; PME carries no
    such shift, so `load_top` only calls this when that is the
    requested `nonbonded_method`.
    '''
    nb.setReactionFieldDielectric(1.0)
    rc = nonbonded_cutoff.value_in_unit(unit.nanometer)
    correction = mm.CustomNonbondedForce(
        '%s*charge1*charge2/%s' % (_ONE_4PI_EPS0, rc)
    )
    correction.addPerParticleParameter('charge')
    correction.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    correction.setCutoffDistance(nonbonded_cutoff)
    correction.setName('UnshiftedCoulombCorrection')
    for i in range(nb.getNumParticles()):
        q, _, _ = nb.getParticleParameters(i)
        correction.addParticle([q])
    for k in range(nb.getNumExceptions()):
        p1, p2, *_ = nb.getExceptionParameters(k)
        correction.addExclusion(p1, p2)
    system.addForce(correction)


def _rebuild_onefour_interactions(system, top, ordered_pairs):
    '''Zeroes every 1-4-shaped interaction `createSystem` itself built
    -- whether from the file's own `[ pairs ]` lines (comb-rule-
    dependent misread) or auto-generated from the dihedral list/bond
    graph for a topological 1-4 pair with no `[ pairs ]` line at all --
    and rebuilds all of them from scratch in one dedicated
    `CustomBondForce` (`'OneFourInteractions'`), the sole source of
    truth for which pairs get a 1-4 interaction afterwards: exactly the
    file's own `[ pairs ]` lines, one bond per line. See this module's
    docstring, issues 1 and 3, for the full reasoning and measurements
    (verified in `oligomer_builder`, where this fix originates).
    '''
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    old_cbf = next((f for f in system.getForces()
                     if isinstance(f, mm.CustomBondForce) and f.getName() == 'LennardJonesExceptions'),
                    None)

    for k in range(nb.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb.getExceptionParameters(k)
        if (chargeProd.value_in_unit(unit.elementary_charge**2) != 0.0
                or epsilon.value_in_unit(unit.kilojoule_per_mole) != 0.0):
            nb.setExceptionParameters(k, p1, p2, 0.0, 1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    if old_cbf is not None:
        for k in range(old_cbf.getNumBonds()):
            p1, p2, params = old_cbf.getBondParameters(k)
            if any(p != 0.0 for p in params):
                old_cbf.setBondParameters(k, p1, p2, [0.0, 0.0])

    comb_rule = top._defaults[1]
    onefour = mm.CustomBondForce('-C6/r^6 + C12/r^12 + %s*chargeProd/r' % _ONE_4PI_EPS0)
    onefour.addPerBondParameter('C6')
    onefour.addPerBondParameter('C12')
    onefour.addPerBondParameter('chargeProd')
    onefour.setName('OneFourInteractions')
    system.addForce(onefour)

    base = 0
    for mol_name, n_copies in top._molecules:
        n_atoms = len(top._moleculeTypes[mol_name].atoms)
        for _ in range(n_copies):
            for name, i, j, fudgeQQ, qi, qj, v, w in ordered_pairs:
                if name != mol_name:
                    continue
                p1, p2 = base + i - 1, base + j - 1
                chargeProd = fudgeQQ * qi * qj
                if comb_rule == '1':
                    c6, c12 = v, w  # already C6, C12
                else:
                    c6, c12 = 4 * w * v**6, 4 * w * v**12  # v=sigma, w=epsilon
                onefour.addBond(p1, p2, [c6, c12, chargeProd])
            base += n_atoms


def load_top(top_path, gro_path, nonbonded_cutoff=1.2 * unit.nanometer,
             use_dispersion_correction=False, nonbonded_method=app.CutoffPeriodic,
             constraints=None):
    '''Parses `top_path`/`gro_path` into an OpenMM `(topology, system)`
    pair that computes the same energy real GROMACS does -- not just
    one that OpenMM can build without error.

    Every `[ dihedrals ]` funct-1 term with multiplicity 0 is dropped
    from the returned `System`; its total energy -- a constant,
    independent of positions -- is returned as `dropped_dihedral_offset`
    (kJ/mol) for callers that need the absolute potential energy to
    match GROMACS exactly. Zero for forces/dynamics regardless.

    `use_dispersion_correction=False` by default, matching GROMACS's
    own default (`DispCorr = no`); pass `True` for a real MD run whose
    `.mdp`/protocol turns dispersion correction on.

    `nonbonded_method=app.CutoffPeriodic` by default (reaction-field-
    shifted cutoff); pass `app.PME` to match a real GROMACS `coulombtype
    = PME` run instead. `_add_unshifted_coulomb_correction` (cancels
    `CutoffPeriodic`'s own implicit shift) is only applied when
    `nonbonded_method is app.CutoffPeriodic` -- meaningless under PME.

    `constraints=None` by default; pass `app.HBonds` to match a real
    run's GROMACS `constraints = h-bonds` (OpenMM's own CCMA solver
    enforces it, not GROMACS's LINCS -- same physical constraint,
    different algorithm).

    Returns `(topology, system, dropped_dihedral_offset)`.
    '''
    gro = app.GromacsGroFile(str(gro_path))
    top_path = Path(top_path)

    with tempfile.TemporaryDirectory() as workdir:
        workdir = Path(workdir)
        ordered_pairs = _resolve_and_rewrite(top_path, workdir)
        top = app.GromacsTopFile(str(workdir / top_path.name),
                                  periodicBoxVectors=gro.getPeriodicBoxVectors())
        system = top.createSystem(nonbondedMethod=nonbonded_method,
                                   nonbondedCutoff=nonbonded_cutoff,
                                   useDispersionCorrection=use_dispersion_correction,
                                   constraints=constraints)

    _rebuild_onefour_interactions(system, top, ordered_pairs)
    if nonbonded_method is app.CutoffPeriodic:
        nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
        _add_unshifted_coulomb_correction(system, nb, nonbonded_cutoff)
    dropped_offset = _strip_zero_multiplicity_torsions(system)
    return top, system, dropped_offset


def single_point_energy(top_path, gro_path, platform_name='Reference'):
    '''Real single-point potential energy via OpenMM, with the fixes
    above applied. Matches GROMACS's own to float32/float64 rounding
    (verified in `oligomer_builder`, where this fix originates -- see
    `HANDOFF.md`'s verification table).

    `platform_name` defaults to `'Reference'` **on purpose** -- OpenMM's
    slowest but simplest, most portable platform, matching the numbers
    this fix was verified against. Pass `platform_name=None` to let
    OpenMM auto-select the fastest platform actually present instead.

    Returns `(total_energy_kJ_per_mol, dropped_dihedral_offset)`;
    `total_energy_kJ_per_mol` already has the offset added back in.
    '''
    top, system, offset = load_top(top_path, gro_path)
    gro = app.GromacsGroFile(str(gro_path))

    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    if platform_name is None:
        context = mm.Context(system, integrator)
    else:
        platform = mm.Platform.getPlatformByName(platform_name)
        context = mm.Context(system, integrator, platform)
    context.setPositions(gro.getPositions())
    if system.usesPeriodicBoundaryConditions():
        context.setPeriodicBoxVectors(*gro.getPeriodicBoxVectors())

    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return energy + offset, offset
