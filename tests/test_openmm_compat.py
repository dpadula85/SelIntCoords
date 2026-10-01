"""Tests for openmm_compat / joyce_to_openmm on a synthetic Joyce-style
molecule: a 5-site united-atom chain, every pair excluded, one explicit
1-5 LJ pair (funct 2), stiff (funct 2) and flexible (funct 1, incl.
multiplicity 0) torsions -- the shape of a real Joyce `.top`.
"""
from pathlib import Path

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
import openmm.app as app
import openmm.unit as unit

from SelIntCoords import openmm_compat as oc
from SelIntCoords import joyce_to_openmm as j2o

HEADER = """[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
  1  3  no  1.0  0.0

[ atomtypes ]
  CT  12.011  0.000  A  3.50000e-01  2.76144e-01
"""

MOLECULE = """[ moleculetype ]
CHAIN 3

[ atoms ]
  1 CT 1 CHN C1 1  0.2 12.011
  2 CT 1 CHN C2 1 -0.2 12.011
  3 CT 1 CHN C3 1  0.1 12.011
  4 CT 1 CHN C4 1 -0.2 12.011
  5 CT 1 CHN C5 1  0.1 12.011

[ bonds ]
  1 2 1 0.153 250000.0
  2 3 1 0.153 250000.0
  3 4 1 0.153 250000.0
  4 5 1 0.153 250000.0

[ angles ]
  1 2 3 1 112.0 500.0
  2 3 4 1 112.0 500.0
  3 4 5 1 112.0 500.0

[ dihedrals ]
  1 2 3 4 2 180.0 40.0
  2 3 4 5 1 0.0 3.0 0
  2 3 4 5 1 0.0 2.5 1
  2 3 4 5 1 180.0 1.2 3

[ pairs ]
  1 5 2 1.000 0.000 0.000 0.3000 0.2000

[ exclusions ]
  1 2 3 4 5
  2 3 4 5
  3 4 5
  4 5
"""

FOOTER = """
[ system ]
chain

[ molecules ]
CHAIN 1
"""

GRO = """chain
 5
    1CHN     C1    1   0.000   0.000   0.000
    1CHN     C2    2   0.153   0.000   0.000
    1CHN     C3    3   0.210   0.142   0.000
    1CHN     C4    4   0.363   0.150   0.030
    1CHN     C5    5   0.420   0.280   0.110
   5.00000   5.00000   5.00000
"""


def _energy(system, gro_path, offset):
    gro = app.GromacsGroFile(str(gro_path))
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3),
                     mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(gro.getPositions())
    if system.usesPeriodicBoundaryConditions():
        ctx.setPeriodicBoxVectors(*gro.getPeriodicBoxVectors())
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    return e.value_in_unit(unit.kilojoule_per_mole) + offset


@pytest.fixture
def single_file(tmp_path):
    top = tmp_path / "chain.top"
    top.write_text(HEADER + MOLECULE + FOOTER)
    gro = tmp_path / "chain.gro"
    gro.write_text(GRO)
    return top, gro


def test_vacuum_needs_no_gro_and_matches_large_periodic_box(single_file):
    top, gro = single_file
    _, sys_vac, off_vac = oc.load_top(top, nonbonded_method=app.NoCutoff)
    _, sys_pbc, off_pbc = oc.load_top(top, gro)
    assert not sys_vac.usesPeriodicBoundaryConditions()
    # Every pair excluded: nonbonded SR is zero either way, so the two
    # set-ups must agree to rounding.
    assert _energy(sys_vac, gro, off_vac) == pytest.approx(
        _energy(sys_pbc, gro, off_pbc), abs=1e-8)


def test_periodic_method_without_gro_is_refused(single_file):
    top, _ = single_file
    with pytest.raises(ValueError):
        oc.load_top(top)


def test_include_from_sibling_and_parent_dirs_gives_same_energy(tmp_path, single_file):
    top, gro = single_file
    _, sys_ref, off_ref = oc.load_top(top, nonbonded_method=app.NoCutoff)
    e_ref = _energy(sys_ref, gro, off_ref)

    # .top in a subdirectory including an .itp one level up, with a
    # same-named decoy .itp next to the .top that must NOT be used.
    (tmp_path / "ffs").mkdir()
    (tmp_path / "ffs" / "chain.itp").write_text(MOLECULE)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "chain.itp").write_text("; decoy, never included\n")
    split = tmp_path / "run" / "system.top"
    split.write_text(HEADER + '#include "../ffs/chain.itp"\n' + FOOTER)

    _, sys_inc, off_inc = oc.load_top(split, nonbonded_method=app.NoCutoff)
    assert _energy(sys_inc, gro, off_inc) == pytest.approx(e_ref, abs=1e-10)


def test_convert_top_mirrors_include_layout(tmp_path):
    (tmp_path / "ffs").mkdir()
    (tmp_path / "ffs" / "chain.itp").write_text(MOLECULE)
    (tmp_path / "run").mkdir()
    top = tmp_path / "run" / "system.top"
    top.write_text(HEADER + '#include "../ffs/chain.itp"\n' + FOOTER)

    out = j2o.convert_top(top, tmp_path / "converted")
    assert out == tmp_path / "converted" / "run" / "system.top"
    itp = (tmp_path / "converted" / "ffs" / "chain.itp").read_text()
    assert "1      5 1 0.3000 0.2000" in itp


def test_missing_include_is_reported(tmp_path):
    top = tmp_path / "system.top"
    top.write_text(HEADER + '#include "nowhere.itp"\n' + FOOTER)
    with pytest.raises(FileNotFoundError, match="nowhere.itp"):
        oc.load_top(top, nonbonded_method=app.NoCutoff)


def test_bare_pair_line_is_refused(tmp_path):
    bare = MOLECULE.replace("  1 5 2 1.000 0.000 0.000 0.3000 0.2000", "  1 5 1")
    top = tmp_path / "bare.top"
    top.write_text(HEADER + bare + FOOTER)
    with pytest.raises(j2o.UnparameterisedPairError):
        oc.load_top(top, nonbonded_method=app.NoCutoff)


def test_explicit_pair_is_the_only_nonbonded_term(single_file):
    """Charges are nonzero but every pair is excluded: the only
    nonbonded energy is the explicit 1-5 LJ pair, 4 eps[(s/r)^12-(s/r)^6]."""
    top, gro = single_file
    _, system, _ = oc.load_top(top, nonbonded_method=app.NoCutoff)
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
    names = [f.getName() for f in system.getForces()]
    ctx = mm.Context(system, mm.VerletIntegrator(1e-3),
                     mm.Platform.getPlatformByName("Reference"))
    pos = app.GromacsGroFile(str(gro)).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ctx.setPositions(pos)

    def group(name):
        g = names.index(name)
        return ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)

    r = np.linalg.norm(pos[4] - pos[0])
    s, eps = 0.3, 0.2
    assert group("OneFourInteractions") == pytest.approx(
        4 * eps * ((s / r) ** 12 - (s / r) ** 6), rel=1e-10)
    assert group("NonbondedForce") == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("enabled", [False, True])
def test_dispersion_flag_reaches_every_nonbonded_force(single_file, enabled):
    top, gro = single_file
    _, system, _ = oc.load_top(top, gro, use_dispersion_correction=enabled)
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            assert force.getUseDispersionCorrection() is enabled
        elif isinstance(force, mm.CustomNonbondedForce) and force.getName() == "LennardJonesForce":
            assert force.getUseLongRangeCorrection() is enabled
