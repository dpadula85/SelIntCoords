"""make_top: internal-coordinate counts for the example cores.

Point-group detection needs psi4 (conda-only), so most tests replace it with
"no symmetry"; the counts below do not depend on it. The one test that does
is skipped when psi4 is absent.
"""
import subprocess
import sys

import numpy as np
import pytest

from SelIntCoords import sel_intcoords as si
from SelIntCoords.make_top import add_terms, find_deps, geom_avg_mixing, avg_mixing

# bonds, angles, stiff, impropers, flexible, LJ 1,4 / 1,5 / 1,6 / 1,7 / other
COUNTS = {
    "BTBT": (39, 68, 34, 14, 30, (82, 81, 75, 55, 230)),
    "DNBDT": (48, 80, 64, 26, 0, (103, 106, 91, 79, 354)),
    "PN": (40, 66, 52, 22, 0, (89, 92, 79, 64, 200)),
    "DPP": (51, 88, 46, 20, 38, (100, 99, 114, 119, 464)),
    "PNDI2OD": (57, 98, 52, 24, 38, (124, 132, 141, 137, 637)),
    "D18": (97, 167, 84, 36, 80, (195, 206, 222, 210, 2644)),
    "PBDTTT-O": (69, 118, 54, 25, 60, (135, 134, 149, 151, 1197)),
}


@pytest.fixture
def no_symmetry(monkeypatch):
    monkeypatch.setattr(si, "get_equivalent_atoms", lambda u: {})


@pytest.mark.parametrize("name", COUNTS)
def test_internal_coordinate_counts(name, examples, no_symmetry):
    bds, angs, stiff, imps, flex, LJs, excls, rings, eq = si.list_intcoords(str(examples / name / f"{name}.xyz"))
    nb, na, ns, ni, nf, nlj = COUNTS[name]
    assert (len(bds), len(angs), len(stiff), len(imps), len(flex)) == (nb, na, ns, ni, nf)
    assert tuple(len(LJs[k]) for k in ("1,4", "1,5", "1,6", "1,7", "other")) == nlj


def test_add_terms_writes_every_term(examples, no_symmetry, tmp_path):
    bds, angs, stiff, imps, flex, LJs, excls, rings, eq = si.list_intcoords(str(examples / "BTBT" / "BTBT.xyz"))
    top = add_terms(str(examples / "BTBT" / "BTBT.top"), bds, angs, stiff, imps, flex, LJs, excls,
                    mixing=geom_avg_mixing)
    mol = top.molecules[0]
    assert (len(mol.bonds), len(mol.angles)) == (39, 68)
    assert len(mol.dihedrals) == 30  # flexible; stiff + improper are stored as impropers
    assert len(mol.impropers) == 34 + 14
    out = tmp_path / "out.top"
    top.write(str(out))
    assert out.stat().st_size > 0


def test_mixing_rules():
    assert geom_avg_mixing(4.0, 1.0, 0.4, 0.1) == pytest.approx((2.0, 0.2))
    eps, sig = avg_mixing(4.0, 1.0, 0.4, 0.1)
    assert sig == pytest.approx(0.25)


def test_find_deps_pairs_equivalent_coordinates():
    coords = np.array([[0, 1], [2, 3], [4, 5]])
    eqs = {0: [0, 2], 1: [1, 3], 2: [2, 0], 3: [3, 1], 4: [4], 5: [5]}
    assert find_deps(coords, eqs).tolist() == [[0, 1]]


def test_find_deps_without_equivalences_is_empty():
    coords = np.array([[0, 1], [2, 3]])
    eqs = {i: [i] for i in range(4)}
    assert find_deps(coords, eqs).size == 0


def test_missing_psi4_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(si, "psi4", None)
    with pytest.raises(ImportError, match="psi4"):
        si.get_equivalent_atoms(None)


def test_make_top_cli_with_symmetry(examples, tmp_path):
    pytest.importorskip("psi4")
    out = tmp_path / "BTBT_symm.top"
    subprocess.run([sys.executable, "-m", "SelIntCoords.make_top", "-p", str(examples / "BTBT" / "BTBT.top"),
                    "-m", str(examples / "BTBT" / "BTBT.xyz"), "-o", str(out)], check=True)
    assert out.exists() and (tmp_path / "BTBT_symm_deps.dat").exists()
    csv = (tmp_path / "BTBT_symm.csv").read_text()
    assert '"Bonds",39,1,39' in csv and '"Flex_dihedrals",30,156,185' in csv
