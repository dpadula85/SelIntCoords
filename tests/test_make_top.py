"""make_top: internal-coordinate counts for the example cores.

The counts below do not depend on the symmetry of the molecule.
"""
import re
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


@pytest.mark.parametrize("name", ["BTBT", "DPP"])
def test_dependence_file_ties_each_coordinate_once_to_the_lowest_of_its_class(name, examples, tmp_path):
    out = tmp_path / f"{name}.top"
    subprocess.run([sys.executable, "-m", "SelIntCoords.make_top", "-m", str(examples / name / f"{name}.xyz"),
                    "-o", str(out)], check=True)
    ties = [tuple(int(x) for x in re.findall(r"\d+", line)[:2])
            for line in (tmp_path / f"{name}_deps.dat").read_text().split("\n") if "=" in line]
    dependents = [a for a, _ in ties]
    assert ties and len(set(dependents)) == len(dependents)
    # a reference is never itself tied, and sits below everything that depends on it
    assert not {b for _, b in ties} & set(dependents)
    assert all(b < a for a, b in ties)
    # every tie says what it is and why
    lines = [line for line in (tmp_path / f"{name}_deps.dat").read_text().split("\n") if "=" in line]
    assert all(" ; " in line and ": " in line.split(" ; ")[1] for line in lines)
    assert any(line.endswith(": symm eq") for line in lines) and any("same k in ring" in line for line in lines)
    assert max(len(line) for line in lines) < 140


def test_make_top_cli_with_symmetry(examples, tmp_path):
    out = tmp_path / "BTBT_symm.top"
    subprocess.run([sys.executable, "-m", "SelIntCoords.make_top", "-p", str(examples / "BTBT" / "BTBT.top"),
                    "-m", str(examples / "BTBT" / "BTBT.xyz"), "-o", str(out)], check=True)
    assert out.exists() and (tmp_path / "BTBT_symm_deps.dat").exists()
    csv = (tmp_path / "BTBT_symm.csv").read_text()
    assert '"Bonds",39,1,39' in csv and '"Flex_dihedrals",20,156,175' in csv


def _brute_force_deps(coords, eqs):
    """Pairs of coordinates linked by substituting each atom by an equivalent one."""
    from itertools import product
    pairs = set()
    for c, coord in enumerate(coords):
        for sub in product(*[eqs[i] for i in coord]):
            for d in np.where((coords == np.asarray(sub)).all(axis=1))[0]:
                if d != c:
                    pairs.add((min(c, d), max(c, d)))
    return pairs


@pytest.mark.parametrize("name", ["BTBT", "D18"])
def test_find_deps_equals_the_substitution_of_equivalent_atoms(name, examples):
    from SelIntCoords.make_top import find_deps
    data = si.list_intcoords(str(examples / name / f"{name}.xyz"))
    eq = data[-1]
    for coords in (data[0], data[1], data[2], data[4]):
        coords = np.asarray(coords)
        found = find_deps(coords, eq)
        pairs = {tuple(p) for p in found.tolist()} if found.shape[0] > 1 else set()
        assert pairs == _brute_force_deps(coords, eq)


def test_find_deps_on_large_equivalence_classes_is_not_exponential():
    """60 equivalent atoms (a fullerene): the substitution of every atom of a dihedral would be 60**4."""
    from SelIntCoords.make_top import find_deps
    eq = {i: np.arange(60) for i in range(60)}
    coords = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    found = find_deps(coords, eq)
    assert sorted({tuple(p) for p in found.tolist()}) == [(0, 1), (0, 2), (1, 2)]


def test_molecule_without_hydrogens_has_no_equivalent_hydrogens(tmp_path):
    import MDAnalysis as mda
    import networkx as nx
    a = np.arange(6) * np.pi / 3
    xyz = tmp_path / "c6.xyz"
    xyz.write_text("6\n\n" + "\n".join("C %.4f %.4f 0.0" % (1.4 * np.cos(t), 1.4 * np.sin(t)) for t in a) + "\n")
    u = mda.Universe(str(xyz), guess_bonds=True)
    assert si.get_equivalent_hydrogens(u, nx.Graph(u.bonds.to_indices().tolist())) == {}
