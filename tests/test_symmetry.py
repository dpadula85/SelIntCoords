"""symmetry: atoms equivalent by the point-group symmetry of a geometry.

The class counts of the example cores are those the previous detection
(a symmetrised geometry through psi4 and pyscf's point-group search) gave.
"""
import numpy as np
import pytest

from SelIntCoords.mol_features import read_xyz
from SelIntCoords.symmetry import equivalent_atoms, symmetry_permutations


def classes(eq):
    return {tuple(sorted((k,) + tuple(v))) for k, v in eq.items()}


def benzene():
    ang = np.arange(6) * np.pi / 3
    c = 1.40 * np.c_[np.cos(ang), np.sin(ang), np.zeros(6)]
    h = 2.48 * np.c_[np.cos(ang), np.sin(ang), np.zeros(6)]
    return ["C"] * 6 + ["H"] * 6, np.r_[c, h]


@pytest.mark.parametrize("name, n_classes", [("BTBT", 18), ("DNBDT", 21), ("PN", 10), ("DPP", 46)])
def test_example_cores(name, n_classes, examples):
    el, xyz = read_xyz(examples / name / f"{name}.xyz")
    assert len(classes(equivalent_atoms(el, xyz))) == n_classes


def test_benzene_has_the_full_non_abelian_group():
    el, xyz = benzene()
    eq = equivalent_atoms(el, xyz)
    assert classes(eq) == {tuple(range(6)), tuple(range(6, 12))}
    # D6h has 24 operations, but the mirror in the molecular plane leaves every atom in place
    assert len(symmetry_permutations(el, xyz)) == 12


def test_every_atom_is_a_key_and_singletons_are_empty():
    eq = equivalent_atoms(["O", "H", "H"], np.array([[0, 0, 0.12], [0.76, 0, -0.48], [-0.76, 0, -0.48]]))
    assert sorted(eq) == [0, 1, 2]
    assert eq[0].size == 0 and eq[1].tolist() == [2]


def test_ethane_staggered():
    z = 0.77
    ang = np.arange(3) * 2 * np.pi / 3
    ch1 = 1.0 * np.c_[np.cos(ang), np.sin(ang), np.full(3, z + 0.37)]
    ch2 = 1.0 * np.c_[np.cos(ang + np.pi / 3), np.sin(ang + np.pi / 3), np.full(3, -z - 0.37)]
    xyz = np.r_[[[0, 0, z], [0, 0, -z]], ch1, ch2]
    eq = equivalent_atoms(["C", "C"] + ["H"] * 6, xyz)
    assert classes(eq) == {(0, 1), tuple(range(2, 8))}


def test_linear_molecule():
    eq = equivalent_atoms(["O", "C", "O"], np.array([[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]]))
    assert classes(eq) == {(0, 2), (1,)}


def test_tolerance_accepts_noise_and_rejects_distortion():
    el, xyz = benzene()
    rng = np.random.default_rng(0)
    noisy = xyz + rng.normal(scale=0.002, size=xyz.shape)
    assert len(classes(equivalent_atoms(el, noisy, 0.01))) == 2
    distorted = xyz.copy()
    distorted[0] *= 1.1
    assert len(classes(equivalent_atoms(el, distorted, 0.01))) > 2


def test_chiral_molecule_has_no_improper_operation():
    """CHFClBr: four different substituents on a tetrahedron, nothing is equivalent."""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * 0.63
    el = ["C", "H", "F", "Cl", "Br"]
    eq = equivalent_atoms(el, np.r_[[[0, 0, 0]], v])
    assert all(len(v) == 0 for v in eq.values())


def test_labels_keep_isotopes_apart():
    el, xyz = benzene()
    el[6] = "D"
    eq = equivalent_atoms(el, xyz)
    assert 6 not in {i for k in eq for i in eq[k]} and eq[6].size == 0
