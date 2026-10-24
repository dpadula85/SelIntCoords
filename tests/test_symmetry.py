"""symmetry: atoms equivalent by the point-group symmetry of a geometry.

The class counts of the example cores are those the previous detection
(a symmetrised geometry through psi4 and pyscf's point-group search) gave.
"""
import networkx as nx
import numpy as np
import pytest

from SelIntCoords.mol_features import read_xyz
from SelIntCoords import sel_intcoords as si
from SelIntCoords.symmetry import (atom_orbits, equivalent_atoms, graph_equivalences, pair_orbit_closure,
                                   symmetry_permutations)


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


def test_atom_orbits_follow_the_symmetry_and_the_colours():
    star = nx.star_graph(3)                                              # centre 0, leaves 1-3
    assert [o.tolist() for o in atom_orbits(star, [0, 1, 1, 1])] == [[0], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert [o.tolist() for o in atom_orbits(star, [0, 1, 2, 1])][1] == [1, 3]


def test_pair_orbits_are_orbits_of_pairs_not_products_of_atom_orbits():
    # two identical arms joined at the centre: the pairs within an arm and across arms are different orbits
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])   # mirror: i <-> 6 - i
    colours = [0, 1, 2, 3, 2, 1, 0]
    assert pair_orbit_closure(graph, colours, {(0, 4)}) == {(0, 4), (2, 6)}
    assert pair_orbit_closure(graph, colours, {(0, 6)}) == {(0, 6)}
    # a pair joining the two mirror images is not an image of a pair within one arm
    assert (1, 5) not in pair_orbit_closure(graph, colours, {(0, 4)})
    # a lower symmetry of the colours gives smaller orbits
    assert pair_orbit_closure(graph, [0, 1, 2, 3, 2, 1, 4], {(0, 4)}) == {(0, 4)}


def test_gem_dimethyl_hydrogens_are_one_orbit():
    # C0 carries two methyls (C1, C2) and two other carbons (C3, C4); every methyl has three H
    graph = nx.Graph([(0, 1), (0, 2), (0, 3), (0, 4)])
    nxt = 5
    for c in (1, 2):
        for _ in range(3):
            graph.add_edge(c, nxt)
            nxt += 1
    eq = graph_equivalences(graph, ["C"] * 5 + ["H"] * 6)
    assert eq[5].tolist() == [5, 6, 7, 8, 9, 10]
    assert eq[1].tolist() == [1, 2]
    assert eq[3].tolist() == [3, 4]


def test_graph_equivalences_do_not_need_a_symmetric_geometry():
    # butane skeleton: the two ends are equivalent by the graph whatever the torsion
    graph = nx.path_graph(4)
    assert graph_equivalences(graph, ["C"] * 4)[0].tolist() == [0, 3]
    assert graph_equivalences(graph, ["C", "C", "C", "N"])[0].tolist() == [0]


@pytest.mark.parametrize("name", ["BTBT", "DNBDT", "PN", "DPP"])
def test_graph_orbits_contain_those_of_the_point_group(name, examples):
    el, xyz = read_xyz(examples / name / f"{name}.xyz")
    eq = equivalent_atoms(el, xyz)
    bonds = si.list_intcoords(str(examples / name / f"{name}.xyz"))[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(el)))
    graph.add_edges_from(bonds)
    orbit = graph_equivalences(graph, el)
    assert all(set(eq[i]) | {i} <= set(orbit[i]) for i in eq)


def test_list_intcoords_equivalences_follow_the_graph(examples):
    # the two halves of DPP are equivalent by the graph, but the geometry is not symmetric within the tolerance
    eq = si.list_intcoords(str(examples / "DPP" / "DPP.xyz"))[-1]
    assert 22 in eq[0] and len(eq) == 46
    # an atom type that tells the two halves apart splits the orbit
    el, _ = read_xyz(examples / "DPP" / "DPP.xyz")
    types = [f"{e}1" for e in el]
    types[22] = "special"
    eq = si.list_intcoords(str(examples / "DPP" / "DPP.xyz"), atom_types=types)[-1]
    assert eq[0].tolist() == [0]
