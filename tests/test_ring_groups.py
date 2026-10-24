import numpy as np
import pytest

import SelIntCoords.sel_intcoords as si
from SelIntCoords.flex_classes import classify_flex, is_double_bond
from SelIntCoords.mol_features import Features, read_xyz
from SelIntCoords.ring_groups import RingGrouper, insert_headers, ring_name, ring_systems
from SelIntCoords.symmetry import topological_classes


def features(xyz):
    data = si.list_intcoords(str(xyz))
    elements, coords = read_xyz(xyz)
    return Features(elements, coords, data[0]), data


def same_atoms(f, bonds):
    classes = topological_classes(f.el, bonds)
    return {i: np.where(classes == c)[0] for i, c in enumerate(classes)}


def ring(centre, radius=1.4, n=6, start=0.0):
    angles = start + np.arange(n) * 2 * np.pi / n
    return np.array([[centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), 0.0] for a in angles])


@pytest.mark.parametrize("name, expected", [
    ("PN", ["pentacene"]),
    ("BTBT", ["2 benzene + 2 thiophene (fused)"]),
    ("D18", ["thiophene 1", "benzene + thiadiazole + 2 thiophene (fused)", "thiophene 2",
             "benzene + 2 thiophene (fused)", "thiophene 3", "thiophene 4"]),
])
def test_ring_systems_are_fused_by_shared_bonds_and_named(name, expected, examples):
    f, _ = features(examples / name / f"{name}.xyz")
    assert [s.name for s in ring_systems(f)] == expected


def test_single_rings_are_named_by_kind(examples):
    f, _ = features(examples / "D18" / "D18.xyz")
    names = [ring_name(f, r) for r in f.rings]
    assert {"benzene", "thiophene", "thiadiazole"} <= set(names)


def test_every_ring_atom_has_one_owner_and_hydrogens_follow_their_atom(examples):
    f, _ = features(examples / "D18" / "D18.xyz")
    grouper = RingGrouper(f)
    in_rings = {i for r in f.rings for i in r}
    assert in_rings <= set(grouper.owner)
    for i in range(f.n):
        if f.el[i] == "H" and f.nbrs[i][0] in in_rings:
            assert grouper.owner[i] == grouper.owner[f.nbrs[i][0]]


def test_grouping_is_a_permutation_and_each_header_appears_once(examples):
    f, data = features(examples / "D18" / "D18.xyz")
    grouper = RingGrouper(f)
    for coords in (data[0], data[1], np.vstack([data[2], data[4]])):
        order, labels = grouper.group(coords)
        assert sorted(order.tolist()) == list(range(len(coords)))
        runs = [labels[0]] + [b for a, b in zip(labels, labels[1:]) if a != b]
        assert len(runs) == len(set(runs))


def test_links_and_methyls_get_their_own_groups(examples):
    f, data = features(examples / "D18" / "D18.xyz")
    grouper = RingGrouper(f)
    _, labels = grouper.group(data[0])
    assert any(" to " in label for label in labels)
    assert "methyls" in labels
    # nothing but the methyl groups themselves is a methyl coordinate
    order, labels = grouper.group(data[0])
    for k, label in zip(order, labels):
        if label == "methyls":
            assert any(i in grouper.methyls for i in data[0][k])


def test_a_chain_between_two_ring_systems_is_a_link():
    # two benzenes joined by a vinylene: ring carbons 0-5 and 6-11, the chain
    # carbons 12-15, then one hydrogen on every ring carbon but 0 and 6
    centres = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    rings = np.vstack([ring(centres[0]), ring(centres[1], start=np.pi)])
    chain = np.array([[2.4, 0.0, 0.0], [3.7, 0.0, 0.0], [4.3, 0.0, 0.0], [5.6, 0.0, 0.0]])
    free = [i for i in range(12) if i not in (0, 6)]
    outward = [rings[i] - centres[i // 6] for i in free]
    hydrogens = np.array([rings[i] + 1.08 * v / np.linalg.norm(v) for i, v in zip(free, outward)])
    coords = np.vstack([rings, chain, hydrogens])
    elements = ["C"] * 16 + ["H"] * len(free)
    bonds = [(6 * r + i, 6 * r + (i + 1) % 6) for r in (0, 1) for i in range(6)]
    bonds += [(0, 12), (12, 13), (13, 14), (14, 15), (15, 6)] + [(i, 16 + k) for k, i in enumerate(free)]
    f = Features(elements, coords, np.array(bonds))
    grouper = RingGrouper(f)
    names = [s.name for s in grouper.systems]
    assert names == ["benzene 1", "benzene 2"]
    assert grouper.label([0, 12])[1] == "benzene 1 to benzene 2"
    assert grouper.label([12, 13, 14])[1] == "benzene 1 to benzene 2"
    assert grouper.label([0, 1])[1] == "benzene 1"
    assert grouper.label([6, 7, 8])[1] == "benzene 2"


def test_insert_headers_goes_before_each_run(tmp_path):
    top = tmp_path / "t.top"
    top.write_text("[ bonds ]\n;  3 bonds\n  1 2 1 0.0 0.0 ; 1\n  2 3 1 0.0 0.0 ; 2\n  3 4 1 0.0 0.0 ; 3\n\n[ pairs ]\n  1 3 2\n")
    insert_headers(str(top), {1: "ring A", 2: "ring A", 3: "ring B"})
    lines = top.read_text().splitlines()
    assert lines[:8] == ["[ bonds ]", ";  3 bonds", "; ring A", "  1 2 1 0.0 0.0 ; 1",
                         "  2 3 1 0.0 0.0 ; 2", "; ring B", "  3 4 1 0.0 0.0 ; 3", ""]
    assert "; ring" not in "\n".join(lines[8:])


def test_topological_classes_split_what_the_graph_distinguishes():
    # propane: the two methyl carbons are alike, the CH2 is not
    elements = ["C", "C", "C"] + ["H"] * 8
    bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8), (2, 9), (2, 10)]
    classes = topological_classes(elements, np.array(bonds))
    assert classes[0] == classes[2] != classes[1]
    assert len({classes[i] for i in (3, 4, 5, 8, 9, 10)}) == 1
    assert classes[6] == classes[7] != classes[3]


def test_methyl_rotors_take_literature_parameters_and_scans_one_dihedral(examples):
    f, data = features(examples / "D18" / "D18.xyz")
    flex = data[4]
    plan = classify_flex(f, flex, same_atoms(f, data[0]))
    assert plan.counts == {"methyl": 36, "ring": 0, "scan": 9, "phi": 0, "double": 0, "other": 0}
    # every dihedral kept was a flexible one
    assert {tuple(d) for d in plan.flex.tolist()} <= {tuple(d) for d in np.asarray(flex).tolist()}
    assert len(plan.labels) == len(plan.params) == len(plan.flex)
    for label, par, d in zip(plan.labels, plan.params, plan.flex):
        if label == "methyls":
            assert par == {"delta": 0.0, "kchi": 0.651, "n": 3}
            assert "H" in (f.el[d[0]], f.el[d[3]])
        elif label.startswith("delta"):
            assert par == {"delta": 0.0, "kchi": 0.0, "n": 2}
    # methyls first, then the scanned rotors, equivalent bonds under one header
    kinds = ["methyls" if l == "methyls" else "scan" if l.startswith("delta") else "other" for l in plan.labels]
    assert kinds == sorted(kinds, key={"methyls": 0, "scan": 1, "other": 2}.get)
    assert len({l for l in plan.labels if l.startswith("delta")}) <= plan.counts["scan"]


def test_scanned_rotor_keeps_the_dihedral_over_the_carbons():
    # two benzenes bonded through atoms 0 and 6, a sulfur and a carbon next to each
    # atom 0 of the first ring and atom 6 (first of the second ring) face each other 1.5 A apart
    a, b = ring((0.0, 0.0)), ring((4.3, 0.0), start=np.pi)
    coords = np.vstack([a, b])
    elements = ["C", "S", "C", "C", "C", "C"] + ["C", "C", "C", "C", "C", "S"]
    bonds = [(i, (i + 1) % 6) for i in range(6)] + [(6 + i, 6 + (i + 1) % 6) for i in range(6)] + [(0, 6)]
    f = Features(elements, coords, np.array(bonds))
    # dihedrals i-0-6-l over the neighbours 1, 5 of atom 0 and 7, 11 of atom 6
    dih = np.array([[i, 0, 6, l] for i in (1, 5) for l in (7, 11)])
    plan = classify_flex(f, dih)
    assert plan.labels == ["delta1"] and plan.counts["scan"] == 1
    # atom 1 and atom 11 are sulfur: the carbon neighbours 5 and 7 are taken
    assert tuple(plan.flex[0]) == (5, 0, 6, 7)


def test_double_bond_is_told_from_a_single_bond_by_its_length():
    # butadiene-like planar chain C=C-C=C: 1.34, 1.46, 1.34 Angstrom, each carbon
    # given the neighbours that make it planar
    x = np.cumsum([0.0, 1.34, 1.46, 1.34])
    coords = np.array([[xi, 0.0, 0.0] for xi in x])
    bonds = [(0, 1), (1, 2), (2, 3)]
    helpers = np.array([
        [-0.67, 1.16, 0.0], [-0.67, -1.16, 0.0],
        [x[1], 1.1, 0.0], [x[2], -1.1, 0.0],
        [x[3] + 0.67, 1.16, 0.0], [x[3] + 0.67, -1.16, 0.0]])
    coords = np.vstack([coords, helpers])
    bonds += [(0, 4), (0, 5), (1, 6), (2, 7), (3, 8), (3, 9)]
    f = Features(["C"] * 4 + ["H"] * 6, coords, np.array(bonds))
    assert is_double_bond(f, 0, 1) and is_double_bond(f, 2, 3)
    assert not is_double_bond(f, 1, 2)


DATA = __import__("pathlib").Path(__file__).resolve().parent / "data"


def plan_of(xyz):
    f, data = features(xyz)
    return f, classify_flex(f, data[4], same_atoms(f, data[0]))


def by_label(plan, prefix):
    """1-based atom quadruplets of the dihedrals under headers starting with `prefix`."""
    return sorted(tuple(int(i) + 1 for i in d) for d, l in zip(plan.flex, plan.labels) if l.startswith(prefix))


def test_retinal_chain_gets_the_deltas_and_phis_of_the_hand_made_topology():
    """11-cis retinal PSB: the dihedrals the hand-made Joyce topology scans (delta1-5 single, phi1-5 double bonds)."""
    f, plan = plan_of(DATA / "rPSB11.xyz")
    assert by_label(plan, "delta") == [(11, 12, 13, 15), (13, 15, 17, 18), (17, 18, 20, 22), (20, 22, 24, 25), (24, 25, 27, 53)]
    assert by_label(plan, "phi") == [(12, 13, 15, 17), (15, 17, 18, 20), (18, 20, 22, 24), (22, 24, 25, 27), (25, 27, 53, 49)]
    assert plan.counts["double"] == 0
    # the ring torsions take the literature value, like the methyl rotation
    assert plan.counts["ring"] > 0
    for label, par in zip(plan.labels, plan.params):
        if label.startswith("ring"):
            assert par == {"delta": 0.0, "kchi": 0.651, "n": 3}
    assert sorted({l for l in plan.labels if l.startswith("phi")}) == ["phi1", "phi2", "phi3", "phi4", "phi5"]


ZR1_DELTAS = {
    "delta1": {(2, 1, 19, 23), (5, 4, 24, 28)},        # core - phenyl
    "delta2": {(10, 11, 29, 33), (16, 17, 34, 38)},    # core - thiophene
    "delta3": {(32, 31, 39, 43), (37, 36, 44, 48)},    # thiophene - thiophene
    "delta4": {(42, 41, 49, 50), (47, 46, 57, 58)},    # thiophene - rhodanine
    "delta5": {(27, 26, 82, 101), (22, 21, 79, 97)},   # phenyl - CH2 of the ethyl
    "delta6": {(33, 32, 85, 105), (38, 37, 88, 109)},  # thiophene - CH2 of the ethyl
    "delta7": {(52, 54, 91, 117), (60, 62, 94, 113)},  # rhodanine N - CH2 of the ethyl
}


def test_zr1_scans_and_exocyclic_double_bonds_match_the_hand_made_topology():
    """ZR1 Step1: delta1-7 are these dihedrals (junctions first, then the first bond of each side chain), the methyl rotations come before them, and the C=C bonds joined to a ring are stiff."""
    f, plan = plan_of(DATA / "ZR1.xyz")
    canon = lambda d: min(d, d[::-1])
    for label, dihedrals in ZR1_DELTAS.items():
        mine = {canon(tuple(int(i) + 1 for i in d)) for d, l in zip(plan.flex, plan.labels) if l == label}
        assert mine == {canon(d) for d in dihedrals}, label
    assert plan.counts["methyl"] == 54 and plan.counts["other"] == 0 and plan.counts["scan"] == 14
    # methyls first, then the deltas in order
    first_delta = next(n for n, l in enumerate(plan.labels) if l.startswith("delta"))
    assert set(plan.labels[:first_delta]) == {"methyls"}
    stiff = {tuple(int(i) + 1 for i in d) for d in plan.double}
    assert {(41, 49, 50, 51), (41, 49, 50, 55), (77, 49, 50, 51), (77, 49, 50, 55)} <= stiff
    assert plan.counts["phi"] == 0


def side_chain_molecule(kind):
    """A benzene ring (atoms 0-5) with a substituent on atom 0: ethyl, S-methyl or CF3."""
    ring_xyz = ring((0.0, 0.0))
    if kind == "ethyl":
        extra = [[2.9, 0.0, 0.0], [3.4, 1.45, 0.0], [3.3, -0.5, 0.9], [3.3, -0.5, -0.9]]
        elements = ["C"] * 6 + ["C", "C", "H", "H"]
        bonds = [(0, 6), (6, 7), (6, 8), (6, 9)]
        far = [7, 8, 9]
    elif kind == "thioether":
        extra = [[3.15, 0.0, 0.0], [3.9, 1.5, 0.0]]
        elements = ["C"] * 6 + ["S", "C"]
        bonds = [(0, 6), (6, 7)]
        far = [7]
    else:
        extra = [[2.9, 0.0, 0.0], [3.4, 1.1, 0.0], [3.4, -0.55, 0.95], [3.4, -0.55, -0.95]]
        elements = ["C"] * 6 + ["C", "F", "F", "F"]
        bonds = [(0, 6), (6, 7), (6, 8), (6, 9)]
        far = [7, 8, 9]
    bonds = [(i, (i + 1) % 6) for i in range(6)] + bonds
    f = Features(elements, np.vstack([ring_xyz, extra]), np.array(bonds))
    return f, np.array([[i, 0, 6, l] for i in (1, 5) for l in far])


@pytest.mark.parametrize("kind", ["ethyl", "thioether"])
def test_the_first_bond_of_a_side_chain_is_scanned(kind):
    f, dih = side_chain_molecule(kind)
    plan = classify_flex(f, dih)
    assert plan.labels == ["delta1"] and plan.counts["scan"] == 1 and plan.counts["other"] == 0
    # the dihedral runs over the ring carbon and the heavy atom beyond the first atom of the chain
    assert tuple(plan.flex[0])[1:3] == (0, 6) and f.el[plan.flex[0][3]] == "C" and f.el[plan.flex[0][0]] == "C"


def test_a_group_that_turns_as_a_unit_is_not_a_side_chain_scan():
    f, dih = side_chain_molecule("cf3")
    plan = classify_flex(f, dih)
    assert plan.counts["scan"] == 0 and plan.counts["other"] == len(dih)


def test_equivalent_bonds_take_the_same_dihedral(examples):
    # the two ethyls of BTBT: the dihedral of the second is the symmetry image of the first
    f, data = features(examples / "BTBT" / "BTBT.xyz")
    plan = classify_flex(f, data[4], same_atoms(f, data[0]))
    assert plan.counts["scan"] == 2 and sorted({l for l in plan.labels if l.startswith("delta")}) == ["delta1"]
    (a, b) = [tuple(int(i) for i in d) for d, l in zip(plan.flex, plan.labels) if l == "delta1"]
    assert [f.el[i] for i in a] == [f.el[i] for i in b] and [f.n_h(i) for i in a] == [f.n_h(i) for i in b]


def test_ring_locking_the_retinal_chain_is_a_cyclohexene():
    f, _ = features(DATA / "rPSB11.xyz")
    assert [s.name for s in ring_systems(f)] == ["cyclohexene"]


def test_azobenzene_with_ether_linkers_scans_every_bond_between_the_rigid_units():
    """Azo dye with O-CH2-CH2-imidazole arms: the N=N bond, the C-N bonds and the four bonds of each linker are scanned, as in the hand-made topology."""
    f, plan = plan_of(DATA / "DAB.xyz")
    assert plan.counts["methyl"] == 0 and plan.counts["other"] == 0 and plan.counts["phi"] == 1
    assert by_label(plan, "phi") == [(3, 15, 16, 17)]
    deltas = by_label(plan, "delta")
    assert len(deltas) == 10
    # one dihedral per linker bond on the first arm, the same on the second, equivalent bonds under one header
    assert {(4, 5, 7, 8), (5, 7, 8, 9), (7, 8, 9, 10), (8, 9, 10, 11), (2, 3, 15, 16)} <= set(deltas)
    assert sorted({l for l in plan.labels if l.startswith("delta")}) == ["delta1", "delta2", "delta3", "delta4", "delta5"]
