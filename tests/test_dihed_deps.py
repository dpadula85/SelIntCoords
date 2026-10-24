"""dihed_deps: which stiff coordinates share a force constant because they sit in the same ring."""
import networkx as nx
import numpy as np

from SelIntCoords.dihed_deps import ring_class_groups, tie_dependencies


def benzene():
    """Benzene, ring atoms 0-5, hydrogens 6-11."""
    graph = nx.cycle_graph(6)
    graph.add_edges_from((i, i + 6) for i in range(6))
    return graph, ["C"] * 6 + ["H"] * 6


def ring_dihedrals():
    return [(i % 6, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6) for i in range(6)]


def test_ring_torsions_form_one_group_and_hydrogen_ones_another():
    graph, elements = benzene()
    heavy = ring_dihedrals()
    with_h = [(6, 0, 1, 2), (7, 1, 2, 3), (8, 2, 3, 4)]
    groups, links = ring_class_groups(graph, elements, list(range(12)), np.array(heavy + with_h))
    assert sorted(map(sorted, groups)) == [[0, 1, 2, 3, 4, 5], [6, 7, 8]]
    assert links == []


def test_starlike_impropers_are_linked_to_the_torsions_of_their_ring():
    graph, elements = benzene()
    elements = ["C"] * 12                                                     # heavy substituents on every ring atom
    star = [(i, (i + 1) % 6, (i - 1) % 6, i + 6) for i in range(6)]          # centre first
    groups, links = ring_class_groups(graph, elements, list(range(12)), np.array(ring_dihedrals() + star))
    assert sorted(map(sorted, groups)) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    assert links == [(6, 0)]
    # with hydrogens the starlike impropers pair with the torsions that carry hydrogen, not with the heavy ones
    graph, elements = benzene()
    star = [(i, (i + 1) % 6, (i - 1) % 6, i + 6) for i in range(6)]
    with_h = [(6, 0, 1, 2), (7, 1, 2, 3)]
    groups, links = ring_class_groups(graph, elements, list(range(12)), np.array(ring_dihedrals() + with_h + star))
    assert links == [(8, 6)]


def test_equivalent_rings_share_the_group_and_different_rings_do_not():
    # biphenyl-like: two benzene rings joined by 0-6; ring 2 carries atoms 6-11, hydrogens after
    graph = nx.cycle_graph(6)
    graph.add_edges_from(nx.relabel_nodes(nx.cycle_graph(6), lambda i: i + 6).edges)
    graph.add_edge(0, 6)
    elements = ["C"] * 12
    orbit_same = [i % 6 for i in range(12)]
    first = [(i, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6) for i in range(6)]
    second = [(6 + a % 6, 6 + (a + 1) % 6, 6 + (a + 2) % 6, 6 + (a + 3) % 6) for a in range(6)]
    coords = np.array(first + second)
    groups, _ = ring_class_groups(graph, elements, orbit_same, coords)
    assert len(groups) == 1 and len(groups[0]) == 12
    groups, _ = ring_class_groups(graph, elements, list(range(12)), coords)
    assert len(groups) == 2 and all(len(g) == 6 for g in groups)


def test_a_coordinate_outside_any_ring_is_not_grouped():
    graph = nx.path_graph(5)
    groups, links = ring_class_groups(graph, ["C"] * 5, list(range(5)), np.array([(0, 1, 2, 3)]))
    assert groups == [] and links == []


def test_fused_bond_goes_to_the_larger_ring():
    # naphthalene-like: a six-ring (0-5) fused on 0-1 with a five-ring (0, 1, 6, 7, 8)
    graph = nx.cycle_graph(6)
    graph.add_edges_from([(1, 6), (6, 7), (7, 8), (8, 0)])
    groups, _ = ring_class_groups(graph, ["C"] * 9, list(range(9)), np.array([(5, 0, 1, 2), (8, 0, 1, 6), (0, 1, 2, 3)]))
    # (5,0,1,2) and (0,1,2,3): central bonds 0-1 (shared) and 1-2 (six-ring) -> six-ring; (8,0,1,6): five-ring bond 0-1 too, larger ring wins
    assert sorted(map(sorted, groups)) == [[0, 1, 2]]


def test_each_tie_says_why_and_is_written_once_against_the_lowest_index():
    symmetry = np.array([[1, 2], [3, 4]])
    rings = np.array([[2, 3], [10, 11]])
    ties = tie_dependencies(symmetry, rings, [(5, 0)], first_stiff=10)
    # 1~2 and 3~4 by symmetry, 2~3 by the ring: one class {1, 2, 3, 4}; 10~11 by the ring; the starlike 15 x5 to coordinate 10
    assert ties == [(2, 1, 1, 'symmetry'), (3, 1, 1, 'ring'), (4, 1, 1, 'ring'), (11, 10, 1, 'ring'),
                    (15, 10, 5, 'starlike')]
