"""pair_selection: which 1-n pairs a topology needs, from rigid-rotor conformers.

Small graphs for the pieces (rotors, symmetry), syn-pentane for the physics, and the trans azo dye
(`tests/data/DAB.xyz`) for the invariants of the whole selection.
"""
import csv
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from SelIntCoords import pair_selection as ps
from SelIntCoords.flex_classes import double_bonds
from SelIntCoords.mol_features import Features

DATA = Path(__file__).resolve().parent / "data"

# Lennard-Jones sigma (nm) of OPLS-AA carbon, nitrogen, oxygen and hydrogen
SIGMA = {"C": 0.35, "N": 0.325, "O": 0.296, "H": 0.25}


def pentane_skeleton():
    """Carbon skeleton of all-trans pentane (C-C 1.53 A, C-C-C 112.4 degrees)."""
    half = np.radians((180.0 - 112.4) / 2)
    xyz = np.array([[i * 1.53 * np.cos(half), (i % 2) * 1.53 * np.sin(half), 0.0] for i in range(5)])
    return nx.path_graph(5), xyz


def test_rotors_of_a_chain_turn_the_smaller_side():
    graph, _ = pentane_skeleton()
    rotors = ps.find_rotors(graph, set())
    # the bonds to the terminal atoms do not turn anything
    assert [(a, b, m.tolist()) for a, b, m in rotors] == [(2, 1, [0, 1]), (2, 3, [3, 4])]


def test_double_bonds_are_not_rotors():
    graph, _ = pentane_skeleton()
    rotors = ps.find_rotors(graph, {(1, 2)})
    assert [(a, b) for a, b, _ in rotors] == [(2, 3)]


def test_ring_bonds_are_not_rotors():
    assert ps.find_rotors(nx.cycle_graph(6), set()) == []


def test_ring_systems_merge_fused_rings():
    graph = nx.cycle_graph(6)
    graph.add_edges_from([(0, 6), (6, 7), (7, 8), (8, 9), (9, 1)])      # a ring fused on the bond 0-1
    graph.add_edges_from([(3, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 3)])
    systems = ps.ring_systems(graph)
    assert len({systems[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)}) == 1
    assert systems[11] != systems[0] and 16 not in systems


def test_atom_orbits_follow_the_symmetry_and_the_colours():
    star = nx.star_graph(3)                                              # centre 0, leaves 1-3
    assert [o.tolist() for o in ps.atom_orbits(star, [0, 1, 1, 1])] == [[0], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert [o.tolist() for o in ps.atom_orbits(star, [0, 1, 2, 1])][1] == [1, 3]


def test_pair_orbits_are_orbits_of_pairs_not_products_of_atom_orbits():
    # two identical arms joined at the centre: the pairs within an arm and across arms are different orbits
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])   # mirror: i <-> 6 - i
    colours = [0, 1, 2, 3, 2, 1, 0]
    assert ps.pair_orbit_closure(graph, colours, {(0, 4)}) == {(0, 4), (2, 6)}
    assert ps.pair_orbit_closure(graph, colours, {(0, 6)}) == {(0, 6)}
    # a pair joining the two mirror images is not an image of a pair within one arm
    assert (1, 5) not in ps.pair_orbit_closure(graph, colours, {(0, 4)})
    # a lower symmetry of the colours gives smaller orbits
    assert ps.pair_orbit_closure(graph, [0, 1, 2, 3, 2, 1, 4], {(0, 4)}) == {(0, 4)}


def test_syn_pentane_puts_the_end_carbons_in_contact():
    graph, xyz = pentane_skeleton()
    rotors = ps.find_rotors(graph, set())
    result = ps.select_pairs(graph, xyz, rotors, np.array([[0, 4]]), np.array([0.35]), [False] * 5, [0, 1, 2, 1, 0])
    assert result.written == [(0, 4)]
    # measured in conformers that respect the written pair, so it sits at the threshold
    assert result.closest[(0, 4)] == pytest.approx(0.8, abs=0.01)


def test_no_rotor_no_pair():
    graph, xyz = pentane_skeleton()
    result = ps.select_pairs(graph, xyz, [], np.array([[0, 4]]), np.array([0.35]), [False] * 5, [0] * 5)
    assert result.written == []


def test_a_pair_that_stays_far_is_left_out():
    graph, xyz = pentane_skeleton()
    rotors = ps.find_rotors(graph, set())
    # a sigma a tenth of the real one: no conformer brings the end carbons within 0.8 sigma
    result = ps.select_pairs(graph, xyz, rotors, np.array([[0, 4]]), np.array([0.035]), [False] * 5, [0, 1, 2, 1, 0])
    assert result.written == []


def test_a_pair_without_sigma_is_never_selected():
    graph, xyz = pentane_skeleton()
    rotors = ps.find_rotors(graph, set())
    result = ps.select_pairs(graph, xyz, rotors, np.array([[0, 4]]), np.array([np.nan]), [False] * 5, [0, 1, 2, 1, 0])
    assert result.written == []
    assert "sigma not known" in ps.format_commented_pair(0, 4, 1.0, np.nan, result.closest[(0, 4)], "1,5")
    assert "covered by the dihedral" in ps.format_commented_pair(0, 3, 1.0, 0.35, None, "1,4")


@pytest.fixture(scope="module")
def trans():
    features = Features.from_xyz(DATA / "DAB.xyz")
    graph = features.graph
    sep = dict(nx.all_pairs_shortest_path_length(graph))
    candidates = np.array([(i, j) for i in range(features.n) for j in range(i + 1, features.n) if sep[i][j] >= 4])
    atom_sigma = np.array([SIGMA[e] for e in features.el])
    sigma = np.sqrt(atom_sigma[candidates[:, 0]] * atom_sigma[candidates[:, 1]])
    colours = [sorted(set(features.el)).index(e) for e in features.el]
    rotors = ps.find_rotors(graph, double_bonds(features))
    hydrogen = [e == "H" for e in features.el]
    result = ps.select_pairs(graph, features.pos, rotors, candidates, sigma, hydrogen, colours)
    return features, candidates, colours, hydrogen, rotors, sigma, result


def test_trans_selects_about_two_hundred_pairs(trans):
    features, candidates, colours, hydrogen, rotors, sigma, result = trans
    assert 185 <= len(result.written) <= 225
    # every written pair is heavy-heavy or H...H and 1,5 or farther
    assert all(hydrogen[i] == hydrogen[j] for i, j in result.written)
    sep = dict(nx.all_pairs_shortest_path_length(features.graph))
    assert all(sep[i][j] >= 4 for i, j in result.written)


def test_trans_selection_is_closed_under_symmetry(trans):
    features, candidates, colours, hydrogen, rotors, sigma, result = trans
    written = set(result.written)
    assert ps.pair_orbit_closure(features.graph, colours, written) & {tuple(p) for p in candidates} == written


def test_trans_ring_blocks_are_complete(trans):
    features, candidates, colours, hydrogen, rotors, sigma, result = trans
    systems = ps.ring_systems(features.graph)
    written = set(result.written)
    blocks = {}
    for i, j in candidates:
        i, j = int(i), int(j)
        if i in systems and j in systems and systems[i] != systems[j] and not hydrogen[i] and not hydrogen[j]:
            blocks.setdefault((min(systems[i], systems[j]), max(systems[i], systems[j])), []).append((i, j) in written)
    assert blocks
    assert all(all(flags) or not any(flags) for flags in blocks.values())
    assert any(all(flags) for flags in blocks.values())


def test_the_selection_is_reproducible(trans):
    features, candidates, colours, hydrogen, rotors, sigma, result = trans
    again = ps.select_pairs(features.graph, features.pos, rotors, candidates, sigma, hydrogen, colours)
    assert again.written == result.written


def test_make_top_writes_the_selected_pairs_and_comments_the_rest(tmp_path):
    out = tmp_path / "DAB.top"
    subprocess.run([sys.executable, "-m", "SelIntCoords.make_top", "-m", str(DATA / "DAB.xyz"), "-o", str(out)], check=True)
    section = []
    on = False
    for line in out.read_text().split("\n"):
        if line.strip().startswith("["):
            on = line.split()[1] == "pairs"
            continue
        if on:
            section.append(line)
    # GROMACS stops reading a section at a blank line
    last = max(n for n, line in enumerate(section) if line.strip())
    assert all(line.strip() for line in section[:last + 1])
    written = [line for line in section if line.strip() and not line.lstrip().startswith(";")]
    commented = [line for line in section if line.lstrip().startswith(";") and "closest approach" in line]
    assert 185 <= len(written) <= 225
    assert len(commented) > 500
    with open(tmp_path / "DAB.csv") as f:
        counts = [int(row["Number"]) for row in csv.DictReader(f) if row["Type"].startswith("LJ")]
    assert sum(counts) == len(written) and counts[0] == 0                # 1,4 pairs are never written
    # the pairs are 2-function lines with the unscaled mixing-rule parameters
    fields = written[0].split()
    assert fields[2] == "2" and float(fields[6]) > 0 and float(fields[7]) > 0
