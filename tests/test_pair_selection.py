from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from SelIntCoords.pair_selection import append_commented_pairs, closest_approach, format_commented_pair

DATA = Path(__file__).resolve().parent / "data"


def chain():
    """Atoms 0-1-2-3 in a line along x; rotating about the 1-2 bond turns atom 3 about the axis."""
    coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 1.0, 0.0]])
    return nx.path_graph(4), coords


def test_closest_approach_is_the_cis_distance_of_a_dihedral():
    graph, coords = chain()
    # atoms 0 and 3 start cis (1.5 apart) and end trans (sqrt(1.5**2 + 2**2) = 2.5)
    r = closest_approach(graph, coords, [(1, 2)], np.array([[0, 3]]))
    assert r[0] == pytest.approx(1.5)


def test_pair_not_separated_by_the_rotor_is_nan_and_a_pair_across_two_rotors_takes_the_closest():
    graph, coords = chain()
    r = closest_approach(graph, coords, [(1, 2)], np.array([[0, 1], [2, 3], [0, 3]]))
    assert np.isnan(r[0]) and np.isnan(r[1]) and r[2] == pytest.approx(1.5)
    r = closest_approach(graph, coords, [(0, 1), (1, 2)], np.array([[0, 3]]))
    assert r[0] <= 1.5 + 1e-9


def test_a_bond_in_a_ring_is_not_a_rotor():
    ring = nx.cycle_graph(4)
    coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0], [0.0, 1.0, 0]])
    assert np.isnan(closest_approach(ring, coords, [(0, 1)], np.array([[0, 2]]))[0])


def test_commented_pairs_end_the_pairs_section(tmp_path):
    top = tmp_path / "t.top"
    top.write_text("[ pairs ]\n;  1 pairs\n  1  5   2  1.000  0.000  0.000   0.3000   0.100000\n\n[ exclusions ]\n  1 2\n")
    append_commented_pairs(str(top), [format_commented_pair(2, 8, 0.2, 0.35, 0.93, "1,6"),
                                      format_commented_pair(3, 9, 0.2, 0.35, None, "1,7")], ["; why these"])
    text = top.read_text()
    pairs = text.split("[ pairs ]")[1].split("[ exclusions ]")[0]
    assert pairs.index("  1  5") < pairs.index("; pairs left out") < pairs.index(";   3    9")
    # no blank line between the pairs and the commented ones, and they stay in the order given
    lines = pairs.splitlines()
    assert lines[1:4] == [";  1 pairs", "; why these", lines[3]]
    first = next(n for n, l in enumerate(lines) if l.startswith("  1  5"))
    assert lines[first + 1].startswith("; pairs left out") and lines[first + 2].startswith(";   3")
    assert lines[first + 3].startswith(";   4") and lines[first + 4] == ""
    assert "r_min/sigma 0.93" in pairs and "no rotor between them" in pairs
    assert text.endswith("  1 2\n")


def test_zr1_keeps_the_chalcogen_contacts_across_the_ring_links(tmp_path, monkeypatch):
    """The 8 pairs of the hand-made ZR1 topology (S...S and S...H across the ring links) are written, everything else is commented."""
    import shutil
    import sys
    from SelIntCoords.make_top import main

    try:
        from SelIntCoords.ffdb import find_gmx_top
        gmx = find_gmx_top()
    except FileNotFoundError:
        pytest.skip("GROMACS topology database not found")
    shutil.copy(DATA / "ZR1.xyz", tmp_path / "ZR1.xyz")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_top", "-m", "ZR1.xyz", "--gmx-top", str(gmx)])
    main()
    text = (tmp_path / "ZR1.top").read_text()
    pairs = text.split("[ pairs ]")[1].split("[ exclusions ]")[0]
    active = {tuple(int(x) for x in l.split()[:2]) for l in pairs.splitlines()
              if l.strip() and not l.lstrip().startswith(";")}
    hand = {(7, 20), (13, 25), (12, 25), (18, 20), (7, 68), (13, 70), (12, 70), (18, 68)}
    assert hand <= active
    assert len(active) < 60
    head = pairs.splitlines()
    assert head[1].startswith(";") and "pairs" in head[1] and "rotatable bond" in head[2] and "0.80 sigma" in head[3]
    # the pairs left out are listed, commented out, with the closest approach that excluded them
    commented = [l for l in pairs.splitlines() if l.startswith(";") and "r_min/sigma" in l]
    assert len(commented) > 1000
    # the counts of the csv are those written
    import csv
    rows = list(csv.reader((tmp_path / "ZR1.csv").read_text().splitlines()))[1:]
    assert sum(int(r[1]) for r in rows if r[0].startswith("LJ")) == len(active)
