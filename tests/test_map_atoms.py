"""map_atoms / renumber: recover a shuffled atom order and apply it to a topology."""
import numpy as np
import MDAnalysis as mda
import pytest

from SelIntCoords.map_atoms import AtomMapper
from SelIntCoords.renumber import read_map, renumber_molecule
from SelIntCoords.top import TOP


@pytest.fixture
def shuffled(tmp_path, examples):
    """BTBT.xyz with its atoms in a random order."""
    lines = (examples / "BTBT" / "BTBT.xyz").read_text().splitlines()
    atoms = lines[2:]
    perm = np.random.default_rng(0).permutation(len(atoms))
    out = tmp_path / "shuffled.xyz"
    out.write_text("\n".join([lines[0], "shuffled"] + [atoms[i] for i in perm]) + "\n")
    return out


def test_mapper_recovers_the_reference_geometry(examples, shuffled):
    mapper = AtomMapper(examples / "BTBT" / "BTBT.xyz", shuffled)
    result = mapper.run()
    assert result.rmsd < 1e-4
    ref = mda.Universe(str(examples / "BTBT" / "BTBT.xyz"))
    assert np.allclose(mapper.reordered_atomgroup().positions, ref.atoms.positions, atol=1e-3)
    assert list(mapper.reordered_atomgroup().names) == list(ref.atoms.names)


def test_mapping_is_a_bijection_and_round_trips_through_a_file(examples, shuffled, tmp_path):
    result = AtomMapper(examples / "BTBT" / "BTBT.xyz", shuffled).run()
    pairs = result.as_array(one_based=True)
    assert sorted(pairs[:, 0]) == sorted(pairs[:, 1]) == list(range(1, len(pairs) + 1))
    result.write(tmp_path / "map.txt")
    assert read_map(tmp_path / "map.txt") == {int(t): int(r) for r, t in pairs}  # target -> reference


def test_non_isomorphic_structures_are_refused(examples):
    with pytest.raises((RuntimeError, ValueError)):
        AtomMapper(examples / "BTBT" / "BTBT.xyz", examples / "PN" / "PN.xyz").run()


def test_renumber_applies_a_bijection(examples):
    mol = TOP(str(examples / "BTBT" / "BTBT.top")).molecules[0]
    n = len(mol.atoms)
    renummap = {i: n + 1 - i for i in range(1, n + 1)}  # reverse the numbering
    first = mol.atoms[0]
    applied = renumber_molecule(mol, renummap)
    assert first.number == n and mol.atoms[-1] is first  # atom list is re-sorted
    assert len(applied) == n


def test_renumber_refuses_a_partial_map_unless_allowed(examples):
    mol = TOP(str(examples / "BTBT" / "BTBT.top")).molecules[0]
    with pytest.raises(ValueError):
        renumber_molecule(mol, {1: 2})


def test_renumber_refuses_duplicate_targets(examples):
    mol = TOP(str(examples / "BTBT" / "BTBT.top")).molecules[0]
    renummap = {a.number: 1 for a in mol.atoms}
    with pytest.raises(ValueError):
        renumber_molecule(mol, renummap)
