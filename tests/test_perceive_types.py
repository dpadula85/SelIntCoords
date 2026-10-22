"""perceive_types: force-field atom types from a bare geometry.

The reference is the set of types the example topologies in `examples/` were
built with by hand: the perceived type of every atom must have the same mass
and Lennard-Jones parameters (the number can differ where several types share
them, e.g. opls_135-139 for alkane carbons). The GROMACS database is not part
of the repository: tests that need it are skipped without it (set $GMXLIB or
install GROMACS).
"""
import re
from pathlib import Path

import pytest

from SelIntCoords import sel_intcoords as si
from SelIntCoords.ffdb import ForceField, find_gmx_top
from SelIntCoords.learned_types import LearnedTyper
from SelIntCoords.make_top import add_terms, geom_avg_mixing
from SelIntCoords.mol_features import Features
from SelIntCoords.perceive_types import RULESETS, class_names, perceive, perceive_to_top

NAMES = ["BTBT", "DNBDT", "PN", "DPP", "PNDI2OD", "D18", "PBDTTT-O"]


@pytest.fixture(scope="module")
def gmx_top():
    """The OPLS-AA files shipped with the package."""
    return find_gmx_top()


@pytest.fixture(scope="module")
def oplsaa(gmx_top):
    return ForceField("oplsaa", gmx_top)


@pytest.fixture(scope="module")
def learned(oplsaa):
    return LearnedTyper(oplsaa)


def perceive_opls(f, ff, learned):
    rules, refuse = RULESETS["oplsaa"]
    return perceive(f, ff, rules, learned, refuse)


def read_top(path):
    """Atomtype (mass, sigma, epsilon) by name, and atomtype name by atom (0-based)."""
    sec, types, atoms = None, {}, {}
    for line in Path(path).read_text().splitlines():
        s = line.split(";")[0].strip()
        m = re.match(r"\[\s*(\w+)\s*\]", s)
        if m:
            sec = m.group(1)
            continue
        tok = s.split()
        if sec == "atomtypes" and len(tok) >= 6:
            types[tok[0]] = (float(tok[1]), float(tok[4]), float(tok[5]))
        elif sec == "atoms" and tok and tok[0].isdigit():
            atoms[int(tok[0]) - 1] = tok[1]
    return types, atoms


# ----------------------------------------------------------------------
# force-field database
# ----------------------------------------------------------------------
def test_ffdb_oplsaa_parameters(oplsaa):
    t = oplsaa.atomtypes["opls_633"]
    assert (t.element, t.mass, t.sigma, t.epsilon) == ("S", 32.06, 0.355, 1.046)
    assert "thiazole" in t.description
    assert oplsaa.comb_rule == 3
    assert len(oplsaa.residues) > 40


def test_ffdb_gromos_c6_c12_converted():
    """Combination rule 1 stores C6, C12: sigma/epsilon must come out in nm and kJ/mol."""
    try:
        top = find_gmx_top(None, "gromos54a7")
    except FileNotFoundError:
        pytest.skip("GROMACS topology database not found")
    ff = ForceField("gromos54a7", top)
    assert ff.comb_rule == 1
    t = ff.atomtypes["OA"]  # hydroxyl oxygen: sigma ~ 0.30 nm, epsilon ~ 0.8 kJ/mol
    assert 0.25 < t.sigma < 0.35
    assert 0.3 < t.epsilon < 1.5


def test_ffdb_unknown_force_field(gmx_top):
    with pytest.raises(FileNotFoundError, match="available"):
        ForceField("nosuchff", gmx_top)


# ----------------------------------------------------------------------
# curated rules against the hand-built example topologies
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", NAMES)
def test_examples_lj_and_mass_match_the_hand_assigned_types(name, examples, oplsaa, learned):
    f = Features.from_xyz(str(examples / name / f"{name}.xyz"))
    types, atoms = read_top(examples / name / f"{name}.top")
    assign = perceive_opls(f, oplsaa, learned)
    assert len(assign) == f.n == len(atoms)

    wrong = []
    for a in assign:
        mass, sigma, eps = types[atoms[a.index]]
        t = oplsaa.atomtypes[a.type]
        if not (abs(t.mass - mass) < 0.02 and abs(t.sigma - sigma) < 1e-6 and abs(t.epsilon - eps) < 1e-5):
            wrong.append((a.index + 1, atoms[a.index], a.type))

    # PBDTTT-O: the furan oxygen was typed by hand with epsilon = 0.5876, a
    # transposition of opls_566's 0.58576 kJ/mol; everything else agrees
    expected_wrong = [w for w in wrong if w[1] == "OF"] if name == "PBDTTT-O" else []
    assert wrong == expected_wrong
    if name == "PBDTTT-O":
        assert len(expected_wrong) == 2 and all(w[2] == "opls_566" for w in expected_wrong)


@pytest.mark.parametrize("name", NAMES)
def test_examples_are_typed_by_rules_without_atoms_to_check(name, examples, oplsaa, learned):
    f = Features.from_xyz(str(examples / name / f"{name}.xyz"))
    assign = perceive_opls(f, oplsaa, learned)
    assert all(a.source == "rule" for a in assign)
    flagged = [a for a in assign if a.flags or a.alternatives]
    # only the ester alkoxy H of PBDTTT-O (half-epsilon opls_469 against the
    # generic alkane H of the residue templates) is a real ambiguity
    assert [(a.index + 1, a.type) for a in flagged] == (
        [(35, "opls_469"), (36, "opls_469")] if name == "PBDTTT-O" else [])


# (element, neighbour elements) -> expected type, per molecule; from the
# descriptions in oplsaa.ff/atomtypes.atp
CASES = {
    "Cc1ccccc1": {("C", "CCH"): "opls_145", ("C", "CCC"): "opls_145", ("C", "HHHC"): "opls_135",
                  ("H", "C"): {"opls_146", "opls_140"}},
    "c1ccncc1": {("N", "CC"): "opls_520"},
    "c1cc[nH]c1": {("N", "CCH"): "opls_542", ("C", "CHN"): "opls_543", ("C", "CCH"): "opls_544",
                   ("H", "N"): "opls_545"},
    "c1ccsc1": {("S", "CC"): "opls_633", ("C", "CHS"): "opls_145"},
    "c1ccoc1": {("O", "CC"): "opls_566"},
    "CC(N)=O": {("C", "CNO"): "opls_235", ("O", "C"): "opls_236", ("N", "CHH"): "opls_237",
                ("H", "N"): "opls_240"},
    "CNC(C)=O": {("N", "CCH"): "opls_238", ("H", "N"): "opls_241"},
    "CN(C)C(C)=O": {("N", "CCC"): "opls_239"},
    "CC(O)=O": {("C", "COO"): "opls_267", ("O", "C"): "opls_269", ("O", "CH"): "opls_268",
                ("H", "O"): "opls_270"},
    "COC(C)=O": {("C", "COO"): "opls_465", ("O", "C"): "opls_466", ("O", "CC"): "opls_467"},
    "CC(C)=O": {("C", "CCO"): "opls_280", ("O", "C"): "opls_281"},
    "CC=O": {("C", "CHO"): "opls_277", ("H", "C"): {"opls_140", "opls_279"}},
    "CCO": {("O", "CH"): "opls_154", ("H", "O"): "opls_155"},
    "Oc1ccccc1": {("O", "CH"): "opls_167", ("H", "O"): "opls_168", ("C", "CCO"): "opls_166"},
    "CCOCC": {("O", "CC"): "opls_180"},
    "COc1ccccc1": {("O", "CC"): "opls_179"},
    "N#Cc1ccccc1": {("N", "C"): "opls_753", ("C", "CN"): "opls_754"},
    "[O-][N+](=O)c1ccccc1": {("N", "COO"): "opls_767", ("O", "N"): "opls_761"},
    "Fc1ccccc1": {("F", "C"): "opls_719", ("C", "CCF"): "opls_718"},
    "Clc1ccccc1": {("Cl", "C"): "opls_264"},
    "FC(F)(F)c1ccccc1": {("C", "CFFF"): "opls_961", ("F", "C"): "opls_965"},
    "CCS": {("S", "CH"): "opls_200", ("H", "S"): "opls_204"},
    "CSC": {("S", "CC"): "opls_202"},
    "Nc1ccccc1": {("N", "CHH"): "opls_900"},
    "CC=C": {("C", "CCH"): "opls_142", ("C", "CHH"): "opls_143", ("H", "C"): {"opls_140", "opls_144"}},
    "CC#C": {("C", "CH"): "opls_925", ("C", "CC"): "opls_931"},
    "CS(C)=O": {("S", "CCO"): "opls_496", ("O", "S"): "opls_497"},
    "CS(C)(=O)=O": {("S", "CCOO"): "opls_493", ("O", "S"): "opls_494"},
    "CC([O-])=O": {("C", "COO"): "opls_271", ("O", "C"): "opls_272"},
    "NC(=O)c1ccccc1": {("C", "CNO"): "opls_234"},
    "c1ccc2scnc2c1": {("S", "CC"): "opls_633", ("N", "CC"): "opls_557"},
    "c1ccccc1N=Nc1ccccc1": {("N", "CN"): "opls_520"},
}


@pytest.fixture(scope="module")
def smiles_geometry(tmp_path_factory):
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    folder = tmp_path_factory.mktemp("smiles")

    def build(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, randomSeed=7)
        AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
        el = [a.GetSymbol() for a in m.GetAtoms()]
        pos = m.GetConformer().GetPositions()
        path = folder / ("m%d.xyz" % abs(hash(smiles)))
        path.write_text("%d\n\n" % len(el) + "\n".join(
            "%s %.5f %.5f %.5f" % (e, *p) for e, p in zip(el, pos)) + "\n")
        return str(path)
    return build


@pytest.mark.parametrize("smiles", CASES)
def test_functional_groups(smiles, smiles_geometry, oplsaa, learned):
    f = Features.from_xyz(smiles_geometry(smiles))
    assign = perceive_opls(f, oplsaa, learned)
    found = {}
    for a in assign:
        found.setdefault((a.element, "".join(f.nbr_el(a.index))), set()).add(a.type)
    for key, expected in CASES[smiles].items():
        expected = expected if isinstance(expected, set) else {expected}
        # neighbour strings are sorted element symbols
        key = (key[0], "".join(sorted(re.findall(r"[A-Z][a-z]?", key[1]))))
        assert found[key] == expected, (smiles, key, found)


@pytest.mark.parametrize("smiles, fragment", [
    ("C[n+]1ccccc1", "pyridinium"),
    ("CP(C)C", "no type for atom"),
    ("CC(N)=S", "no type for atom"),
])
def test_unsupported_chemistry_raises(smiles, fragment, smiles_geometry, oplsaa, learned):
    f = Features.from_xyz(smiles_geometry(smiles))
    with pytest.raises(ValueError, match=fragment):
        perceive_opls(f, oplsaa, learned)


def test_ammonium_is_learned_and_reported(smiles_geometry, oplsaa, learned):
    f = Features.from_xyz(smiles_geometry("C[NH3+]"))
    assign = perceive_opls(f, oplsaa, learned)
    n = [a for a in assign if a.element == "N"][0]
    assert n.type == "opls_287" and n.source == "learned" and n.flags


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name, aromatic_rings, all_rings", [
    ("BTBT", 4, 4), ("PN", 5, 5), ("DPP", 4, 6), ("PNDI2OD", 4, 6), ("D18", 11, 11),
])
def test_aromatic_rings(name, aromatic_rings, all_rings, examples):
    """Lactam and imide rings (carbonyl carbon) are not aromatic; thiophene, furan and thiadiazole are."""
    f = Features.from_xyz(str(examples / name / f"{name}.xyz"))
    assert (len(f.aromatic_rings), len(f.rings)) == (aromatic_rings, all_rings)


# ----------------------------------------------------------------------
# learned engine
# ----------------------------------------------------------------------
def test_learned_leave_one_residue_out(oplsaa):
    """Types of a residue predicted from the others (variants of it left out too)."""
    from SelIntCoords.learned_types import residue_features

    def group(n):
        return n[:3] if len(n) > 3 else n

    typers, total, same_lj = {}, 0, 0
    for res in oplsaa.residues:
        g = group(res.name)
        if g not in typers:
            typers[g] = LearnedTyper(oplsaa, skip=[r.name for r in oplsaa.residues if group(r.name) == g])
        f, atoms = residue_features(res, oplsaa)
        pred = typers[g].predict(f, list(range(len(atoms))))
        for k, n in enumerate(atoms):
            total += 1
            p = pred[k]
            same_lj += p is not None and oplsaa.lj_key(p.type) == oplsaa.lj_key(res.atoms[n])
    assert same_lj / total > 0.9


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def test_class_names_split_environments(examples, oplsaa, learned):
    f = Features.from_xyz(str(examples / "BTBT" / "BTBT.xyz"))
    assign = perceive_opls(f, oplsaa, learned)
    names = class_names(assign, oplsaa, f)
    # three aromatic C environments (CCH, CCC, CCS), one S, one H on aromatic C
    assert {"CA1", "CA2", "CA3", "S", "HA"} <= set(names)
    # one name never covers two force-field types
    by_name = {}
    for n, a in zip(names, assign):
        by_name.setdefault(n, set()).add(a.type)
    assert all(len(t) == 1 for t in by_name.values())


def test_perceive_to_top_writes_top_and_report(examples, gmx_top, tmp_path):
    out = tmp_path / "BTBT_types.top"
    top, assign = perceive_to_top(examples / "BTBT" / "BTBT.xyz", "oplsaa", gmx_top, out, class_name=False)
    assert top == out and out.with_suffix(".types").exists()
    types, atoms = read_top(out)
    assert len(atoms) == len(assign) == 36
    assert types["opls_633"][1:] == pytest.approx((0.355, 1.046))
    assert "Typed by analogy" in out.with_suffix(".types").read_text()


@pytest.fixture
def no_symmetry(monkeypatch):
    monkeypatch.setattr(si, "get_equivalent_atoms", lambda u: {})


@pytest.mark.parametrize("name", ["BTBT", "DPP", "PNDI2OD"])
def test_perceived_top_gives_the_same_terms_as_the_hand_written_one(
        name, examples, gmx_top, no_symmetry, tmp_path):
    """make_top fed with the perceived topology writes the same bonded terms, pairs and exclusions."""

    def terms(top_path):
        data = si.list_intcoords(str(examples / name / f"{name}.xyz"))
        bds, angs, stiff, imps, flex, LJs, excls, rings, eq = data
        top = add_terms(str(top_path), bds, angs, stiff, imps, flex, LJs, excls, mixing=geom_avg_mixing)
        out = tmp_path / "terms.top"
        top.write(str(out))
        text = out.read_text()
        # [ bonds ] up to [ system ]: independent of atomtype names, charges and molecule name
        body = text[text.index("[ bonds ]"):text.index("[ system ]")]
        return "\n".join(l.split(";")[0].strip() for l in body.splitlines())

    perceived, _ = perceive_to_top(examples / name / f"{name}.xyz", "oplsaa", gmx_top,
                                   tmp_path / "types.top")
    assert terms(perceived) == terms(examples / name / f"{name}.top")


@pytest.mark.parametrize("name", ["BTBT", "DPP"])
def test_make_top_from_xyz_alone_gives_the_full_topology(name, examples, gmx_top, tmp_path, monkeypatch):
    """`make_top -m x.xyz`: no starting topology, one step, no intermediate file left behind."""
    import shutil
    import sys
    from SelIntCoords.make_top import main

    shutil.copy(examples / name / f"{name}.xyz", tmp_path / f"{name}.xyz")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_top", "-m", f"{name}.xyz", "--gmx-top", str(gmx_top)])
    main()

    assert sorted(p.name for p in tmp_path.iterdir() if p.name != "timer.dat") == sorted([
        f"{name}.xyz", f"{name}.top", f"{name}.csv", f"{name}_deps.dat", f"{name}.types"])

    def terms(path):
        """(section, atoms, funct) of every bonded line, whatever the order and the parameters."""
        found, section = set(), None
        for line in Path(path).read_text().splitlines():
            tok = line.split(";")[0].split()
            if line.strip().startswith("["):
                section = line.split()[1]
            elif tok and section in ("bonds", "angles", "dihedrals"):
                n = {"bonds": 2, "angles": 3, "dihedrals": 4}[section]
                found.add((section, tuple(int(x) for x in tok[:n]), int(tok[n])))
        return found

    # the terms of make_top fed with the hand-written topology: same bonds,
    # angles and stiff terms, every flexible dihedral kept is one of the
    # reference ones or moved to the stiff ones (double bond)
    data = si.list_intcoords(str(examples / name / f"{name}.xyz"))
    bds, angs, stiff, imps, flex, LJs, excls, rings, eq = data
    ref = add_terms(str(examples / name / f"{name}.top"), bds, angs, stiff, imps, flex, LJs, excls,
                    mixing=geom_avg_mixing)
    ref.write(str(tmp_path / "ref.top"))
    new_terms, ref_terms = terms(tmp_path / f"{name}.top"), terms(tmp_path / "ref.top")
    assert {t for t in ref_terms if t[0] != "dihedrals" or t[2] == 2} <= new_terms
    assert {t for t in new_terms if t[0] != "dihedrals"} == {t for t in ref_terms if t[0] != "dihedrals"}
    assert {t for t in new_terms if t[0] == "dihedrals" and t[2] == 1} <= {t for t in ref_terms if t[2] == 1}
    assert "; " in Path(tmp_path / f"{name}.top").read_text().split("[ bonds ]")[1].splitlines()[2]


def test_make_top_output_name_follows_dash_o(examples, gmx_top, tmp_path, monkeypatch):
    import shutil
    import sys
    from SelIntCoords.make_top import main

    shutil.copy(examples / "PN" / "PN.xyz", tmp_path / "PN.xyz")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_top", "-m", "PN.xyz", "-o", "pn_full.top", "--gmx-top", str(gmx_top)])
    main()
    assert {"pn_full.top", "pn_full.csv", "pn_full_deps.dat", "pn_full.types"} <= {p.name for p in tmp_path.iterdir()}


def test_make_top_with_a_topology_is_unchanged(examples, tmp_path, monkeypatch):
    """-p still works as before and writes no report."""
    import shutil
    import sys
    from SelIntCoords.make_top import main

    for ext in ("xyz", "top"):
        shutil.copy(examples / "PN" / f"PN.{ext}", tmp_path / f"PN.{ext}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_top", "-p", "PN.top", "-m", "PN.xyz"])
    main()
    names = {p.name for p in tmp_path.iterdir()}
    assert {"PN.new.top", "PN.csv", "PN_deps.dat"} <= names and "PN.types" not in names


# ----------------------------------------------------------------------
# elements without OPLS-AA types, rhodanine, rotors

def atomtype_lines(path):
    """Atomtype rows (name, mass, sigma, epsilon as written) of a .top."""
    rows, sec = {}, None
    for line in Path(path).read_text().splitlines():
        s = line.split(";")[0].strip()
        m = re.match(r"\[\s*(\w+)\s*\]", s)
        if m:
            sec = m.group(1)
        elif sec == "atomtypes" and s:
            tok = s.split()
            rows[tok[0]] = tok
    return rows


def with_element(path, old, new, folder):
    """Copy of an .xyz with one element relabelled (a thiophene geometry stands in for selenophene)."""
    lines = Path(path).read_text().splitlines()
    out = [(new + l[len(old):]) if l.startswith(old + " ") else l for l in lines]
    target = Path(folder) / ("%s_%s" % (new, Path(path).name))
    target.write_text("\n".join(out) + "\n")
    return str(target)


@pytest.mark.parametrize("smiles, old, element, mass, sigma, epsilon", [
    ("c1ccsc1", "S", "Se", 78.971, 0.37741, 1.21754),
    ("C[Si](C)(C)C", "Si", "Si", 28.0855, 0.4184, 0.295),
])
def test_selenium_and_silicon_take_the_literature_values(smiles, old, element, mass, sigma, epsilon,
                                                         smiles_geometry, gmx_top, tmp_path):
    xyz = with_element(smiles_geometry(smiles), old, element, tmp_path)
    top, assign = perceive_to_top(xyz, "oplsaa", gmx_top, tmp_path / "t.top")
    types = atomtype_lines(top)
    atom = [a for a in assign if a.element == element][0]
    assert atom.source == "extra"
    tok = types[atom.type]
    assert (float(tok[1]), float(tok[4]), float(tok[5])) == pytest.approx((mass, sigma, epsilon))


def test_unknown_element_is_marked_for_the_user_to_fill_in(smiles_geometry, examples, gmx_top, tmp_path, monkeypatch):
    import sys
    from SelIntCoords.make_top import main

    xyz = with_element(smiles_geometry("C[Si](C)(C)C"), "Si", "B", tmp_path)
    (tmp_path / "b.xyz").write_text(Path(xyz).read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_top", "-m", "b.xyz", "--gmx-top", str(gmx_top)])
    main()
    final = (tmp_path / "b.top").read_text()
    # the two parameters of the boron atomtype read FILL_ME, nothing else does
    b_line = [l for l in final.split("[ atomtypes ]")[1].split("[ moleculetype ]")[0].splitlines() if "FILL_ME" in l]
    assert len(b_line) == 1 and b_line[0].count("FILL_ME") == 2
    assert sum("FILL_ME" in l for l in final.splitlines() if not l.startswith(";")) == 1
    # the topology to fill in and give back with -p is kept
    assert (tmp_path / "b_types.top").exists()


def test_rhodanine_ring_follows_the_zr1_conventions(smiles_geometry, oplsaa, learned):
    f = Features.from_xyz(smiles_geometry("S=C1SC(=C)C(=O)N1C"))
    assign = {a.index: a for a in perceive_opls(f, oplsaa, learned)}
    ring = [i for i in f.rings[0]]
    types = {(f.el[i], a.type) for i, a in assign.items() if i in ring}
    thione = [i for i in range(f.n) if f.terminal(i) and f.el[i] == "S"][0]
    assert assign[thione].type == "opls_633"
    assert {t for e, t in types if e == "S"} == {"opls_633"}
    assert {t for e, t in types if e == "N"} == {"opls_239"}
    # every ring carbon but the C=O one is typed as benzene C
    assert all(assign[i].type == "opls_145" for i in ring if f.el[i] == "C" and not f.is_carbonyl(i) or f.el[i] == "C")


def test_make_top_writes_headers_and_literature_methyls(smiles_geometry, gmx_top, tmp_path, monkeypatch):
    import sys
    from SelIntCoords.make_top import main

    xyz = Path(smiles_geometry("Cc1cccs1"))
    (tmp_path / "mt.xyz").write_text(xyz.read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_top", "-m", "mt.xyz", "--gmx-top", str(gmx_top)])
    main()
    text = (tmp_path / "mt.top").read_text()
    assert "; thiophene" in text and "; methyls" in text
    dihedrals = text.split("[ dihedrals ]")[1].split("[ pairs ]")[0]
    methyl_rows = dihedrals.split("; methyls")[-1].split("\n;")[0].strip().splitlines()
    # three H-C-C-S dihedrals, funct 1, delta 0, k 0.651, n 3
    assert len(methyl_rows) == 3
    for row in methyl_rows:
        tok = row.split(";")[0].split()
        assert tok[4:] == ["1", "0.0000", "0.6510", "3"]


def test_default_names_are_classes_and_element_numbers(examples, gmx_top, tmp_path, monkeypatch):
    """Atomtypes by environment (CA1, S, HA...) and atoms named element + number; the opls type stays in a comment."""
    import shutil
    import sys
    from SelIntCoords.make_top import main

    shutil.copy(examples / "BTBT" / "BTBT.xyz", tmp_path / "BTBT.xyz")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_top", "-m", "BTBT.xyz", "--gmx-top", str(gmx_top)])
    main()
    text = (tmp_path / "BTBT.top").read_text()
    assert "opls_" not in text.split("[ moleculetype ]")[1]
    atoms = text.split("[ atoms ]")[1].split("[ bonds ]")[0].splitlines()
    rows = [l.split() for l in atoms if l.strip() and not l.startswith(";")]
    assert [r[4] for r in rows][:3] == ["S1", "C2", "C3"] and len({r[4] for r in rows}) == 36
    assert all(r[1] != r[4] for r in rows)
    assert "; opls_145" in text.split("[ atomtypes ]")[1].split("[ moleculetype ]")[0]


def test_oplsaa_files_are_shipped_with_the_package_and_found_by_default():
    from SelIntCoords.ffdb import BUNDLED_TOP
    assert find_gmx_top(None, "oplsaa") == BUNDLED_TOP
    ff = ForceField("oplsaa")
    assert ff.top == BUNDLED_TOP and ff.atomtypes["opls_145"].sigma == pytest.approx(0.355)
    with pytest.raises(FileNotFoundError, match="nosuchff"):
        find_gmx_top(None, "nosuchff")
