#!/usr/bin/env python

'''
Object-oriented XYZ atom mapper.

Maps atoms between two XYZ structures (e.g. an experimental crystal
geometry and a force-field-optimised geometry with a different atom
ordering) by finding the bond-graph isomorphism, matched on element
identity, that minimises the Kabsch-aligned RMSD between the two sets
of coordinates.

The resulting map can be written to file and fed to `renumber_top`
(:mod:`SelIntCoords.renumber`) to renumber a GROMACS topology so that
it follows the reference numbering.
'''

import argparse as arg
import logging
from pathlib import Path

import numpy as np
import networkx as nx
import MDAnalysis as mda
from networkx.algorithms.isomorphism import GraphMatcher

log = logging.getLogger("map_atoms")


class AtomMapping(object):
    '''Class to hold the result of mapping reference atoms onto target atoms.

    :attr:`mapping` is a list such that ``mapping[i]`` is the 0-based
    index of the target atom corresponding to reference atom ``i``.
    '''

    def __init__(self, mapping, rmsd, n_isomorphisms):
        self.mapping = mapping
        self.rmsd = rmsd
        self.n_isomorphisms = n_isomorphisms

    def as_dict(self, one_based=False):
        '''Return the mapping as a dictionary.

        Parameters
        ----------
        one_based: bool.
            Whether indices should be 1-based (GROMACS/Joyce convention)
            rather than 0-based.

        Returns
        -------
        mapping: dict.
            {reference_index: target_index}.
        '''

        offset = 1 if one_based else 0
        return {i + offset: j + offset for i, j in enumerate(self.mapping)}

    def as_array(self, one_based=False):
        '''Return the mapping as an array.

        Parameters
        ----------
        one_based: bool.
            Whether indices should be 1-based (GROMACS/Joyce convention)
            rather than 0-based.

        Returns
        -------
        mapping: np.ndarray, (N, 2).
            Columns are [reference_index, target_index].
        '''

        offset = 1 if one_based else 0
        ref_idx = np.arange(len(self.mapping)) + offset
        tgt_idx = np.asarray(self.mapping) + offset
        return np.column_stack([ref_idx, tgt_idx])

    def write(self, filename, one_based=True):
        '''Write the mapping to a plain-text two-column file.

        Parameters
        ----------
        filename: str.
            Output file name.
        one_based: bool.
            Whether indices should be 1-based (GROMACS/Joyce convention,
            default) rather than 0-based.
        '''

        np.savetxt(
            filename,
            self.as_array(one_based=one_based),
            fmt="%6d",
            header="reference   target",
        )


class AtomMapper(object):
    '''Class to compute a graph-isomorphism-based atom mapping between two
    XYZ structures.
    '''

    def __init__(self, reference, target):
        '''
        Parameters
        ----------
        reference: str.
            XYZ file defining the desired (e.g. crystallographic) atom
            numbering.
        target: str.
            XYZ file to be mapped onto the reference numbering (e.g. the
            geometry a force field was generated on).
        '''

        self.reference = Path(reference)
        self.target = Path(target)
        self.ref_universe = mda.Universe(str(self.reference), to_guess=["bonds", "types"])
        self.tgt_universe = mda.Universe(str(self.target), to_guess=["bonds", "types"])

        if self.ref_universe.atoms.n_atoms != self.tgt_universe.atoms.n_atoms:
            raise ValueError(
                "Atom count mismatch: reference has %d atoms, target has %d atoms."
                % (self.ref_universe.atoms.n_atoms, self.tgt_universe.atoms.n_atoms)
            )

        self._result = None

    @property
    def result(self):
        '''The most recently computed :class:`AtomMapping`. Call :meth:`map` first.'''

        if self._result is None:
            raise RuntimeError("No mapping computed yet; call map() first.")
        return self._result

    @staticmethod
    def _element(atom):
        element = getattr(atom, "element", "") or atom.name[0]
        return element.strip()

    @classmethod
    def _build_graph(cls, universe):
        graph = nx.Graph()
        for i, atom in enumerate(universe.atoms):
            graph.add_node(i, element=cls._element(atom))
        graph.add_edges_from(universe.bonds.to_indices())
        return graph

    @staticmethod
    def _kabsch_rmsd(p, q):
        '''RMSD between P and Q after optimal rigid-body superposition.'''

        p_c = p - p.mean(axis=0)
        q_c = q - q.mean(axis=0)
        v, _, wt = np.linalg.svd(p_c.T @ q_c)
        sign = np.sign(np.linalg.det(v @ wt))
        rotation = v @ np.diag([1.0, 1.0, sign]) @ wt
        return float(np.sqrt(np.mean(np.sum((p_c @ rotation - q_c) ** 2, axis=1))))

    def map(self, log_every=10000):
        '''Find the isomorphism that minimises RMSD; cache and return the result.

        Parameters
        ----------
        log_every: int.
            Log progress every this many isomorphisms tested (set to 0 to
            disable). Symmetric molecules (fused aromatics, alkyl chains,
            phenyl substituents, ...) can have very many automorphisms, so
            this run can take a while - progress logging helps track it.

        Returns
        -------
        result: AtomMapping.
        '''

        graph_ref = self._build_graph(self.ref_universe)
        graph_tgt = self._build_graph(self.tgt_universe)

        matcher = GraphMatcher(
            graph_ref, graph_tgt, node_match=lambda a, b: a["element"] == b["element"]
        )
        if not matcher.is_isomorphic():
            raise RuntimeError("Reference and target molecular graphs are not isomorphic.")

        ref_positions = self.ref_universe.atoms.positions.copy()
        tgt_positions = self.tgt_universe.atoms.positions

        best_rmsd = np.inf
        best_mapping = None
        n_isomorphisms = 0

        for iso in matcher.isomorphisms_iter():
            n_isomorphisms += 1
            order = [iso[i] for i in range(len(iso))]
            rmsd = self._kabsch_rmsd(ref_positions, tgt_positions[order])
            if rmsd < best_rmsd:
                best_rmsd, best_mapping = rmsd, order
            if log_every and n_isomorphisms % log_every == 0:
                log.info(
                    "...%d isomorphisms tested so far (best RMSD so far: %.6f A)",
                    n_isomorphisms, best_rmsd,
                )

        log.info("Tested %d isomorphism(s); best RMSD = %.6f A", n_isomorphisms, best_rmsd)

        self._result = AtomMapping(best_mapping, best_rmsd, n_isomorphisms)
        return self._result

    def reordered_atomgroup(self, mapping=None):
        '''Return the target AtomGroup reordered to match the reference atom order.'''

        mapping = mapping if mapping is not None else self.result.mapping
        return self.tgt_universe.atoms[mapping]

    def write_reordered(self, filename, mapping=None):
        '''Write the reordered target structure (format inferred from extension).'''

        self.reordered_atomgroup(mapping).write(str(filename))


def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(
        description="Map atoms between two XYZ structures via graph isomorphism, "
                     "e.g. to align a force-field geometry's atom numbering with "
                     "an experimental crystal structure.",
        formatter_class=arg.ArgumentDefaultsHelpFormatter,
    )

    inp = parser.add_argument_group("Input Data")
    inp.add_argument(
        "-r", "--ref", type=str, required=True, dest="RefFile",
        help="Reference structure (.xyz), defines the desired atom numbering "
             "(e.g. the experimental crystal structure).",
    )
    inp.add_argument(
        "-t", "--tgt", type=str, required=True, dest="TgtFile",
        help="Target structure (.xyz) to be mapped onto the reference numbering "
             "(e.g. the geometry a force field was generated on).",
    )

    out = parser.add_argument_group("Output Data")
    out.add_argument(
        "-m", "--map", type=str, default="map.txt", dest="MapFile",
        help="Output file for the atom map (reference target columns, 1-based).",
    )
    out.add_argument(
        "-o", "--output", type=str, default=None, dest="OutFile",
        help="Optional output file for the target structure reordered to follow "
             "the reference numbering (format inferred from extension, e.g. .xyz).",
    )
    out.add_argument(
        "-v", "--verbose", action="store_true", dest="Verbose",
        help="Enable progress logging.",
    )

    return vars(parser.parse_args())


def main():
    Opts = options()

    logging.basicConfig(
        level=logging.INFO if Opts["Verbose"] else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    mapper = AtomMapper(Opts["RefFile"], Opts["TgtFile"])
    result = mapper.map()
    result.write(Opts["MapFile"])
    log.info("Map written to %s", Opts["MapFile"])

    if Opts["OutFile"]:
        mapper.write_reordered(Opts["OutFile"])
        log.info("Reordered structure written to %s", Opts["OutFile"])


if __name__ == "__main__":
    main()
