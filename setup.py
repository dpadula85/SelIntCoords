#!/usr/bin/env python

import setuptools
from setuptools import Extension, setup

setup(
    name="SelIntCoords",
    version="1.1",
    author="Daniele Padula",
    author_email="dpadula85@yahoo.it",
    description="A python package to select internal coordinates for FF fitting",
    url="https://github.com/dpadula85/selintcoords",
    packages=setuptools.find_packages(),
    install_requires=[
        "chain-cropper",
        # sel_intcoords.py/make_top.py import these directly; only
        # chain-cropper's own transitive MDAnalysis/numpy happened to be
        # covered before -- networkx and pandas were missing entirely.
        "MDAnalysis",
        "networkx",
        "numpy",
        "pandas",
        # Point-group detection (make_top/sel_intcoords only -- see
        # README.md's Requirements section).
        "pyscf",
        # NOT listed here: `psi4`, also required by make_top/sel_intcoords
        # for point-group detection, is not distributed on PyPI at all --
        # pip has no way to satisfy it (only conda-forge, `conda install
        # -c conda-forge psi4`), so declaring it here would just make
        # `pip install` fail outright rather than document a real gap.
    ],
    entry_points={
        'console_scripts' : [
            'make_top=SelIntCoords.make_top:main',
            'map_atoms=SelIntCoords.map_atoms:main',
            'renumber_top=SelIntCoords.renumber:main',
            'joyce_to_openmm=SelIntCoords.joyce_to_openmm:main',
            ]
        },
    zip_safe=False
)
