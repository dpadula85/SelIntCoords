#!/usr/bin/env python

import setuptools
from setuptools import Extension, setup

setup(
    name="SelIntCoords",
    version="1.2",
    author="Daniele Padula",
    author_email="dpadula85@yahoo.it",
    description="A python package to select internal coordinates for FF fitting",
    url="https://github.com/dpadula85/selintcoords",
    packages=setuptools.find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.10",
    install_requires=[
        "chain-cropper",
        "MDAnalysis",
        "networkx",
        "numpy",
        "pandas",
        "scipy",
    ],
    extras_require={"test": ["pytest", "openmm", "rdkit"]},
    entry_points={
        'console_scripts' : [
            'make_top=SelIntCoords.make_top:main',
            'perceive_types=SelIntCoords.perceive_types:main',
            'map_atoms=SelIntCoords.map_atoms:main',
            'renumber_top=SelIntCoords.renumber:main',
            'joyce_to_openmm=SelIntCoords.joyce_to_openmm:main',
            ]
        },
    zip_safe=False
)
