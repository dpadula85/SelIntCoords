#!/usr/bin/env python

import setuptools
from setuptools import Extension, setup

setup(
    name="SelIntCoords",
    version="1.0",
    author="Daniele Padula",
    author_email="dpadula85@yahoo.it",
    description="A python package to select internal coordinates for FF fitting",
    url="https://github.com/dpadula85/selintcoords",
    packages=setuptools.find_packages(),
    entry_points={ 
        'console_scripts' : [
            'make_top=SelIntCoords.make_top:main'
            ]
        },
    zip_safe=False
)
