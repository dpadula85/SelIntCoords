#!/bin/bash

make_top -p BTBT.top -m btbt_Et.xyz -o BTBT_nosymm.top
make_top -p BTBT.top -m btbt_Et_symm.xyz -o BTBT_symm.top
make_top -p PN.top -m pn_symm.xyz -o PN_symm.top
make_top -p DNBDT.top -m dnbdt_symm.xyz -o DNBDT_symm.top
