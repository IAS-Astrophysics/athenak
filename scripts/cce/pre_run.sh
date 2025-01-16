#!/usr/bin/env bash

./athk_to_specter.py -ftype bin -d_out ./debug -debug y \
-t_deriv fin_diff -r_deriv ChebU -interpolation ChebU \
-radius 30 \
-fpath ./dat/bin/cce_bbh/

