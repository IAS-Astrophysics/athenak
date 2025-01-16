#!/usr/bin/env bash

if false
then
./athk_to_spectre.py -ftype bin -d_out ./debug -debug y \
-t_deriv fin_diff -r_deriv ChebU -interpolation ChebU \
-radius 30 \
-fpath ./dat/bin/cce_bbh/
fi

# plot fig
./debug_athk_to_spectre.py -debug plot_simple -fpath ./debug/CceR0030.00.h5 -dout ./debug
