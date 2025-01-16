#!/usr/bin/env bash

if false
then
./athk_to_spectre.py -ftype bin -d_out ./debug -debug y \
-t_deriv fin_diff -r_deriv ChebU -interpolation ChebU \
-radius 30 \
-fpath ./dat/bin/cce_bbh/
fi

file="./debug/CceR0030.00.h5"
# plot fig
fname="gxx"
mode="Re(0,0)"
./debug_athk_to_spectre.py -debug plot_simple -dout ./debug \
	-fpath $file  -field_name $fname -field_mode $mode

mode="Re(2,2)"
./debug_athk_to_spectre.py -debug plot_simple -dout ./debug \
	-fpath $file  -field_name $fname -field_mode $mode


fname="alp"
mode="Re(0,0)"
./debug_athk_to_spectre.py -debug plot_simple -dout ./debug \
	-fpath $file  -field_name $fname -field_mode $mode

mode="Re(2,2)"
./debug_athk_to_spectre.py -debug plot_simple -dout ./debug \
	-fpath $file  -field_name $fname -field_mode $mode


fname="betax"
mode="Re(0,0)"
./debug_athk_to_spectre.py -debug plot_simple -dout ./debug \
	-fpath $file  -field_name $fname -field_mode $mode

mode="Re(2,2)"
./debug_athk_to_spectre.py -debug plot_simple -dout ./debug \
	-fpath $file  -field_name $fname -field_mode $mode

