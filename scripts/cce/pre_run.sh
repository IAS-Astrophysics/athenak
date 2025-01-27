#!/usr/bin/env bash

# bin
if true
then
# create
dout='./debug/dev'
file='./dat/bin/cce_bbh/'
fdebug='./debug/dev/CceR0030.00.h5'

./athk_to_spectre.py -ftype bin -d_out "$dout" -debug y \
-t_deriv fin_diff -r_deriv ChebU -interpolation ChebU \
-radius 30 \
-fpath $file

# plot fig:
fname="gxx"
mode="Re(2,2)"
./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
	-fpath $fdebug  -field_name $fname -field_mode $mode

# h5 bbh, q=2
elif false
then
./athk_to_spectre.py -fpath dat/bbh_q2.0_chizp0.0_chizm0.0_d10.0_lev13_n128_fixed_cce_decomp_shell_3.h5 \
-ftype h5 -d_out ./debug/q2/ -radius 100

file="./debug/q2/CceR0100.00.h5"
dout='./debug/q2'

# plot fig:
fname="gxx"
mode="Re(2,2)"
./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
	-fpath $file  -field_name $fname -field_mode $mode

#mode="Re(0,0)"
#./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
#	-fpath $file  -field_name $fname -field_mode $mode


#fname="alp"
#mode="Re(0,0)"
#./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
#	-fpath $file  -field_name $fname -field_mode $mode

#mode="Re(2,2)"
#./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
#	-fpath $file  -field_name $fname -field_mode $mode


#fname="betax"
#mode="Re(0,0)"
#./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
#	-fpath $file  -field_name $fname -field_mode $mode

#mode="Re(2,2)"
#./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
#	-fpath $file  -field_name $fname -field_mode $mode
fi

