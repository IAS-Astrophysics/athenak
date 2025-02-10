#!/usr/bin/env bash

set -x

# bin
if false
then
# create
dout='./debug/dev'
file='./dat/bin/cce_bbh/'
fdebug='./debug/dev/CceR0030.00.h5'

rm -vf ${dout}/*

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
elif true
then
file="./debug/q2/CceR0100.00.h5"
dout='./debug/q2'
mkdir -p ${dout}
rm -rvf ${dout}/*

./athk_to_spectre.py -fpath dat/bbh_q2.0_chizp0.0_chizm0.0_d10.0_lev13_n128_fixed_cce_decomp_shell_3.h5 \
-ftype h5 -d_out "${dout}" -radius 100

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

# h5 bbh, q=2, fourier time derivatives
elif false
then
file="./debug/q2_t_fourier/CceR0100.00.h5"
dout='./debug/q2_t_fourier'
mkdir -p ${dout}

./athk_to_spectre.py -fpath dat/bbh_q2.0_chizp0.0_chizm0.0_d10.0_lev13_n128_fixed_cce_decomp_shell_3.h5 \
-ftype h5 -d_out "$dout" -radius 100 -t_deriv "Fourier"


# plot fig:
fname="gxx"
mode="Re(2,2)"
./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
	-fpath $file  -field_name $fname -field_mode $mode

elif false
then
file="./debug/q2_dev/CceR0100.00.h5"
dout='./debug/q2_dev/'
mkdir -p ${dout}

./athk_to_spectre.py -fpath dat/bbh_q2.0_chizp0.0_chizm0.0_d10.0_lev13_n128_fixed_cce_decomp_shell_3.h5 \
-ftype h5 -d_out "$dout" -radius 100


# plot fig:
fname="gxx"
mode="Re(2,2)"
./debug_athk_to_spectre.py -debug plot_simple -dout "${dout}" \
	-fpath $file  -field_name $fname -field_mode $mode

# h5 quick development
elif false
then
dout='./debug/quick_dev/'
mkdir -p ${dout}

./athk_to_spectre.py -fpath dat/backup_dilute_d32_fixed_cce_decomp_shell_4.h5 \
-ftype h5 -d_out "$dout" -radius 70

fi

