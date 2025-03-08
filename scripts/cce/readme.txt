# h5
./athk_to_spectre.py -fpath dat/bbh_q2.0_chizp0.0_chizm0.0_d10.0_lev13_n128_fixed_cce_decomp_shell_3.h5 -ftype h5 -d_out ./debug/q2/ -radius 100

# extract txt
./cce_extract_txt.py -f ./runs/CharacteristicExtractReduction.h5 -dout  ./debug/q2/ -field Strain
./cce_extract_txt.py -f ./runs/CharacteristicExtractReduction.h5 -dout  ./debug/q2/ -field Psi4

