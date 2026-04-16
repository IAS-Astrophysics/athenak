# Directory Guide

## Role
Pytest coverage for shearing-box and orbital-advection behavior.

## Important Files
- `test_sbox_hydroshwave_mpicpu.py`, `test_sbox_mhdshwave_mpicpu.py`: shearing-wave regressions.
- `test_sbox_hydroshwave_sts_mpicpu.py`: Hydro shearing-wave MPI acceptance for the STS
  parabolic loop on the shearing-box path.
- `test_sbox_hydro_orbital_sts_cpu.py`: Hydro orbital-advection STS smoke using the
  tracked orbital mesh template with a temporary Hydro viscosity diffusion setup.
- `test_sbox_mhdshwave_sts_mpicpu.py`: MHD shearing-wave MPI acceptance for the STS
  parabolic loop on the shearing-box path.
- `test_sbox_mhd_orbital_sts_cpu.py`: MHD orbital-advection STS smoke using the tracked
  orbital mesh template with a temporary built-in resistive-diffusion setup.
- `test_sbox_mri3d_gpu.py`: MRI-focused GPU regression.

## Read This Next
- Pair these with `src/shearing_box/AGENTS.md`.
