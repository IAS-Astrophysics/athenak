# Dynamical Radiation Test Matrix

This file records the tests added for the standalone `dyn_radiation` solver and
how they map onto the verification suite in `radiation_method.tex`.

## Added Inputs

All inputs below use `<dyn_radiation>` rather than `<radiation>`.

| Input | Coverage |
| --- | --- |
| `inputs/tests/dynrad_tetrad_cks.athinput` | CKS-mode construction and orthonormality check through the built-in `rad_beam` test pgen. |
| `inputs/tests/dynrad_tetrad_adm.athinput` | ADM-mode Eulerian tetrad construction and orthonormality check on flat ADM data. |
| `inputs/tests/dynrad_beam_cks.athinput` | CKS-mode transport, beam source term, radiation-moment output, physical boundary fills, and MPI communication. |
| `inputs/tests/dynrad_beam_adm_flat.athinput` | ADM-mode flat-spacetime transport, ADM flux path, and the zero-valued ADM geometric source path. |
| `inputs/tests/dynrad_lwave.athinput` | Radiation-hydro coupling with the linear-wave pgen and the copied local implicit matter update. |
| `inputs/tests/dynrad_lwave_smr.athinput` | Same linear-wave coupling with static mesh refinement, covering radiation restriction/prolongation and AMR buffer packing for `pdynrad`. |

## Commands Used In This Port

```bash
cmake --build build -j6
./build/src/athena -i inputs/tests/dynrad_tetrad_cks.athinput
./build/src/athena -i inputs/tests/dynrad_tetrad_adm.athinput
./build/src/athena -i inputs/tests/dynrad_lwave.athinput job/basename=/tmp/dynrad_lwave_input_smoke
./build/src/athena -i inputs/tests/dynrad_lwave_smr.athinput job/basename=/tmp/dynrad_lwave_smr_smoke
./build/src/athena -i inputs/tests/dynrad_beam_cks.athinput job/basename=dynrad_beam_input_smoke
./build/src/athena -i inputs/tests/dynrad_beam_adm_flat.athinput job/basename=dynrad_beam_adm_flat_smoke
mpirun -n 2 ./build/src/athena -i inputs/tests/dynrad_beam_cks.athinput job/basename=dynrad_beam_mpi_smoke
./build/src/athena -i inputs/tests/dynrad_beam_cks.athinput job/basename=dynrad_restart_test output1/file_type=rst output1/dt=0.1 time/tlim=0.1
./build/src/athena -r rst/dynrad_restart_test.00001.rst time/tlim=0.12 job/basename=dynrad_restart_resume output1/file_type=tab output1/dt=0.12
```

The beam output was checked for finite values and for a positive transported
`r00` signal.  The flat CKS and flat ADM beam outputs were also compared
component-by-component and matched to tabular precision.  The linear-wave output
was checked for finite errors.

## Mapping To `radiation_method.tex`

| Paper test | Current dyn-radiation status |
| --- | --- |
| Colliding beams | Partially covered by the beam transport/source smoke test. A dedicated two-beam pgen or two-source input is still needed for a literal reproduction. |
| Beams in curvilinear coordinates | The current committed dyn-radiation input covers CKS/Minkowski beam transport. The old snake-coordinate pgen is not part of the default built-in pgen set and still assumes the legacy radiation object. |
| Beams around black holes | CKS geometry mode is the standalone analogue of the old stationary-spacetime solver, but the long black-hole beam input has not been duplicated as a compact committed regression test. |
| Hohlraums | Not yet ported to a built-in dyn-radiation test input; the existing hohlraum pgen is a custom pgen outside the default test binary. |
| Radiating disk | Not yet ported. The `gr_torus`/`dynbbh` initializers still initialize the legacy radiation object only. |
| Equilibration | Covered at the source-kernel level by the linear-wave coupling test, but the standalone relaxation pgen is not yet ported to `dyn_radiation`. |
| Diffusion | Covered indirectly by the optically thick linear wave. The dedicated diffusion pgen remains legacy-only. |
| Shocks | No dedicated dyn-radiation shock input exists in this pass. |
| Linear waves | Covered by `dynrad_lwave.athinput` and `dynrad_lwave_smr.athinput`. |
| Schwarzschild atmosphere | Not yet ported; there is no compact built-in atmosphere test in this tree. |

The current implementation therefore provides regression coverage for every new
code path introduced in `dyn_radiation`: CKS transport, ADM tetrad construction,
ADM geometric energy source setup, output, restart, MPI, and SMR/AMR data motion.
It does not yet claim a full one-for-one reproduction of every production-scale
test in the paper.
