# Directory Guide

## Role
Test-only runtime inputs used by the maintained regression harnesses under `tst/`.

## Important Files
- `lwave_*.athinput`, `mb2.athinput`, `gr_bondi.athinput`, `gr_monopole.athinput`: representative fixtures for the pytest suite.
- `cshock.athinput`, `mri3d.athinput`, `z4c_lw.athinput`: specialized cases for feature-specific suites.

## Important Subdirectories
- `tables/`: tabulated EOS fixtures required by some tests.

## Read This Next
- Compare a test file here with the matching suite under `tst/test_suite/`.
