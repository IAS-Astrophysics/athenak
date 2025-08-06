"""
Linear wave convergence test for Z4c with AMR and MPI.
Because it is not possible to run the Z4c module in true 2D, this test runs a
quasi-2D problem with symmetry and only 4 cells in the z-direction.
Only 2nd order algorithm is tested on CPUs.
"""

# Modules
import sys
sys.path.insert(0, '../vis/python')
sys.path.insert(0, '../test_suite')
import pytest
import test_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
errors={
    ('2nd-order'): (6.0e-11,0.25)}

_res = [32, 64]     # resolutions to test

def arguments(res):
    """Assemble arguments for run command"""
    return ['mesh/nx1=' + repr(res),
            'mesh/nx2=' + repr(res),
            'mesh/nx3=4',
            'meshblock/nx1=' + repr(res//8),
            'meshblock/nx2=' + repr(res//8),
            'meshblock/nx3=4',
            'problem/kx1=1',
            'problem/kx2=1',
            'problem/kx3=0']

def test_run():
    """Run a single test."""
    try:
        for res in _res:
            results = mpi_run(input_file, arguments(res))
            assert results, f"Z4c linear wave run failed for {res}."
        maxerror, errorratio = errors[('2nd-order')]
        data = athena_read.error_dat(f'z4c_lin_wave-errs.dat')
        L1_RMS_INDEX = 4  # Index for L1 RMS error in data
        l1_rms_err0 = data[0][L1_RMS_INDEX]
        l1_rms_err1 = data[1][L1_RMS_INDEX]
        if l1_rms_err1 > maxerror:
            pytest.fail(f"Z4c wave error too large,"
                        f"error: {l1_rms_err1:g} threshold: {maxerror:g}")
        if (l1_rms_err1 / l1_rms_err0) > errorratio:
            pytest.fail(f"Z4c wave converging too slow,"
                        f"error ratio: {(l1_rms_err1/l1_rms_err0):g}"
                        f"  expected ratio: {errorratio:g}")
    finally:
        cleanup()
