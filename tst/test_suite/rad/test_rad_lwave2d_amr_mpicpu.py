"""
Linear wave convergence test for 2D radiation hydro with AMR + MPI.

Problem generator only initializes L-going radiation acoustic mode

Moroever, the analytic solution against which the numerical evolution is compared
requires the Eddington approximation, and this can only be enforced in the code
when the wave propagates along a grid direction. So in 2D radiation hydro waves
propagate along the x2-axis instead of the domain diagonal.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
errors = {("rad-hydro"): (3.5e-7, 0.23)}

_res = [32, 64]  # resolutions to test


def arguments(res):
    """Assemble arguments for run command"""
    return [
        "mesh/nx1=4",
        "mesh/nx2=" + repr(res),
        "mesh/nx3=1",
        "meshblock/nx1=4",
        "meshblock/nx2=" + repr(res // 8),
        "meshblock/nx3=1",
        "problem/along_x1=false",
        "problem/along_x2=true",
        "problem/along_x3=false"
    ]


input_file = "inputs/lwave_rad.athinput"


# run test
def test_run():
    """Run a single test."""
    try:
        for res in _res:
            # set number of threads to initial number of MeshBlocks
            results = testutils.mpi_run(input_file, arguments(res), threads=8)
            assert results, f"2D radiation hydro linear wave run failed for {res}."
        maxerror, errorratio = errors[("rad-hydro")]
        data = athena_read.error_dat("rad_linwave-errs.dat")
        L1_RMS_INDEX = 4  # Index for L1 RMS error in data
        l1_rms_err0 = data[0][L1_RMS_INDEX]
        l1_rms_err1 = data[1][L1_RMS_INDEX]
        if l1_rms_err1 > maxerror:
            pytest.fail(
                f"2D radiation hydro wave error too large,"
                f"error: {l1_rms_err1:g} threshold: {maxerror:g}"
            )
        if (l1_rms_err1 / l1_rms_err0) > errorratio:
            pytest.fail(
                f"2D radiation hydro wave converging too slow,"
                f"error ratio: {(l1_rms_err1/l1_rms_err0):g}"
                f"  expected ratio: {errorratio:g}"
            )
    finally:
        testutils.cleanup()
