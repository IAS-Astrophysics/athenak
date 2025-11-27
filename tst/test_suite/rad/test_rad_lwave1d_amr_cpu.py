"""
Linear wave convergence test for 1D radiation hydro with AMR.
Problem generator only initializes L-going radiation acoustic mode
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
        "mesh/nx1=" + repr(res),
        "mesh/nx2=1",
        "mesh/nx3=1",
        "meshblock/nx1=" + repr(res // 8),
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        "problem/along_x1=true",
        "problem/along_x2=false",
        "problem/along_x3=false"
    ]


input_file = "inputs/lwave_rad.athinput"


# run test
def test_run():
    """Run a single test."""
    try:
        for res in _res:
            results = testutils.run(input_file, arguments(res))
            assert results, f"1D radiation hydro linear wave run failed for {res}."
        maxerror, errorratio = errors[("rad-hydro")]
        data = athena_read.error_dat("rad_linwave-errs.dat")
        L1_RMS_INDEX = 4  # Index for L1 RMS error in data
        l1_rms_err0 = data[0][L1_RMS_INDEX]
        l1_rms_err1 = data[1][L1_RMS_INDEX]
        if l1_rms_err1 > maxerror:
            pytest.fail(
                f"1D radiation hydro wave error too large,"
                f"error: {l1_rms_err1:g} threshold: {maxerror:g}"
            )
        if (l1_rms_err1 / l1_rms_err0) > errorratio:
            pytest.fail(
                f"1D radiation hydro wave converging too slow,"
                f"error ratio: {(l1_rms_err1/l1_rms_err0):g}"
                f"  expected ratio: {errorratio:g}"
            )
    finally:
        testutils.cleanup()
