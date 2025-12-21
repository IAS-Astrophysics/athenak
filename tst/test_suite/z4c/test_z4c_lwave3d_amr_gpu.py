"""
Linear wave convergence test for Z4c with AMR in full 3D.
Does a convergence test for 2nd-order, but only tests amplitude error for 6th-order.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read


# Threshold errors and error ratios for different finite-difference orders
errors = {("2nd-order"): (3.5e-11, 0.25), ("6th-order"): (6.0e-12, 0.0)}

_res = [32, 64]  # resolutions to test


def arguments(res, ng):
    """Assemble arguments for run command"""
    return [
        "mesh/nghost=" + repr(ng),
        "mesh/nx1=" + repr(res),
        "mesh/nx2=" + repr(res),
        "mesh/nx3=" + repr(res),
        "meshblock/nx1=" + repr(res // 8),
        "meshblock/nx2=" + repr(res // 8),
        "meshblock/nx3=" + repr(res // 8),
        "mesh_refinement/max_nmb_per_rank=4096",
        "problem/kx1=1",
        "problem/kx2=1",
        "problem/kx3=1",
    ]


input_file = "inputs/lwave_z4c.athinput"


# tests both maxerror and convergence at 2nd-order
def test_run_2nd():
    """Run a single test."""
    ng = 2
    try:
        for res in _res:
            results = testutils.run(input_file, arguments(res, ng))
            assert results, f"2nd-order Z4c linear wave run failed for {res}."
        maxerror, errorratio = errors[("2nd-order")]
        data = athena_read.error_dat("z4c_lin_wave-errs.dat")
        L1_RMS_INDEX = 4  # Index for L1 RMS error in data
        l1_rms_err0 = data[0][L1_RMS_INDEX]
        l1_rms_err1 = data[1][L1_RMS_INDEX]
        if l1_rms_err1 > maxerror:
            pytest.fail(
                f"2nd-order Z4c wave error too large,"
                f"error: {l1_rms_err1:g} threshold: {maxerror:g}"
            )
        if (l1_rms_err1 / l1_rms_err0) > errorratio:
            pytest.fail(
                f"2nd-order Z4c wave converging too slow,"
                f"error ratio: {(l1_rms_err1/l1_rms_err0):g}"
                f"  expected ratio: {errorratio:g}"
            )
    finally:
        testutils.cleanup()


# only tests maxerror at 6th-order
def test_run_6th():
    """Run a single test."""
    res = 64
    ng = 4
    try:
        results = testutils.run(input_file, arguments(res, ng))
        assert results, f"6th-order Z4c linear wave run failed for {res}."
        maxerror, errorratio = errors[("6th-order")]
        data = athena_read.error_dat("z4c_lin_wave-errs.dat")
        L1_RMS_INDEX = 4  # Index for L1 RMS error in data
        l1_rms_err0 = data[0][L1_RMS_INDEX]
        if l1_rms_err0 > maxerror:
            pytest.fail(
                f"6th-order Z4c wave error too large,"
                f"error: {l1_rms_err0:g} threshold: {maxerror:g}"
            )
    finally:
        testutils.cleanup()
