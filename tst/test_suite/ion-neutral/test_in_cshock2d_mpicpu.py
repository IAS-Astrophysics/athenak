"""
Oblique cshock test for non-relativistic two-fluid (ion-neutral) MHD in 2D using MPI
Runs tests for different
  - IMEX time integrators
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read


# Threshold errors for test
errors = {("imex2"): (0.75), ("imex3"): (0.85)}

_int = ["imex2", "imex3"]
_res = [128]
input_file = "inputs/cshock.athinput"


# On CPU with MPI runs a 2D test
def arguments(iv, rv, res):
    """Assemble arguments for run command"""
    return [
        "mesh/nx1=" + repr(res // 4),
        "mesh/ix1_bc=periodic",
        "mesh/ox1_bc=periodic",
        "mesh/nx2=" + repr(res),
        "mesh/ix2_bc=inflow",
        "mesh/ox2_bc=outflow",
        "mesh/nx3=1",
        "mesh/ix3_bc=periodic",
        "mesh/ox3_bc=periodic",
        "meshblock/nx1=" + repr(res // 8),
        "meshblock/nx2=" + repr(res // 8),
        "meshblock/nx3=1",
        "mesh/nghost=" + repr(2 if rv == "plm" else 3),
        "time/integrator=" + iv,
        "time/cfl_number=0.3",
        "hydro/reconstruct=" + rv,
        "mhd/reconstruct=" + rv,
        "problem/shock_dir=2",
    ]


@pytest.mark.parametrize("iv", _int)
@pytest.mark.parametrize("res", _res)
def test_run(iv, res):
    """Run test with given integrator/reconstruction."""
    rv = "plm" if iv == "imex2" else "wenoz"
    try:
        results = testutils.mpi_run(input_file, arguments(iv, rv, res))
        assert results, f"cshock test run failed for {iv}+{rv}."
        maxerr = errors[(iv)]
        data = athena_read.error_dat("cshock-errs.dat")
        L1_RMS_INDEX = 4  # Index for L1 RMS error in data
        l1_rms_err = data[0][L1_RMS_INDEX]
        # Check the errors in the output
        if l1_rms_err > maxerr:
            pytest.fail(
                f"Cshock error too large for {iv}+{rv}, "
                f"conv: {l1_rms_err:g} threshold: {maxerr:g}"
            )
    finally:
        testutils.cleanup()
