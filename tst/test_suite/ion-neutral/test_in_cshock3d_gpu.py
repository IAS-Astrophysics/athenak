"""
Oblique cshock test for non-relativistic two-fluid (ion-neutral) MHD in 3D on GPUs
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


# On GPU runs a 3D test
def arguments(iv, rv, res):
    """Assemble arguments for run command"""
    return [
        "mesh/nx1=" + repr(res // 8),
        "mesh/ix1_bc=periodic",
        "mesh/ox1_bc=periodic",
        "mesh/nx2=" + repr(res // 8),
        "mesh/ix2_bc=periodic",
        "mesh/ox2_bc=periodic",
        "mesh/nx3=" + repr(res),
        "mesh/ix3_bc=inflow",
        "mesh/ox3_bc=outflow",
        "meshblock/nx1=" + repr(res // 8),
        "meshblock/nx2=" + repr(res // 8),
        "meshblock/nx3=" + repr(res // 4),
        "mesh/nghost=" + repr(2 if rv == "plm" else 3),
        "time/integrator=" + iv,
        "time/cfl_number=0.3",
        "hydro/reconstruct=" + rv,
        "mhd/reconstruct=" + rv,
        "problem/shock_dir=3",
    ]


@pytest.mark.parametrize("iv", _int)
@pytest.mark.parametrize("res", _res)
def test_run(iv, res):
    """Run test with given integrator/reconstruction."""
    rv = "plm" if iv == "imex2" else "wenoz"
    try:
        results = testutils.run(input_file, arguments(iv, rv, res))
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
