"""
Linear wave convergence test for general-relativistic MHD in dynamical spacetime (dyngr)
in 3D and with AMR.
Runs tests in MHD for RK2+PLM and RK3+WENOZ using HLLE Riemann solver
Only tests "0" wave
"""

# Modules
import pytest
import test_suite.testutils as testutils

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
errors = {
    ("mhd", "rk2", "plm", "0"): (8.2e-05, 0.25),
    ("mhd", "rk3", "wenoz", "0"): (2.7e-05, 0.18),
}

_wave = ["0"]  # do not change order
_res = [32, 64]  # resolutions to test
_recon = ["plm", "wenoz"]  # spatial reconstruction
_soe = ["mhd"]  # system of equations to test


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nghost=" + repr(2 if rv == "plm" else 4),
        "mesh/nx1=" + repr(res),
        "mesh/nx2=" + repr(res // 2),
        "mesh/nx3=" + repr(res // 2),
        "meshblock/nx1=" + repr(res // 8),
        "meshblock/nx2=" + repr(res // 8),
        "meshblock/nx3=" + repr(res // 8),
        "mesh_refinement/max_nmb_per_rank=2048",
        "time/cfl_number=0.3",
        "coord/special_rel=false",
        "coord/general_rel=true",
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/amp=1.0e-3",
        "problem/wave_flag=" + wv,
    ]


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("fv", ["hlle"])
@pytest.mark.parametrize("soe", _soe)
def test_run(rv, fv, soe):
    """Run a single test with given parameters."""
    iv = "rk2" if rv == "plm" else "rk3"
    # Ignore return arguments
    _, _ = testutils.test_error_convergence(
        "inputs/lwave_dyngrmhd.athinput",
        "lwave3d_amr_{soe}",
        arguments,
        errors,
        _wave,
        _res,
        iv,
        rv,
        fv,
        soe,
    )
