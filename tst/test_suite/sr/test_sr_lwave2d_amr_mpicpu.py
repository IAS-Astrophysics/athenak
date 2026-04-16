"""
Linear wave convergence test for special-relativistic hydro/MHD in 2D with AMR and MPI.
Runs tests in both hydro and MHD for RK2+PLM and RK3+WENOZ using HLLE Riemann solver.
Only tests "0" wave
"""

# Modules
import pytest
import test_suite.testutils as testutils

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
maxerrors = {
    ("hydro", "rk2", "plm", "0"): (9.0e-06, 0.27),
    ("hydro", "rk3", "wenoz", "0"): (9e-07, 0.31),
    ("mhd", "rk2", "plm", "0"): (2.9e-05, 0.27),
    ("mhd", "rk3", "wenoz", "0"): (2e-06, 0.22),
}

_wave = ["0"]  # do not change order
_res = [64, 128]  # resolutions to test
_recon = ["plm", "wenoz"]  # spatial reconstruction
_soe = ["hydro", "mhd"]  # system of equations to test


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nghost=" + repr(2 if rv == "plm" else 4),
        "mesh/nx1=" + repr(res),
        "mesh/nx2=" + repr(res // 2),
        "mesh/nx3=1",
        "meshblock/nx1=" + repr(res // 16),
        "meshblock/nx2=" + repr(res // 16),
        "meshblock/nx3=1",
        "time/cfl_number=0.4",
        "coord/special_rel=true",
        "coord/general_rel=false",
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/amp=1.0e-3",
        "problem/wave_flag=" + wv,
    ]


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("fv", ["hlle"])
@pytest.mark.parametrize("soe", _soe)
def test_run(fv, rv, soe):
    """Run a single test with given parameters."""
    iv = "rk2" if rv == "plm" else "rk3"
    # Ignore return arguments
    _, _ = testutils.test_error_convergence(
        f"inputs/lwave_rel{soe}.athinput",
        f"sr_lwave_{soe}",
        arguments,
        maxerrors,
        _wave,
        _res,
        iv,
        rv,
        fv,
        soe,
        mpi=True,
    )
