"""
Linear wave convergence test for non-relativistic hydro/MHD in 3D with AMR and MPI.
Runs tests in both hydro and MHD for RK2+PLM and RK3+WENOZ using HLLE Riemann solver.
Only tests "0" wave
"""

# Modules
import pytest
import test_suite.testutils as testutils

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
errors = {
    ("hydro", "rk2", "plm", "0"): (5e-05, 0.32),
    ("hydro", "rk3", "wenoz", "0"): (1.2e-05, 0.2),
    ("mhd", "rk2", "plm", "0"): (5.5e-05, 0.3),
    ("mhd", "rk3", "wenoz", "0"): (1.2e-05, 0.21),
}

_wave = ["0"]
_recon = ["plm", "wenoz"]
_res = [32, 64]


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
        "time/cfl_number=0.3",
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/amp=1.0e-3",
        "problem/wave_flag=" + wv,
    ]


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("fv", ["hlle"])
@pytest.mark.parametrize("soe", ["hydro", "mhd"])
def test_run(fv, rv, soe):
    """run test with given integrator/resolution/physics."""
    iv = "rk2" if rv == "plm" else "rk3"
    # Ignore return arguments
    _, _ = testutils.test_error_convergence(
        f"inputs/lwave_{soe}.athinput",
        f"lwave3d_amr_{soe}",
        arguments,
        errors,
        _wave,
        _res,
        iv,
        rv,
        fv,
        soe,
        mpi=True,
    )
