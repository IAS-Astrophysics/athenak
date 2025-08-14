"""
Linear wave convergence test for non-relativistic hydro/MHD in 3D with AMR and MPI.
Runs tests in both hydro and MHD for different
  - time integrators
  - reconstruction algorithms
Only tests "0" wave and HLLC/HLLD Riemann solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
errors = {
    ("hydro", "rk2", "plm", "0"): (1.4e-05, 0.27),
    ("hydro", "rk3", "ppm4", "0"): (8.3e-06, 0.35),
    ("hydro", "rk3", "ppmx", "0"): (7.4e-06, 0.43),
    ("hydro", "rk3", "wenoz", "0"): (5.9e-06, 0.52),
    ("mhd", "rk2", "plm", "0"): (1.5e-05, 0.26),
    ("mhd", "rk3", "ppm4", "0"): (9.8e-06, 0.32),
    ("mhd", "rk3", "ppmx", "0"): (8e-06, 0.39),
    ("mhd", "rk3", "wenoz", "0"): (4.1e-06, 0.34),
}

_recon = ["plm", "ppm4", "ppmx", "wenoz"]
_wave = ["0"]
_res = [64, 128]  # test run on GPU, so use higher res


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
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/amp=1.0e-3",
        "problem/wave_flag=" + wv,
    ]


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("fv", ["hlle"])
@pytest.mark.parametrize("soe", ["hydro", "mhd"])
def test_run(rv, fv, soe):
    """Run a single test with given parameters."""
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
    )
