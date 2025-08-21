"""
Linear wave convergence test for general-relativistic hydro/MHD in 1D.
Runs tests in both hydro and MHD using RK3 for different
  - reconstruction algorithms
  - Riemann solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils


# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and wave types
maxerrors = {
    ("hydro", "rk2", "plm", "0"): (1.9e-08, 0.28),
    ("hydro", "rk3", "ppm4", "0"): (5.6e-09, 0.26),
    ("hydro", "rk3", "ppmx", "0"): (3.8e-11, 0.092),
    ("hydro", "rk3", "wenoz", "0"): (1.3e-11, 0.23),
    ("hydro", "rk2", "plm", "4"): (1.7e-08, 0.28),
    ("hydro", "rk3", "ppm4", "4"): (6.9e-09, 0.30),
    ("hydro", "rk3", "ppmx", "4"): (1.1e-11, 0.035),
    ("hydro", "rk3", "wenoz", "4"): (9.5e-12, 0.27),
    ("hydro", "rk2", "plm", "3"): (1.8e-07, 0.33),
    ("hydro", "rk3", "ppm4", "3"): (3.9e-08, 0.24),
    ("hydro", "rk3", "ppmx", "3"): (1.2e-10, 0.064),
    ("hydro", "rk3", "wenoz", "3"): (2.6e-11, 0.032),
    ("mhd", "rk2", "plm", "0"): (5.6e-08, 0.28),
    ("mhd", "rk3", "ppm4", "0"): (2e-08, 0.27),
    ("mhd", "rk3", "ppmx", "0"): (5e-10, 0.21),
    ("mhd", "rk3", "wenoz", "0"): (4.9e-10, 0.24),
    ("mhd", "rk2", "plm", "6"): (2.3e-08, 0.28),
    ("mhd", "rk3", "ppm4", "6"): (8.8e-09, 0.27),
    ("mhd", "rk3", "ppmx", "6"): (4.5e-10, 0.24),
    ("mhd", "rk3", "wenoz", "6"): (4.4e-10, 0.25),
    ("mhd", "rk2", "plm", "5"): (6e-08, 0.29),
    ("mhd", "rk3", "ppm4", "5"): (2.3e-08, 0.25),
    ("mhd", "rk3", "ppmx", "5"): (8.3e-10, 0.28),
    ("mhd", "rk3", "wenoz", "5"): (8.7e-10, 0.24),
    ("mhd", "rk2", "plm", "1"): (4.3e-08, 0.28),
    ("mhd", "rk3", "ppm4", "1"): (1.4e-08, 0.26),
    ("mhd", "rk3", "ppmx", "1"): (1.2e-09, 0.26),
    ("mhd", "rk3", "wenoz", "1"): (1.2e-09, 0.25),
    ("mhd", "rk2", "plm", "4"): (4.1e-08, 0.33),
    ("mhd", "rk3", "ppm4", "4"): (1.2e-08, 0.23),
    ("mhd", "rk3", "ppmx", "4"): (1.1e-10, 0.23),
    ("mhd", "rk3", "wenoz", "4"): (1.2e-10, 0.21),
    ("mhd", "rk2", "plm", "2"): (1.6e-08, 0.29),
    ("mhd", "rk3", "ppm4", "2"): (5.3e-09, 0.25),
    ("mhd", "rk3", "ppmx", "2"): (5.2e-11, 0.17),
    ("mhd", "rk3", "wenoz", "2"): (4.8e-11, 0.26),
    ("mhd", "rk2", "plm", "3"): (3.3e-08, 0.37),
    ("mhd", "rk3", "ppm4", "3"): (4.9e-09, 0.24),
    ("mhd", "rk3", "ppmx", "3"): (1.4e-11, 0.065),
    ("mhd", "rk3", "wenoz", "3"): (5.7e-12, 0.033),
}

_recon = ["plm", "ppm4", "ppmx", "wenoz"]
_wave = {}
_wave["mhd"] = ["0", "6", "5", "1", "4", "2", "3"]
_wave["hydro"] = ["0", "4", "3"]
_flux = {}
_flux["mhd"] = ["llf", "hlle"]
_flux["hydro"] = ["llf", "hlle"]
_res = [32, 64]  # resolutions to test


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nghost=3",
        "mesh/nx1=" + repr(res),
        "mesh/nx2=1",
        "mesh/nx3=1",
        "meshblock/nx1=16",
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        "mesh_refinement/refinement=none",
        "time/cfl_number=0.4",
        "coord/special_rel=false",
        "coord/general_rel=true",
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/along_x1=true",
        "problem/amp=1.0e-6",
        "problem/wave_flag=" + wv,
    ]


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("soe", ["hydro", "mhd"])  # system of eqns to test
def test_run(rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    iv = "rk2" if rv == "plm" else "rk3"
    for fv in _flux[soe]:
        # Ignore return arguments
        _, _ = testutils.test_error_convergence(
            f"inputs/lwave_rel{soe}.athinput",
            f"gr_lwave_{soe}",
            arguments,
            maxerrors,
            _wave[soe],
            _res,
            iv,
            rv,
            fv,
            soe,
        )
