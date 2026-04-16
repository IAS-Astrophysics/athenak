"""
Linear wave convergence test for non-relativistic hydro/MHD in 1D.
Runs tests in both hydro and MHD for different
  - time integrators
  - reconstruction algorithms
  - Riemann solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils

# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and wave types
errors = {
    ("hydro", "rk2", "plm", "0"): (2.1e-08, 0.28),
    ("hydro", "rk2", "ppm4", "0"): (1.7e-08, 0.35),
    ("hydro", "rk2", "ppmx", "0"): (2.1e-09, 0.26),
    ("hydro", "rk2", "wenoz", "0"): (2.2e-09, 0.26),
    ("hydro", "rk2", "plm", "4"): (2.1e-08, 0.28),
    ("hydro", "rk2", "ppm4", "4"): (1.7e-08, 0.35),
    ("hydro", "rk2", "ppmx", "4"): (2.1e-09, 0.26),
    ("hydro", "rk2", "wenoz", "4"): (2.2e-09, 0.26),
    ("hydro", "rk2", "plm", "3"): (1.2e-08, 0.29),
    ("hydro", "rk2", "ppm4", "3"): (4.1e-09, 0.29),
    ("hydro", "rk2", "ppmx", "3"): (2.5e-10, 0.3),
    ("hydro", "rk2", "wenoz", "3"): (2.6e-10, 0.26),
    ("hydro", "rk3", "plm", "0"): (1.8e-08, 0.28),
    ("hydro", "rk3", "ppm4", "0"): (4.7e-09, 0.23),
    ("hydro", "rk3", "ppmx", "0"): (3.3e-11, 0.076),
    ("hydro", "rk3", "wenoz", "0"): (2.3e-11, 0.11),
    ("hydro", "rk3", "plm", "4"): (1.8e-08, 0.28),
    ("hydro", "rk3", "ppm4", "4"): (4.7e-09, 0.23),
    ("hydro", "rk3", "ppmx", "4"): (3.3e-11, 0.076),
    ("hydro", "rk3", "wenoz", "4"): (2.3e-11, 0.11),
    ("hydro", "rk3", "plm", "3"): (1.2e-08, 0.29),
    ("hydro", "rk3", "ppm4", "3"): (3.5e-09, 0.25),
    ("hydro", "rk3", "ppmx", "3"): (1.3e-11, 0.065),
    ("hydro", "rk3", "wenoz", "3"): (2.5e-12, 0.064),
    ("mhd", "rk2", "plm", "0"): (2.5e-08, 0.28),
    ("mhd", "rk2", "ppm4", "0"): (2e-08, 0.35),
    ("mhd", "rk2", "ppmx", "0"): (2.6e-09, 0.26),
    ("mhd", "rk2", "wenoz", "0"): (2.6e-09, 0.26),
    ("mhd", "rk2", "plm", "6"): (2.5e-08, 0.28),
    ("mhd", "rk2", "ppm4", "6"): (2e-08, 0.35),
    ("mhd", "rk2", "ppmx", "6"): (2.6e-09, 0.26),
    ("mhd", "rk2", "wenoz", "6"): (2.6e-09, 0.26),
    ("mhd", "rk2", "plm", "5"): (1.7e-08, 0.29),
    ("mhd", "rk2", "ppm4", "5"): (5.8e-09, 0.26),
    ("mhd", "rk2", "ppmx", "5"): (3.5e-10, 0.3),
    ("mhd", "rk2", "wenoz", "5"): (3.7e-10, 0.26),
    ("mhd", "rk2", "plm", "1"): (1.7e-08, 0.29),
    ("mhd", "rk2", "ppm4", "1"): (5.8e-09, 0.26),
    ("mhd", "rk2", "ppmx", "1"): (3.5e-10, 0.3),
    ("mhd", "rk2", "wenoz", "1"): (3.7e-10, 0.26),
    ("mhd", "rk2", "plm", "4"): (2.8e-08, 0.32),
    ("mhd", "rk2", "ppm4", "4"): (1.5e-08, 0.54),
    ("mhd", "rk2", "ppmx", "4"): (1.6e-10, 0.7),
    ("mhd", "rk2", "wenoz", "4"): (1.1e-10, 0.26),
    ("mhd", "rk2", "plm", "2"): (2.8e-08, 0.32),
    ("mhd", "rk2", "ppm4", "2"): (1.5e-08, 0.54),
    ("mhd", "rk2", "ppmx", "2"): (1.5e-10, 0.71),
    ("mhd", "rk2", "wenoz", "2"): (1.1e-10, 0.26),
    ("mhd", "rk2", "plm", "3"): (2.2e-08, 0.3),
    ("mhd", "rk2", "ppm4", "3"): (6.2e-09, 0.27),
    ("mhd", "rk2", "ppmx", "3"): (1.6e-10, 0.4),
    ("mhd", "rk2", "wenoz", "3"): (1.8e-10, 0.26),
    ("mhd", "rk3", "plm", "0"): (2.2e-08, 0.28),
    ("mhd", "rk3", "ppm4", "0"): (7.4e-09, 0.3),
    ("mhd", "rk3", "ppmx", "0"): (1.8e-10, 0.2),
    ("mhd", "rk3", "wenoz", "0"): (1.8e-10, 0.23),
    ("mhd", "rk3", "plm", "6"): (2.2e-08, 0.28),
    ("mhd", "rk3", "ppm4", "6"): (7.4e-09, 0.3),
    ("mhd", "rk3", "ppmx", "6"): (1.8e-10, 0.2),
    ("mhd", "rk3", "wenoz", "6"): (1.8e-10, 0.23),
    ("mhd", "rk3", "plm", "5"): (1.7e-08, 0.29),
    ("mhd", "rk3", "ppm4", "5"): (5.1e-09, 0.25),
    ("mhd", "rk3", "ppmx", "5"): (1.8e-11, 0.064),
    ("mhd", "rk3", "wenoz", "5"): (3.6e-12, 0.064),
    ("mhd", "rk3", "plm", "1"): (1.7e-08, 0.29),
    ("mhd", "rk3", "ppm4", "1"): (5.1e-09, 0.25),
    ("mhd", "rk3", "ppmx", "1"): (1.8e-11, 0.064),
    ("mhd", "rk3", "wenoz", "1"): (3.6e-12, 0.064),
    ("mhd", "rk3", "plm", "4"): (2.8e-08, 0.32),
    ("mhd", "rk3", "ppm4", "4"): (8.2e-09, 0.26),
    ("mhd", "rk3", "ppmx", "4"): (2e-11, 0.064),
    ("mhd", "rk3", "wenoz", "4"): (4.9e-12, 0.1),
    ("mhd", "rk3", "plm", "2"): (2.8e-08, 0.32),
    ("mhd", "rk3", "ppm4", "2"): (8.2e-09, 0.26),
    ("mhd", "rk3", "ppmx", "2"): (2e-11, 0.064),
    ("mhd", "rk3", "wenoz", "2"): (4.9e-12, 0.1),
    ("mhd", "rk3", "plm", "3"): (2.2e-08, 0.3),
    ("mhd", "rk3", "ppm4", "3"): (6e-09, 0.26),
    ("mhd", "rk3", "ppmx", "3"): (1.9e-11, 0.066),
    ("mhd", "rk3", "wenoz", "3"): (3.4e-12, 0.045),
}

_int = ["rk2", "rk3"]
_recon = ["plm", "ppm4", "ppmx", "wenoz"]
_wave = {}
_wave["mhd"] = ["0", "6", "5", "1", "4", "2", "3"]
_wave["hydro"] = ["0", "4", "3"]
_flux = {}
_flux["mhd"] = ["llf", "hlle", "hlld"]
_flux["hydro"] = ["llf", "hlle", "hllc", "roe"]
_res = [32, 64]  # resolutions to test


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    vx0 = 1.0 if wv == "3" else 0.0
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
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/along_x1=true",
        "problem/amp=1.0e-6",
        "problem/wave_flag=" + wv,
        "problem/vx0=" + repr(vx0),
    ]


@pytest.mark.parametrize("iv", _int)
@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("soe", ["hydro", "mhd"])  # system of eqns to test
def test_run(iv, rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    for fv in _flux[soe]:
        # returns error in L/R wave specified by 'left_wave'/'right_wave' arguments
        l1_rms_l, l1_rms_r = testutils.test_error_convergence(
            f"inputs/lwave_{soe}.athinput",
            f"lwave1d_{soe}",
            arguments,
            errors,
            _wave[soe],
            _res,
            iv,
            rv,
            fv,
            soe,
            left_wave="0",
            right_wave="4" if soe == "hydro" else "6",
        )
        # test that errors in L/R-going waves are the same
        if l1_rms_l != l1_rms_r and rv == "plm":
            pytest.fail(
                f"Errors in L/R-going waves not equal for {soe}+{iv}+{rv}+{fv}, "
                f"L: {l1_rms_l:g} R: {l1_rms_r:g}"
            )
