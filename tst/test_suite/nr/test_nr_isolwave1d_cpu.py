"""
Linear wave convergence test for non-relativistic isothermal hydro/MHD in 1D.
Runs tests in both hydro and MHD using RK3 for different
  - reconstruction algorithms
  - Riemann solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils


# Threshold errors and error ratios for different integrators, reconstruction,
# algorithms, and waves
errors = {
    ("hydro", "rk2", "plm", "0"): (1.5e-08, 0.28),
    ("hydro", "rk3", "ppm4", "0"): (3.2e-09, 0.23),
    ("hydro", "rk3", "ppmx", "0"): (2.3e-11, 0.077),
    ("hydro", "rk3", "wenoz", "0"): (1.6e-11, 0.11),
    ("hydro", "rk2", "plm", "3"): (1.5e-08, 0.28),
    ("hydro", "rk3", "ppm4", "3"): (3.2e-09, 0.23),
    ("hydro", "rk3", "ppmx", "3"): (2.3e-11, 0.077),
    ("hydro", "rk3", "wenoz", "3"): (1.6e-11, 0.11),
    ("mhd", "rk2", "plm", "0"): (1.5e-08, 0.28),
    ("mhd", "rk3", "ppm4", "0"): (4.3e-09, 0.3),
    ("mhd", "rk3", "ppmx", "0"): (1.5e-10, 0.23),
    ("mhd", "rk3", "wenoz", "0"): (1.5e-10, 0.25),
    ("mhd", "rk2", "plm", "5"): (1.5e-08, 0.28),
    ("mhd", "rk3", "ppm4", "5"): (4.3e-09, 0.3),
    ("mhd", "rk3", "ppmx", "5"): (1.5e-10, 0.23),
    ("mhd", "rk3", "wenoz", "5"): (1.5e-10, 0.25),
    ("mhd", "rk2", "plm", "1"): (1.7e-08, 0.29),
    ("mhd", "rk3", "ppm4", "1"): (5.1e-09, 0.25),
    ("mhd", "rk3", "ppmx", "1"): (1.8e-11, 0.064),
    ("mhd", "rk3", "wenoz", "1"): (3.6e-12, 0.064),
    ("mhd", "rk2", "plm", "4"): (1.7e-08, 0.29),
    ("mhd", "rk3", "ppm4", "4"): (5.1e-09, 0.25),
    ("mhd", "rk3", "ppmx", "4"): (1.8e-11, 0.064),
    ("mhd", "rk3", "wenoz", "4"): (3.6e-12, 0.064),
    ("mhd", "rk2", "plm", "2"): (2.5e-08, 0.32),
    ("mhd", "rk3", "ppm4", "2"): (7.3e-09, 0.28),
    ("mhd", "rk3", "ppmx", "2"): (1.8e-11, 0.064),
    ("mhd", "rk3", "wenoz", "2"): (4e-12, 0.056),
    ("mhd", "rk2", "plm", "3"): (2.5e-08, 0.32),
    ("mhd", "rk3", "ppm4", "3"): (7.3e-09, 0.28),
    ("mhd", "rk3", "ppmx", "3"): (1.8e-11, 0.064),
    ("mhd", "rk3", "wenoz", "3"): (4e-12, 0.056),
}

_int = ["rk3"]
_recon = ["plm", "ppm4", "ppmx", "wenoz"]
_wave = {}
_wave["hydro"] = ["0", "3"]
_wave["mhd"] = ["0", "5", "1", "4", "2", "3"]
_flux = {}
_flux["mhd"] = ["llf", "hlle", "hlld"]
_flux["hydro"] = ["llf", "hlle", "roe"]
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
        f"{soe}/eos=isothermal",
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
        "problem/along_x1=true",
        "problem/amp=1.0e-6",
        "problem/wave_flag=" + wv,
    ]


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("soe", ["hydro", "mhd"])
def test_run(rv, soe):
    """Loop over Riemann solvers and run test with given integrator/resolution/physics."""
    iv = "rk2" if rv == "plm" else "rk3"
    for fv in _flux[soe]:
        # returns error in L/R wave specified by 'left_wave'/'right_wave' arguments
        l1_rms_l, l1_rms_r = testutils.test_error_convergence(
            f"inputs/lwave_{soe}.athinput",
            f"lwave1d_iso{soe}",
            arguments,
            errors,
            _wave[soe],
            _res,
            iv,
            rv,
            fv,
            soe,
            left_wave="0",
            right_wave="3" if soe == "hydro" else "5",
        )
        # Test that errors in L/R going waves are the same
        if l1_rms_l != l1_rms_r and rv == "plm":
            pytest.fail(
                f"Errors in L/R-going waves not equal for {soe}+{iv}+{rv}+{fv}, "
                f"L: {l1_rms_l:g} R: {l1_rms_r:g}"
            )
