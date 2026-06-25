"""
Thermal-conduction convergence tests in 1D and 2D (CPU).
Diffuses an isotropic Gaussian temperature pulse with the gaussian-pulse diffusion pgen
and checks that the L1 error converges at (close to) 2nd order between two resolutions, in
the same style as the linear-wave tests.  The "1d"/"2d" modes select the number of axes
along which the pulse spreads.
"""

# Modules
import test_suite.testutils as testutils

# Threshold (max high-res RMS-L1 error) and max error ratio (err_hi/err_lo), keyed by
# (soe, integrator, label, mode).
errors = {
    ("hydro", "rk2", "diff", "1d"): (6.0e-10, 0.30),
    ("hydro", "rk2", "diff", "2d"): (1.5e-10, 0.30),
}

_mode = ["1d", "2d"]
_res = [32, 64]

# number of axes the Gaussian spreads along, per mode
_ndim = {"1d": 1, "2d": 2}


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    ndim = _ndim[wv]
    nx2 = res if ndim >= 2 else 1
    mbx2 = res // 2 if ndim >= 2 else 1
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nx1=" + repr(res),
        "mesh/nx2=" + repr(nx2),
        "mesh/nx3=1",
        "meshblock/nx1=" + repr(res // 2),
        "meshblock/nx2=" + repr(mbx2),
        "meshblock/nx3=1",
        "problem/conduction_test=true",
        "problem/viscosity_test=false",
        "problem/spread_x1=true",
        "problem/spread_x2=" + ("true" if ndim >= 2 else "false"),
        "problem/spread_x3=false",
        "hydro/alpha_iso=0.5",
        "problem/amp=1.0e-6",
    ]


def test_run():
    """Run the 1D and 2D thermal-conduction convergence tests."""
    testutils.test_error_convergence(
        "inputs/diffusion.athinput",
        "diffusion_heat",
        arguments,
        errors,
        _mode,
        _res,
        "rk2",
        "diff",
        "none",
        "hydro",
        left_wave="1d",
        right_wave="2d",
    )
