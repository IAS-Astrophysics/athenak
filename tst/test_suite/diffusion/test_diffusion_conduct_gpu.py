"""
Thermal-conduction convergence test in 3D (GPU).
Diffuses an isotropic 3D Gaussian temperature pulse (spreading along x1, x2 and x3) and
checks that the L1 error converges at (close to) 2nd order between two resolutions.
"""

# Modules
import test_suite.testutils as testutils

# Threshold (max high-res RMS-L1 error) and max error ratio (err_hi/err_lo), keyed by
# (soe, integrator, label, mode).
errors = {
    ("hydro", "rk2", "diff", "heat"): (3.0e-11, 0.30),
}

_mode = ["heat"]
_res = [32, 64]


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nx1=" + repr(res),
        "mesh/nx2=" + repr(res),
        "mesh/nx3=" + repr(res),
        "meshblock/nx1=" + repr(res // 2),
        "meshblock/nx2=" + repr(res // 2),
        "meshblock/nx3=" + repr(res // 2),
        "problem/conduction_test=true",
        "problem/viscosity_test=false",
        "problem/spread_x1=true",
        "problem/spread_x2=true",
        "problem/spread_x3=true",
        "hydro/alpha_iso=0.5",
        "problem/amp=1.0e-6",
    ]


"""
Following uses test_error_convergence() function in testutils.py, written for linear wave
convergence problems. Runs tests using _mode as wave flag.
"""


def test_run():
    """Run the 3D thermal-conduction convergence test."""
    testutils.test_error_convergence(
        "inputs/diffusion.athinput",
        "diffusion_heat3d",
        arguments,
        errors,
        _mode,
        _res,
        "rk2",
        "diff",
        "none",
        "hydro",
    )
