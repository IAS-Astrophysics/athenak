"""
Viscous-diffusion convergence test in 1D (CPU).
Diffuses a 1D Gaussian pulse in a single transverse velocity component (vy and vz, in
turn) along x1, and checks 2nd-order convergence of the L1 error between two resolutions.
"""

# Modules
import test_suite.testutils as testutils

# Threshold (max high-res RMS-L1 error) and max error ratio (err_hi/err_lo), keyed by
# (soe, integrator, label, mode).  "vy"/"vz" select the diffusing velocity component.
errors = {
    ("hydro", "rk2", "diff", "vy"): (1.5e-10, 0.30),
    ("hydro", "rk2", "diff", "vz"): (1.5e-10, 0.30),
}

_mode = ["vy", "vz"]
_res = [64, 128]

# velocity component (1/2/3) carrying the pulse for each mode
_vel_comp = {"vy": 2, "vz": 3}


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nx1=" + repr(res),
        "mesh/nx2=1",
        "mesh/nx3=1",
        "meshblock/nx1=" + repr(res // 2),
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        "problem/viscosity_test=true",
        "problem/conduction_test=false",
        "problem/spread_x1=true",
        "problem/spread_x2=false",
        "problem/spread_x3=false",
        "problem/vel_comp=" + repr(_vel_comp[wv]),
        "hydro/nu_iso=0.25",
        "problem/amp=1.0e-6",
    ]


"""
Following uses test_error_convergence() function in testutils.py, written for linear wave
convergence problems. Runs vy/vz tests using _mode as wave flag.
"""


def test_run():
    """Run the 1D viscous-diffusion convergence test for vy and vz."""
    testutils.test_error_convergence(
        "inputs/diffusion.athinput",
        "diffusion_visc1d",
        arguments,
        errors,
        _mode,
        _res,
        "rk2",
        "diff",
        "none",
        "hydro",
    )
