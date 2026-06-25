"""
Resistive (Ohmic) diffusion convergence test in 1D (CPU).
Diffuses a 1D Gaussian pulse in a single transverse magnetic-field component (By and Bz,
in turn) along x1, and checks 2nd-order convergence of the L1 error between two
resolutions.  This is the magnetic analogue of the 1D viscosity test.
"""

# Modules
import test_suite.testutils as testutils

# Threshold (max high-res RMS-L1 error) and max error ratio (err_hi/err_lo), keyed by
# (soe, integrator, label, mode).  "By"/"Bz" select the diffusing field component.
errors = {
    ("mhd", "rk2", "diff", "By"): (1.5e-10, 0.30),
    ("mhd", "rk2", "diff", "Bz"): (1.5e-10, 0.30),
}

_mode = ["By", "Bz"]
_res = [64, 128]

# magnetic-field component (1->Bx, 2->By, 3->Bz) carrying the pulse for each mode
_b_comp = {"By": 2, "Bz": 3}


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
        "problem/resistivity_test=true",
        "problem/spread_x1=true",
        "problem/spread_x2=false",
        "problem/spread_x3=false",
        "problem/vel_comp=" + repr(_b_comp[wv]),
        "mhd/eta_ohm=0.25",
        "problem/amp=1.0e-6",
    ]


def test_run():
    """Run the 1D resistive-diffusion convergence test for By and Bz."""
    testutils.test_error_convergence(
        "inputs/diffusion_mhd.athinput",
        "diffusion_resist1d",
        arguments,
        errors,
        _mode,
        _res,
        "rk2",
        "diff",
        "none",
        "mhd",
        left_wave="By",
        right_wave="Bz",
    )
