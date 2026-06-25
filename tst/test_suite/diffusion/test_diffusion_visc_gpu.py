"""
Viscous-diffusion convergence test on 2D planes embedded in 3D (GPU).
For each of the xy, yz and zx planes the mesh is extended in the two in-plane directions
and only 4 cells wide in the perpendicular direction.  The diffusing velocity component
is the one parallel to that perpendicular direction (vz for xy, vx for yz, vy for zx),
which spreads isotropically within the plane.  Checks 2nd-order convergence of the L1
error between two in-plane resolutions.
"""

# Modules
import test_suite.testutils as testutils

# Threshold (max high-res RMS-L1 error) and max error ratio (err_hi/err_lo), keyed by
# (soe, integrator, label, mode).
errors = {
    ("hydro", "rk2", "diff", "xy"): (3.0e-11, 0.30),
    ("hydro", "rk2", "diff", "yz"): (3.0e-11, 0.30),
    ("hydro", "rk2", "diff", "zx"): (3.0e-11, 0.30),
}

_mode = ["xy", "yz", "zx"]
_res = [64, 128]

# Per-plane setup: spread axes, diffusing velocity component, and the thin (perpendicular)
# direction (1->x1, 2->x2, 3->x3).
_plane = {
    "xy": {"spread": (True, True, False), "vel_comp": 3, "thin": 3},
    "yz": {"spread": (False, True, True), "vel_comp": 1, "thin": 1},
    "zx": {"spread": (True, False, True), "vel_comp": 2, "thin": 2},
}
_NTHIN = 4  # cells in the perpendicular direction


def arguments(iv, rv, fv, wv, res, soe, name):
    """Assemble arguments for run command"""
    p = _plane[wv]
    sx1, sx2, sx3 = p["spread"]
    # mesh/meshblock sizes: in-plane = res (tiled by 2), perpendicular = _NTHIN
    nx = {1: res, 2: res, 3: res}
    mbx = {1: res // 2, 2: res // 2, 3: res // 2}
    nx[p["thin"]] = _NTHIN
    mbx[p["thin"]] = _NTHIN
    return [
        f"job/basename={name}",
        "time/tlim=1.0",
        "time/integrator=" + iv,
        "mesh/nx1=" + repr(nx[1]),
        "mesh/nx2=" + repr(nx[2]),
        "mesh/nx3=" + repr(nx[3]),
        "meshblock/nx1=" + repr(mbx[1]),
        "meshblock/nx2=" + repr(mbx[2]),
        "meshblock/nx3=" + repr(mbx[3]),
        "problem/viscosity_test=true",
        "problem/conduction_test=false",
        "problem/spread_x1=" + ("true" if sx1 else "false"),
        "problem/spread_x2=" + ("true" if sx2 else "false"),
        "problem/spread_x3=" + ("true" if sx3 else "false"),
        "problem/vel_comp=" + repr(p["vel_comp"]),
        "hydro/nu_iso=0.25",
        "problem/amp=1.0e-6",
    ]


def test_run():
    """Run the viscous-diffusion plane tests for xy, yz and zx."""
    testutils.test_error_convergence(
        "inputs/diffusion.athinput",
        "diffusion_visc_planes",
        arguments,
        errors,
        _mode,
        _res,
        "rk2",
        "diff",
        "none",
        "hydro",
        left_wave="xy",
        right_wave="zx",
    )
