"""
Ambipolar-diffusion convergence test on 2D planes embedded in 3D (GPU).

For each of the xy, yz and zx planes the mesh is extended in the two in-plane directions
and only 4 cells wide in the perpendicular direction.  The diffusing magnetic-field
component is the one parallel to that perpendicular direction (Bz for xy, Bx for yz, By for
zx), placed on top of a strong uniform guide field in the SAME component.  The total field
is then essentially single-component, so J = curl(B) is perpendicular to B (J.B=0) and
|B|^2 ~ b_guide^2: the ambipolar EMF reduces to LINEAR, isotropic in-plane diffusion of the
pulse with D = eta_ad*b_guide^2, and the component remains divergence-free (uniform along
its own axis).  This is the ambipolar analogue of the resistivity plane test.

As in the CPU test, we check the L1 error of the diffusing magnetic-field component
(B1/B2/B3 column of -errs.dat), not the combined RMS-L1, because the ambipolar Poynting
flux redistributes energy at O(amp) and would otherwise dominate the metric.
"""

import os
import sys

import test_suite.testutils as testutils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

# Per-plane threshold (max high-res B-field L1 error) and max error ratio (err_hi/err_lo).
# Realized: B-error N=128 ~ 1.6e-11, ratio ~ 0.25 (2nd order) for all three planes.
errors = {
    "xy": (4.0e-11, 0.30),
    "yz": (4.0e-11, 0.30),
    "zx": (4.0e-11, 0.30),
}

_mode = ["xy", "yz", "zx"]
_res = [64, 128]

# Per-plane setup: spread axes, diffusing/guide B component (1->Bx, 2->By, 3->Bz), the thin
# (perpendicular) direction (1->x1, 2->x2, 3->x3), and the -errs.dat column for that
# component's L1 error (B1->11, B2->12, B3->13).
_plane = {
    "xy": {"spread": (True, True, False), "b_comp": 3, "thin": 3, "col": 13},
    "yz": {"spread": (False, True, True), "b_comp": 1, "thin": 1, "col": 11},
    "zx": {"spread": (True, False, True), "b_comp": 2, "thin": 2, "col": 12},
}
_NTHIN = 4  # cells in the perpendicular direction
_TEST_NAME = "diffusion_ambipolar_planes"


def arguments(wv, res):
    """Assemble arguments for run command."""
    p = _plane[wv]
    sx1, sx2, sx3 = p["spread"]
    nx = {1: res, 2: res, 3: res}
    mbx = {1: res // 2, 2: res // 2, 3: res // 2}
    nx[p["thin"]] = _NTHIN
    mbx[p["thin"]] = _NTHIN
    return [
        f"job/basename={_TEST_NAME}",
        "time/tlim=1.0",
        "time/integrator=rk2",
        "mesh/nx1=" + repr(nx[1]),
        "mesh/nx2=" + repr(nx[2]),
        "mesh/nx3=" + repr(nx[3]),
        "meshblock/nx1=" + repr(mbx[1]),
        "meshblock/nx2=" + repr(mbx[2]),
        "meshblock/nx3=" + repr(mbx[3]),
        "problem/resistivity_test=false",
        "problem/ambipolar_test=true",
        "problem/spread_x1=" + ("true" if sx1 else "false"),
        "problem/spread_x2=" + ("true" if sx2 else "false"),
        "problem/spread_x3=" + ("true" if sx3 else "false"),
        "problem/vel_comp=" + repr(p["b_comp"]),
        "mhd/eta_ad=0.25",
        "problem/b_guide=1.0",
        "problem/amp=1.0e-6",
    ]


def test_run():
    """Run the ambipolar-diffusion plane tests for xy, yz and zx."""
    for wv in _mode:
        maxerror, maxratio = errors[wv]
        col = _plane[wv]["col"]
        try:
            for res in _res:
                ok = testutils.run(
                    "inputs/diffusion_ambipolar.athinput", arguments(wv, res)
                )
                assert ok, f"ambipolar {wv} plane run failed at res={res}"
            data = athena_read.error_dat(f"{_TEST_NAME}-errs.dat")
            err_lo = data[0][col]
            err_hi = data[1][col]
            assert err_hi < maxerror, (
                f"{wv} ambipolar B-field error too large: "
                f"{err_hi:g} >= threshold {maxerror:g}"
            )
            ratio = err_hi / err_lo
            assert ratio < maxratio, (
                f"{wv} ambipolar not converging (2nd order): "
                f"ratio {ratio:g} >= threshold {maxratio:g}"
            )
        finally:
            testutils.cleanup()
