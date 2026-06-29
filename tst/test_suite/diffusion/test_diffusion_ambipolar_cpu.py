"""
Ambipolar-diffusion convergence test in 1D (CPU).

Diffuses a 1D Gaussian pulse in a single transverse magnetic-field component (By and Bz,
in turn) along x1.  The pulse rides on a strong uniform guide field in the SAME component,
so the total field is essentially single-component: J = curl(B) is perpendicular to B
(J.B=0) and |B|^2 ~ b_guide^2.  The ambipolar EMF eta_ad*[B^2 J - (J.B)B] then reduces to
LINEAR diffusion of the pulse with diffusivity D = eta_ad*b_guide^2, giving the same
spreading-Gaussian analytic solution as the Ohmic (resistivity) test.

Unlike the Ohmic test we check the L1 error of the diffusing *magnetic-field component*
(B2_L1 / B3_L1 column of the -errs.dat file), NOT the combined RMS-L1.  The ambipolar
Poynting flux physically redistributes energy at the O(amp) level, which the static
Gaussian analytic does not model, so it dominates RMS-L1 without reflecting the accuracy of
the field-diffusion operator under test.  The B-field error is the clean, converging metric.
"""

import os
import sys

import pytest

import test_suite.testutils as testutils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

# Per-mode threshold (max high-res B-field L1 error) and max error ratio (err_hi/err_lo).
# With eta_ad=0.25 and b_guide=1.0 the diffusivity D=0.25 matches the resistivity test.
# Realized: B-error N=128 ~ 7.6e-11, ratio (128/64) ~ 0.25 (2nd order).
errors = {
    "By": (2.0e-10, 0.30),
    "Bz": (2.0e-10, 0.30),
}

_mode = ["By", "Bz"]
_res = [64, 128]

# magnetic-field component (1->Bx, 2->By, 3->Bz) carrying the pulse (and the guide), and
# the column of the -errs.dat file holding that component's L1 error.
#   columns: 0:Nx1 1:Nx2 2:Nx3 3:Ncycle 4:RMS-L1 5:L-inf 6:d 7:M1 8:M2 9:M3 10:E
#            11:B1 12:B2 13:B3
_b_comp = {"By": 2, "Bz": 3}
_b_col = {"By": 12, "Bz": 13}

_TEST_NAME = "diffusion_ambipolar1d"


def arguments(wv, res):
    """Assemble arguments for run command."""
    return [
        f"job/basename={_TEST_NAME}",
        "time/tlim=1.0",
        "time/integrator=rk2",
        "mesh/nx1=" + repr(res),
        "mesh/nx2=1",
        "mesh/nx3=1",
        "meshblock/nx1=" + repr(res // 2),
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        "problem/resistivity_test=false",
        "problem/ambipolar_test=true",
        "problem/spread_x1=true",
        "problem/spread_x2=false",
        "problem/spread_x3=false",
        "problem/vel_comp=" + repr(_b_comp[wv]),
        "mhd/eta_ad=0.25",
        "problem/b_guide=1.0",
        "problem/amp=1.0e-6",
    ]


def test_run():
    """Run the 1D ambipolar-diffusion convergence test for By and Bz."""
    for wv in _mode:
        maxerror, maxratio = errors[wv]
        col = _b_col[wv]
        try:
            for res in _res:
                ok = testutils.run(
                    "inputs/diffusion_ambipolar.athinput", arguments(wv, res)
                )
                assert ok, f"ambipolar {wv} run failed at res={res}"
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
