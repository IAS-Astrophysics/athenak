"""Regression test for HLLD with a weak, nonzero face-normal magnetic field."""

import numpy as np
import pytest

import athena_read
import test_suite.testutils as testutils


@pytest.mark.parametrize(
    ("case", "normal_velocity", "normal_field", "wave_speed"),
    [
        ("left_double_star", 0.005, 0.01, -0.005),
        ("right_double_star", -0.005, -0.01, 0.005),
    ],
)
def test_moving_rotational_discontinuity(
    case, normal_velocity, normal_field, wave_speed
):
    """Check both double-star fluxes for a weak-Bx rotational discontinuity."""
    basename = f"hlld_weak_bx_{case}"
    flags = [
        f"job/basename={basename}",
        "mesh/nx1=16",
        "meshblock/nx1=16",
        "mesh/nghost=2",
        "time/integrator=rk2",
        "time/cfl_number=0.1",
        "time/nlim=1",
        "time/tlim=1.0",
        "mhd/reconstruct=dc",
        "mhd/rsolver=hlld",
        "problem/dl=1.0",
        "problem/pl=1.0",
        f"problem/ul={normal_velocity}",
        "problem/vl=-1.0",
        "problem/wl=0.0",
        f"problem/bxl={normal_field}",
        "problem/byl=-1.0",
        "problem/bzl=0.0",
        "problem/dr=1.0",
        "problem/pr=1.0",
        f"problem/ur={normal_velocity}",
        "problem/vr=1.0",
        "problem/wr=0.0",
        f"problem/bxr={normal_field}",
        "problem/byr=1.0",
        "problem/bzr=0.0",
        "output1/variable=mhd_w_bcc",
        "output1/data_format=%24.17e",
        "output1/dt=1.0",
    ]

    try:
        assert testutils.run("inputs/rj2a.athinput", flags)
        data = athena_read.tab(f"tab/{basename}.mhd_w_bcc.00001.tab")

        dx1 = 1.0 / 16.0
        discontinuity = wave_speed * data["time"]
        left_faces = data["x1v"] - 0.5 * dx1
        left_fraction = np.clip((discontinuity - left_faces) / dx1, 0.0, 1.0)
        exact = 1.0 - 2.0 * left_fraction

        vely_error = np.max(np.abs(data["vely"] - exact))
        by_error = np.max(np.abs(data["bcc2"] - exact))
        assert vely_error < 2.0e-5
        assert by_error < 2.0e-5
        assert np.max(np.abs(data["vely"])) <= 1.0 + 1.0e-8
    finally:
        testutils.cleanup()
