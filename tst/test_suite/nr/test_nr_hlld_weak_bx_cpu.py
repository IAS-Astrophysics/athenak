"""Regression test for HLLD with a weak, nonzero face-normal magnetic field."""

import numpy as np

import athena_read
import test_suite.testutils as testutils


def test_moving_rotational_discontinuity():
    """Check that HLLD upwinds a weak-Bx rotational discontinuity."""
    basename = "hlld_weak_bx"
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
        "problem/ul=0.005",
        "problem/vl=-1.0",
        "problem/wl=0.0",
        "problem/bxl=0.01",
        "problem/byl=-1.0",
        "problem/bzl=0.0",
        "problem/dr=1.0",
        "problem/pr=1.0",
        "problem/ur=0.005",
        "problem/vr=1.0",
        "problem/wr=0.0",
        "problem/bxr=0.01",
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
        discontinuity = -0.005 * data["time"]
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
