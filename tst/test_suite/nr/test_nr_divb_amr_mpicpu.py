"""
Test face-centered div(B) preservation under moving AMR with MPI.

The divb_amr problem initializes B from a discrete vector potential and forces a
moving AMR refinement pattern. The 2D and deep-3D cases exercise repeated
refinement and derefinement across ranks.
"""

import math
import os

import pytest
import test_suite.testutils as testutils
import athena_read


_CASES = [
    pytest.param(
        "2D",
        "inputs/divb_amr_2d.athinput",
        "DivBAMR2D",
        48 * 48,
        20,
        id="2d",
    ),
    # Two physical refinement levels are the minimum 3D case that exposes the bug.
    pytest.param(
        "3D-L2",
        "inputs/divb_amr_3d.athinput",
        "DivBAMR3DL2",
        32 * 32 * 32,
        8,
        id="3d-l2",
    ),
]

_MAX_NDIV_TOL = 2.0e-11
_L1_NDIV_TOL = 2.0e-12
_L2_NDIV_TOL = 5.0e-12


@pytest.mark.parametrize("label,input_file,basename,root_ncell,min_rows", _CASES)
def test_run(label, input_file, basename, root_ncell, min_rows):
    """Run one moving-AMR case and check its face-centered divergence history."""
    history_file = f"{basename}.user.hst"
    try:
        if os.path.exists(history_file):
            os.remove(history_file)

        results = testutils.mpi_run(
            input_file,
            [f"job/basename={basename}"],
            threads=8,
        )
        assert results, f"{label} AMR div(B) test run failed."

        data = athena_read.hst(history_file)
        max_ndiv = max(data["max_ndiv"])
        l1_ndiv = max(s / v for s, v in zip(data["sum_ndiv"], data["vol"]))
        l2_ndiv = max(
            math.sqrt(s / v) for s, v in zip(data["sum_n2"], data["vol"])
        )
        max_ncell = max(data["ncell"])

        if len(data["time"]) < min_rows:
            pytest.fail(
                f"{label} AMR div(B) history is too short: "
                f"{len(data['time'])} rows, expected {min_rows}"
            )
        if max_ncell <= root_ncell:
            pytest.fail(
                f"{label} AMR div(B) test did not refine: "
                f"max_ncell={max_ncell:g}, root_ncell={root_ncell:g}"
            )
        if max_ndiv > _MAX_NDIV_TOL:
            pytest.fail(
                f"{label} max normalized div(B) too large: "
                f"{max_ndiv:g} threshold {_MAX_NDIV_TOL:g}"
            )
        if l1_ndiv > _L1_NDIV_TOL:
            pytest.fail(
                f"{label} L1 normalized div(B) too large: "
                f"{l1_ndiv:g} threshold {_L1_NDIV_TOL:g}"
            )
        if l2_ndiv > _L2_NDIV_TOL:
            pytest.fail(
                f"{label} L2 normalized div(B) too large: "
                f"{l2_ndiv:g} threshold {_L2_NDIV_TOL:g}"
            )
    finally:
        if os.path.exists(history_file):
            os.remove(history_file)
        testutils.cleanup()
