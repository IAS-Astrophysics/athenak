"""Circularly polarized Alfven wave convergence in 1D/2D with static AMR."""

import pytest

import athena_read
import test_suite.testutils as testutils


_CASES = [
    pytest.param("1D", "cpaw1d", 2.0e-3, 0.35, id="1d"),
    pytest.param("2D", "cpaw2d", 7.0e-3, 0.45, id="2d"),
]
_RESOLUTIONS = [32, 64]
_RMS_L1_INDEX = 4


def arguments(label, basename, resolution):
    """Return a four-block-per-active-direction CPAW mesh."""
    one_d = label == "1D"
    return [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}",
        f"mesh/nx2={1 if one_d else resolution // 2}",
        "mesh/nx3=1",
        f"meshblock/nx1={resolution // 4}",
        f"meshblock/nx2={1 if one_d else resolution // 8}",
        "meshblock/nx3=1",
        f"problem/along_x1={'true' if one_d else 'false'}",
    ]


@pytest.mark.parametrize("label,basename,max_error,max_ratio", _CASES)
def test_run(label, basename, max_error, max_ratio):
    """Check lower-dimensional CPAW accuracy and convergence across static AMR."""
    try:
        for resolution in _RESOLUTIONS:
            results = testutils.run(
                "inputs/cpaw.athinput",
                arguments(label, basename, resolution),
            )
            assert results, f"{label} CPAW run failed at resolution {resolution}."

        data = athena_read.error_dat(f"{basename}-errs.dat")
        low_error = data[0][_RMS_L1_INDEX]
        high_error = data[1][_RMS_L1_INDEX]
        ratio = high_error / low_error

        if high_error > max_error:
            pytest.fail(
                f"{label} CPAW error too large: {high_error:g}, "
                f"threshold {max_error:g}"
            )
        if ratio > max_ratio:
            pytest.fail(
                f"{label} CPAW convergence too slow: {ratio:g}, "
                f"threshold {max_ratio:g}"
            )
    finally:
        testutils.cleanup()
