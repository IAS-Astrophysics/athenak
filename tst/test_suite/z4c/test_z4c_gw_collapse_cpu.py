"""Smoke test for the custom Z4c gravitational-wave-collapse pgen."""

import math
import os

import athena_read
import pytest
import test_suite.testutils as testutils


input_file = "inputs/z4c_gw_collapse.athinput"
pytestmark = pytest.mark.skipif(
    os.environ.get("ATHENAK_GW_COLLAPSE_PGEN") != "1",
    reason="requires AthenaK configured with -DPROBLEM=z4c_gravitational_wave_collapse",
)


def test_gw_collapse_smoke():
    try:
        assert testutils.run(input_file, []), "GW collapse pgen smoke run failed."
        data = athena_read.hst("gwcollapse_smoke.z4c.user.hst")
        for name in ("H-norm2", "M-norm2", "Theta-norm"):
            assert name in data, f"Missing history diagnostic {name}."
            assert math.isfinite(data[name][-1]), f"Non-finite {name}."
        assert data["H-norm2"][-1] < 1.0e-4
        assert data["M-norm2"][-1] < 1.0e-4
    finally:
        testutils.cleanup()
