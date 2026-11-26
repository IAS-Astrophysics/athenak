"""
Unit tests for the pointwise kernels of the grey M1 solver.
"""

# Modules
import test_suite.testutils as testutils


def test_rad_m1_core():
    input_file = "inputs/ut_rad_m1_core.athinput"
    assert testutils.run(input_file), "rad_m1_core unit tests failed"
