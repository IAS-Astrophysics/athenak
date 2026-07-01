"""
Unit tests for EOSCompOSE
"""

# Modules
import test_suite.testutils as testutils


def test_gauss_legendre():
    input_file = "inputs/ut_gauss_legendre.athinput"
    testutils.run(input_file)
