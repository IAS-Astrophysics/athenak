"""
Unit tests for orthonormal tetrad transformation
"""

# Modules
import test_suite.testutils as testutils


def test_gauss_legendre():
    input_file = "inputs/ut_tetrad.athinput"
    testutils.run(input_file)
