"""
Unit tests for EOSCompOSE
"""

# Modules
import pytest
import test_suite.testutils as testutils
import numpy as np


def test_gauss_legendre():
   input_file = "inputs/ut_gauss_legendre.athinput"
   testutils.run(input_file)
