"""CPU convergence, coupling, and beam validation for ADM dynamic radiation."""

import os
import sys

import test_suite.testutils as testutils

sys.path.append(os.path.dirname(__file__))
from dynrad_validation_common import run_all  # noqa: E402


def test_dynrad_validation():
    try:
        run_all()
    finally:
        testutils.cleanup()
