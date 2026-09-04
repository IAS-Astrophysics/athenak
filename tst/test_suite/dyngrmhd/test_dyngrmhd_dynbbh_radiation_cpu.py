"""CPU regression for dynbbh radiation transport."""

import os
import sys

import test_suite.testutils as testutils

sys.path.append(os.path.dirname(__file__))
from dynbbh_radiation_common import run_radiation_regression  # noqa: E402


def test_dynbbh_radiation():
    try:
        run_radiation_regression()
    finally:
        testutils.cleanup()
