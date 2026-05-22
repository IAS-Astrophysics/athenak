"""CPU regression for the dynbbh superposed Kerr-Schild metric."""

import os
import sys

import test_suite.testutils as testutils

sys.path.append(os.path.dirname(__file__))
from dynbbh_metric_common import run_convergence_check, run_stress_suite  # noqa: E402


def test_stress():
    try:
        run_stress_suite()
    finally:
        testutils.cleanup()


def test_convergence():
    try:
        run_convergence_check()
    finally:
        testutils.cleanup()
