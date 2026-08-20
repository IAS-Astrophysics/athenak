"""GPU regression for dynbbh metric values and derivatives."""

import os
import sys

import test_suite.testutils as testutils

sys.path.append(os.path.dirname(__file__))
from dynbbh_metric_common import (  # noqa: E402
    run_diagnostic_failure_checks, run_fd_convergence, run_regression_suite,
    run_surface_check,
    run_volume_diagnostics_check,
)


def test_metric_regression():
    try:
        run_regression_suite()
        run_fd_convergence()
        run_surface_check()
        run_volume_diagnostics_check()
        run_diagnostic_failure_checks()
    finally:
        testutils.cleanup()
