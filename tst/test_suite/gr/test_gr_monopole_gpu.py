"""
GR Monopole BZ field rotation rates test.
Runs a 3D GRMHD simulation for a spinning black hole and measures the field rotation
rate at the black hole event horizon (exploiting the SphericalGrid infrastructure).
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np


def arguments():
    """Assemble arguments for run command"""
    return [
        "job/basename=monopole",
        "time/tlim=10.0",
        "time/integrator=rk2",
        "mhd/reconstruct=plm",
    ]


input_file = "inputs/gr_monopole.athinput"


def test_run():
    """Run a single test with given arguments."""
    try:
        results = testutils.run(input_file, arguments())
        assert results, "GR Monopole test run failed."
        # Check the errors in the output
        data = athena_read.error_dat("monopole-diag.dat")
        omega = list(zip(*data))[2]
        omega_error = np.abs(np.average(omega) - 0.5)/0.5
        omega_std = np.std(omega)
        error_threshold = 0.03
        std_threshold = 0.03

        if (omega_error > error_threshold):
            pytest.fail(f"Rotation rate error too large, "
                        f"error: {omega_error:g} threshold: {error_threshold:g}")
        if (omega_std > std_threshold):
            pytest.fail(f"Rotation rate standard deviation too large, "
                        f"std: {omega_std:g} threshold: {std_threshold:g}")
    finally:
        testutils.cleanup()
