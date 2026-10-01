"""
Boosted puncture test for Z4c with the BHaHAHA horizon finder enabled.
Evolves a boosted puncture for 200 time steps and checks that the horizon is found
late in the run with the expected irreducible mass.
"""

# Modules
import shutil
import numpy as np
import pytest
import test_suite.testutils as testutils

# Expected values and thresholds based on the grid in the input file. Theta is not
# checked since the loose BHaHAHA tolerances in the input file bound it already.
mirr_expected = 1.0
min_time = 2.3
maxerrors = {
    ("Mirr-rel"): (3.0e-02),
}
diag_file = "horizon/BHaHAHA_diagnostics.ah1.gp"


def arguments():
    """Assemble arguments for run command"""
    return ["job/basename=boosted_bhahaha", "z4c/horizon_finder=bhahaha"]


input_file = "inputs/z4c_boosted.athinput"


def test_run():
    """Run a single test with given arguments."""
    # BHaHAHA appends to its diagnostics file, so remove output of earlier runs
    shutil.rmtree("horizon", ignore_errors=True)
    try:
        results = testutils.run(input_file, arguments())
        assert results, "Z4c boosted puncture BHaHAHA test run failed."
        # Columns: 2 = time, 13 = irreducible mass
        data = np.atleast_2d(np.loadtxt(diag_file, comments="#"))
        time = data[-1, 1]
        mirr = data[-1, 12]
        if time < min_time:
            pytest.fail(f"Horizon not found at late times, last find at t = {time:g}")
        mirr_err = abs(mirr - mirr_expected) / mirr_expected
        if mirr_err > maxerrors["Mirr-rel"]:
            pytest.fail(
                f"Irreducible mass error too large, mass: {mirr:g} "
                f"expected: {mirr_expected:g}"
            )

    finally:
        testutils.cleanup()
        shutil.rmtree("horizon", ignore_errors=True)
