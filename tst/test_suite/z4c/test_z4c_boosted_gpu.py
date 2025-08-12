"""
Boosted puncture test for Z4c.
Evolves a boosted puncture for 200 time steps and checks all contraints are satisfied
to within error tolerances
"""

# Modules
import sys
sys.path.insert(0, '../vis/python')
sys.path.insert(0, '../test_suite')
import pytest
import test_suite.testutils as testutils
import scripts.utils.athena as athena
import athena_read
import numpy as np

# Threshold errors for constraints based on 6th-order evolution with specific grid in input file
maxerrors={
    ('C-norm'): (1.85e-02),
    ('H-norm'): (4.7e-03),
    ('M-norm'): (1.4e-03),
    ('Z-norm'): (3.2e-03),
    ('Mx-norm'): (1.1e-03),
    ('My-norm'): (4.5e-04),
    ('Mz-norm'): (4.5e-04),
    ('Theta-norm'): (3.1e-05)
}

def arguments():
    """Assemble arguments for run command"""
    return [f'job/basename=boosted']

input_file = "inputs/z4c_boosted.athinput"
def test_run():
    """Run a single test with given arguments."""
    try:
        results = testutils.athenak_run(input_file, arguments())
        assert results, f"Z4c boosted puncture test run failed."
        # Check constraints in the history file
        data = athena_read.hst(f'boosted.z4c.user.hst')
        cnorm = data['C-norm2'][3]
        hnorm = data['H-norm2'][3]
        mnorm = data['M-norm2'][3]
        znorm = data['Z-norm2'][3]
        mxnorm = data['Mx-norm2'][3]
        mynorm = data['My-norm2'][3]
        mznorm = data['Mz-norm2'][3]
        tnorm = data['Theta-norm'][3]
        if cnorm > maxerrors['C-norm']:
            pytest.fail(f"C-norm error too large, error: {cnorm:g} threshold: {maxerrors['C-norm']:g}")
        if hnorm > maxerrors['H-norm']:
            pytest.fail(f"H-norm error too large, error: {hnorm:g} threshold: {maxerrors['H-norm']:g}")
        if mnorm > maxerrors['M-norm']:
            pytest.fail(f"M-norm error too large, error: {mnorm:g} threshold: {maxerrors['M-norm']:g}")
        if znorm > maxerrors['Z-norm']:
            pytest.fail(f"Z-norm error too large, error: {znorm:g} threshold: {maxerrors['Z-norm']:g}")
        if mxnorm > maxerrors['Mx-norm']:
            pytest.fail(f"Mx-norm error too large, error: {mxnorm:g} threshold: {maxerrors['Mx-norm']:g}")
        if mynorm > maxerrors['My-norm']:
            pytest.fail(f"My-norm error too large, error: {mynorm:g} threshold: {maxerrors['My-norm']:g}")
        if mznorm > maxerrors['Mz-norm']:
            pytest.fail(f"Mz-norm error too large, error: {mznorm:g} threshold: {maxerrors['Mz-norm']:g}")
        if tnorm > maxerrors['Theta-norm']:
            pytest.fail(f"Theta-norm error too large, error: {tnorm:g} threshold: {maxerrors['Theta-norm']:g}")


    finally:
        testutils.cleanup()
