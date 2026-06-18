"""
Regression test for the fine/coarse boundary fix on the Z4c side.

A Z4c linear wave is run on a domain with OUTFLOW boundaries and AMR, so a fine/coarse
boundary coincides with a physical boundary. This test checks the L-infty norm, which
should stay below a certain threshold with the boundary fix on.
"""

import os

import test_suite.testutils as testutils
import athena_read

LINF_TOL = 1.0e-12
_LINF_COL = 5
_ERRS = "lwave_z4c_bc-errs.dat"


def test_z4c_lwave_amr_boundary():
    """Run the Z4c outflow+AMR linear wave and assert the L-infty error stays small."""
    try:
        testutils.run("inputs/lwave_z4c_bc.athinput", [])
        assert os.path.exists(_ERRS), f"error file {_ERRS} not produced"
        data = athena_read.error_dat(_ERRS)
        linf = data[0][_LINF_COL]
        assert linf < LINF_TOL, (
            f"Z4c linear-wave L-infty error={linf:g} exceeds tol {LINF_TOL:g} "
            f"(no-fix reference ~1e-9). The coarse-array physical-BC fix in "
            f"Z4c::Prolongate is likely missing or broken."
        )
    finally:
        testutils.cleanup()
