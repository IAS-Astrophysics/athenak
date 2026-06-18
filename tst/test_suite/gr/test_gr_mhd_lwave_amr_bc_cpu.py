"""
Regression test for the fine/coarse boundary fix on the MHD side.

A GR-MHD linear wave (amp=1e-3) is run on a domain with OUTFLOW boundaries and AMR. The
min_max criterion refines the wave crest, dragging a fine/coarse boundary onto the outflow
(physical) boundary. The C2P failures are tracked from the corresponding .log file.
"""

import os

import test_suite.testutils as testutils

EOS_FAIL_TOL = 0   # max per-cycle hard C2P failures allowed
_LOG = "lwave_mhd_bc.log"  # = <job>/basename + ".log"


def _parse_event_log(path):
    """Return (max eos_fail over cycles)."""
    eos_fail_max = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            eos_fail_max = max(eos_fail_max, int(cols[5]))
    return eos_fail_max


def test_mhd_lwave_amr_boundary():
    """Run the GR-MHD outflow+AMR linear wave and assert no boundary C2P trouble."""
    try:
        testutils.run("inputs/lwave_mhd_bc.athinput", [])
        assert os.path.exists(_LOG), f"event-counter log {_LOG} not produced"
        eos_fail_max = _parse_event_log(_LOG)
        assert eos_fail_max <= EOS_FAIL_TOL, (
            f"GR-MHD C2P failures at the refined outflow boundary: max eos_fail="
            f"{eos_fail_max} (tol {EOS_FAIL_TOL}). The coarse-array physical-BC fix in "
            f"MHD::Prolongate is likely missing or broken."
        )
    finally:
        if os.path.exists(_LOG):
            os.remove(_LOG)
        testutils.cleanup()
