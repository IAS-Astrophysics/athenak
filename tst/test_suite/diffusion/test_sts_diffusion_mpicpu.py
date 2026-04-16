"""
Narrow MPI regression coverage for STS diffusion.
"""

import test_suite.testutils as testutils


def test_mhd_resistivity_sts_mpicpu():
    try:
        result = testutils.mpi_run(
            "../../../inputs/tests/sts_resistivity.athinput",
            [
                "job/basename=sts_resist_mpi",
                "time/sts_integrator=rkl2",
                "mhd/ohmic_resistivity_integrator=sts",
                "time/nlim=1",
            ],
            threads=4,
        )
        assert result, "MPI MHD resistive STS smoke failed."
    finally:
        testutils.cleanup()
