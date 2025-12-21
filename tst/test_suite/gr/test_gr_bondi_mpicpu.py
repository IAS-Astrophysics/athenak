"""
GR Bondi hydrodynamic accretion test.
Runs test in 3D with SMR and checks L1 error is below bound.  Errors are computed by
the executable automatically and stored in the temporary file gr_bondi-errs.dat.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read


def arguments():
    """Assemble arguments for run command"""
    return [
        "job/basename=gr_bondi",
        "time/tlim=50.0",
        "time/integrator=rk2",
        "hydro/reconstruct=plm",
    ]


input_file = "inputs/gr_bondi.athinput"


def test_run():
    """Run a single test with given arguments."""
    try:
        results = testutils.mpi_run(input_file, arguments())
        assert results, "GR Bondi test run failed."
        # Check the errors in the output
        data = athena_read.error_dat("gr_bondi-errs.dat")
        L1_RMS_INDEX = 4  # Index for L1 RMS error in data
        l1_rms = data[0][L1_RMS_INDEX]
        maxerror = 2.5e-6

        if l1_rms > maxerror:
            pytest.fail(
                f"L1 error too large, error: {l1_rms:g} threshold: {maxerror:g}"
            )
    finally:
        testutils.cleanup()
