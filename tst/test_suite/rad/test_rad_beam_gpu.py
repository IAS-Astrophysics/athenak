"""
Radiation beam with AMR + tetrad test

rad_beam problem generator first tests tetrad forms orthonormal basis in CKS for BH with
spin a = -0.9.

Then it runs a propagating beam test with the source at the photon ring orbit. This script
reads a 1D x2-slice at x1=2.4 of the final result to check that AMR has followed the
curved beam properly.
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read


input_file = "inputs/rad_beam.athinput"


# run test
def test_run():
    """Run a single test."""
    try:
        results = testutils.run(input_file)
        assert results, "3D AMR radiation beam test run failed"
        data = athena_read.tab("tab/beam_cks.rad_coord.00001.tab")
        # test if AMR was working (in which case x2 slice output will have 56 cells)
        if len(data['x2v']) != 56:
            pytest.fail("3D radiation beam tab output has wrong number of elements")
        # test that radiation energy density is small for x2v < 2
        if max(data['r00'][0:24]) > 0.0025:
            pytest.fail(
                "Energy density too large for x2<2, beam has not propagated correctly"
            )
        # test that radiation energy density is large for x2v > 2
        if max(data['r00'][24:56]) < 0.01:
            pytest.fail(
                "Energy density too small for x2>2, beam has not propagated correctly"
            )
    finally:
        testutils.cleanup()
