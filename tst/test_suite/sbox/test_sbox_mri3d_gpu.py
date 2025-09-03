"""
3D MRI test: runs unstratified net-flux MRI in large box (4x8x1) and checks
whether mean B^2 after saturation (t>25) is within expected range.
Previous tests show we should expect B^2 ~ 0.1 for this problem
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np

input_file = "inputs/mri3d.athinput"


# function to calculate mean of B^2 over time interval t=[25,50]
# assumed data (history file) runs from t=[0,50]
def compute_mean(data):
    time = data['time']
    # zero B2 for t<25
    b2 = np.where(
        time > 25.0,
        # divide by 32 to correct for volume of domain
        (data['1-ME'] + data['2-ME'] + data['3-ME'])/32.0,
        0.0,
    )
    # return 2X mean since 1/2 of array is zero
    return (2.0*(np.mean(b2)))


def arguments():
    """Assemble arguments for run command"""
    return [
        "job/basename=HGB",
        "mesh/nx1=64",
        "mesh/nx2=128",
        "mesh/nx3=16",
        "meshblock/nx1=32",
        "meshblock/nx2=32",
        "meshblock/nx3=16",
        "time/tlim=50.0",
        "mhd/reconstruct=wenoz"
    ]


def test_run():
    """run test with given reconstruction/flux."""
    try:
        results = testutils.run(input_file, arguments())
        assert results, "MRI test run failed"
        data = athena_read.hst("HGB.user.hst")
        # check mean B^2 over time-interval t=[25,50]
        aveb2 = compute_mean(data)
        if aveb2 < 0.05 or aveb2 > 0.15:
            pytest.fail(
                f"Mean B^2 not within expected range for MRI test"
                f"mean: {aveb2:g} range: (0.05-0.15)"
            )
    finally:
        testutils.cleanup()
