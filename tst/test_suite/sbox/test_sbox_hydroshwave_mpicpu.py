"""
Hydro shearing wave (shwave) test for shearing box module.
Runs incompressible (vortical) shwave test of JG05, and compares
evolution of Vx to analytic solution
"""

# Modules
import pytest
from subprocess import Popen, PIPE
import test_suite.testutils as testutils
import athena_read
import numpy as np

_res = [32, 64]  # resolutions to test
input_file = "inputs/hydro_shwave.athinput"


def compute_error(data, amp=1.0e-4):
    # computeds analytic solution of JG05 for dVx
    t = data["time"]
    dvx = amp*17.0/(1.0 + (1.5*t - 4.0)*(1.5*t - 4.0))
    # plotting routines were used to test solutions was correct
    # import matplotlib.pyplot as plt
    # plt.figure
    # plt.plot(t,dvx)
    # plt.plot(t,np.sqrt(32.0*data["1-KE"]),color='red')
    # plt.show()
    return (np.abs(np.sqrt(32.0*data["1-KE"]) - dvx)).mean()


def arguments(res):
    """Assemble arguments for run command"""
    return [
        "job/basename=shwave",
        "mesh/nx1=" + repr(res),
        "meshblock/nx1=32",
        "mesh/nx2=" + repr(res),
        "meshblock/nx2=32",
        "mesh/nx3=4",
        "meshblock/nx3=4",
    ]


errors = {}


def test_run():
    """Loop over resolutions and run test with given reconstruction/flux."""
    try:
        for res in _res:
            # set number of threads to number of MeshBlocks
            nthreads = 4 if res == 64 else 1
            results = testutils.mpi_run(input_file, arguments(res), threads=nthreads)
            assert results, f"Hydro shwave test run failed for {res}."
            data = athena_read.hst("shwave.hydro.hst")
            errors[res] = compute_error(data)
            # delete history files so new data will be read properly
            Popen(["rm shwave.hydro.hst"], shell=True, stdout=PIPE).communicate()
        l1_rms_lr = errors[_res[0]]
        l1_rms_hr = errors[_res[1]]
        # check convergence
        convrate = 0.25 ** (np.log2(_res[1] / _res[0]))
        if l1_rms_hr / l1_rms_lr > convrate:
            pytest.fail(
                f"Hydro shwave not converging, "
                f"conv: {l1_rms_hr / l1_rms_lr:g} threshold: {convrate:g}"
            )
        # check absolute error
        maxerr = 1.6e-5
        if l1_rms_hr > maxerr:
            pytest.fail(
                f"Hydro shwave error too large at highest resolution, "
                f"error: {l1_rms_hr:g} threshold: {maxerr:g}"
            )
    finally:
        testutils.cleanup()
