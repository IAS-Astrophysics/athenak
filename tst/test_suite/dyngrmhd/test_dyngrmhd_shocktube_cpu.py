"""
Relativistic shocktube tests for MHD in dynamical spacetimes
Nearly identical to shocktube tests for GRMHD; just skips hydro tests
Runs tests for different
  - reconstruction algorithms
  - Riemann solvers
For MHD runs "test1" from Mignone, Ugliano, & Bodo 2009, MNRAS 393 1141

Since no analytic solutions are available, a reference solution computed using WENOZ
is used for comparisons. If the reference solution itself is incorrect, this will still
cause tests to fail unless test solutions are also wrong in the same way (unlikely).
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np

_recon = ["plm", "ppm4", "ppmx", "wenoz"]  # do not change order
_flux = ["llf", "hlle"]
_res = [256, 512]  # resolutions to test
_soe = ["mhd"]  # system of equations to test
name = {"mhd": "mub1_dyngrmhd"}  # names of the tests
# algorithmic choices for reference solution
ref_key = {"mhd": ("hlle", "wenoz")}
# convergence ratio threshold for failure
ratio_threshold = {"mhd": 0.8}


def arguments(iv, rv, fv, res, name, soe):
    """Assemble arguments for run command"""
    return [
        f"job/basename={name}_{iv}_{rv}_{fv}_{res}",
        "mesh/nx1=" + repr(res),
        "meshblock/nx1=" + repr(128),
        "mesh/nghost=" + repr(2 if rv == "plm" else 3),
        "time/integrator=" + iv,
        "time/cfl_number=0.2",
        "coord/special_rel=false",
        "coord/general_rel=true",
        f"{soe}/reconstruct=" + rv,
        f"{soe}/rsolver=" + fv,
    ]


def run_test(iv, rv, fv, res, name, soe):
    """Run a single test with given parameters, return density at final time."""
    input_file = f"inputs/{name}.athinput"
    testutils.run(input_file, arguments(iv, rv, fv, res, name, soe))
    data = athena_read.tab(f"tab/{name}_{iv}_{rv}_{fv}_{res}.{soe}_w.00001.tab")
    return data["dens"]


# Run suite of tests, storing density in results[]
results = {}


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("fv", _flux)
@pytest.mark.parametrize("soe", _soe)
def test_run(fv, rv, soe):
    """Run a single test with given parameters, store density in results[]."""
    iv = "rk2" if rv == "plm" else "rk3"
    try:
        for res in _res:
            results[(soe, fv, rv, res)] = run_test(iv, rv, fv, res, name[soe], soe)
    finally:
        testutils.cleanup()


# Check whether results converge over entire suite of tests
@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("fv", _flux)
@pytest.mark.parametrize("soe", _soe)
def test_convergence(fv, rv, soe):
    if ref_key[soe] == (fv, rv):
        pytest.skip("Can't compare reference against reference")

    error = {}
    # Calculate errors for the different resolutions
    for res in _res:
        ref = results[(soe,) + ref_key[soe] + (res,)]
        error[res] = (np.abs(results[(soe, fv, rv, res)] - ref)).mean()
        if error[res] > 3e-2:
            pytest.fail(f"Error for {(soe, fv, rv)} at resolution {res}: {error[res]}")
    # Check convergence
    ratio = error[_res[1]] / error[_res[0]]
    if ratio > ratio_threshold[soe]:
        pytest.fail(
            f"Error for {(soe, fv, rv)} between {_res[1]} and {_res[0]} too large."
            f"Error ration: {ratio} threshold: {ratio_threshold[soe]}"
        )
