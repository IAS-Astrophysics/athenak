"""
Sod shocktube test for non-relativistic hydro
Runs tests for different
  - reconstruction algorithms
  - Riemann solvers
"""

# Modules
import pytest
import test_suite.testutils as testutils
import athena_read
import numpy as np

_recon = ["plm", "ppm4", "ppmx", "wenoz"]  # do not change order
_flux = ["llf", "hlle", "hllc", "roe"]
_res = [128, 256]  # resolutions to test
input_file = "inputs/sod.athinput"


def compute_error(data, tlim=0.25):
    # Positions of shock, contact, head and foot of rarefaction for Sod test
    xs = 1.7522 * tlim
    xc = 0.92745 * tlim
    xf = -0.07027 * tlim
    xh = -1.1832 * tlim
    r = data["x1v"]
    dens = np.where(
        r > xs,
        0.125,
        np.where(
            r > xc,
            0.26557,
            np.where(
                r > xf,
                0.42632,
                np.where(
                    r > xh,
                    0.42632
                    * (1.0 + 0.20046 * (0.92745 - (0.92745 * (r - xh) / (xf - xh))))
                    ** 5,
                    1.0,
                ),
            ),
        ),
    )
    return (np.abs(data["dens"] - dens)).mean()


def arguments(iv, rv, fv, res):
    """Assemble arguments for run command"""
    return [
        "job/basename=sod",
        "mesh/nx1=" + repr(res),
        "meshblock/nx1=" + repr(128),
        "mesh/nghost=" + repr(2 if rv == "plm" else 3),
        "time/integrator=" + iv,
        "time/cfl_number=0.3",
        "hydro/reconstruct=" + rv,
        "hydro/rsolver=" + fv,
    ]


errors = {}


@pytest.mark.parametrize("rv", _recon)
@pytest.mark.parametrize("fv", _flux)
def test_run(fv, rv):
    """Loop over resolutions and run test with given reconstruction/flux."""
    iv = "rk2" if rv == "plm" else "rk3"
    try:
        for res in _res:
            results = testutils.run(input_file, arguments(iv, rv, fv, res))
            assert results, f"Sod test run failed for {iv}+{res}+{fv}+{rv}."
            data = athena_read.tab("tab/sod.hydro_w.00001.tab")
            errors[res] = compute_error(data)
        # check convergence
        l1_rms_lr = errors[_res[0]]
        l1_rms_hr = errors[_res[1]]
        convrate = 0.6 ** (np.log2(_res[1] / _res[0]))
        if l1_rms_hr / l1_rms_lr > convrate:
            pytest.fail(
                f"Not converging for  {iv}+{rv}+{fv} configuration, "
                f"conv: {l1_rms_hr / l1_rms_lr:g} threshold: {convrate:g}"
            )
    finally:
        testutils.cleanup()
