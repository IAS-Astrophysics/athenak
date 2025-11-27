"""
RJ2a MHD shocktube test for non-relativistic MHD
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
_flux = ["llf", "hlle", "hlld"]
_res = [128, 256]  # resolutions to test
input_file = "inputs/rj2a.athinput"


def compute_error(data, tlim=0.2):
    # compares density to analytic solution
    xfp = 2.2638 * tlim
    xrp = (0.53432 + 1.0 / np.sqrt(np.pi * 1.309)) * tlim
    xsp = (0.53432 + 0.48144 / 1.309) * tlim
    xc = 0.57538 * tlim
    xsm = (0.60588 - 0.51594 / 1.4903) * tlim
    xrm = (0.60588 - 1.0 / np.sqrt(np.pi * 1.4903)) * tlim
    xfm = (1.2 - 2.3305 / 1.08) * tlim
    r = data["x1v"]
    dens = np.where(
        r > xfp,
        1.0,
        np.where(
            r > xrp,
            1.3090,
            np.where(
                r > xsp,
                1.3090,
                np.where(
                    r > xc,
                    1.4735,
                    np.where(
                        r > xsm,
                        1.6343,
                        np.where(r > xrm, 1.4903, np.where(r > xfm, 1.4903, 1.08)),
                    ),
                ),
            ),
        ),
    )
    return (np.abs(data["dens"] - dens)).mean()


def arguments(iv, rv, fv, res):
    """Assemble arguments for run command"""
    return [
        "mesh/nx1=" + repr(res),
        "meshblock/nx1=" + repr(int(np.min(_res))),
        "mesh/nghost=" + repr(2 if rv == "plm" else 3),
        "time/integrator=" + iv,
        "time/cfl_number=0.3",
        "mhd/reconstruct=" + rv,
        "mhd/rsolver=" + fv,
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
            assert results, f"RJ2a test run failed for {iv}+{res}+{fv}+{rv}."
            data = athena_read.tab("tab/RJ2a.mhd_w.00001.tab")
            errors[res] = compute_error(data)
        # check convergence
        l1_rms_lr = errors[_res[0]]
        l1_rms_hr = errors[_res[1]]
        convrate = 0.6 ** (np.log2(_res[1] / _res[0]))
        if l1_rms_hr / l1_rms_lr > convrate:
            pytest.fail(
                f"Not converging for {iv}+{rv}+{fv} configuration, "
                f"conv: {l1_rms_hr / l1_rms_lr:g} threshold: {convrate:g}"
            )
    finally:
        testutils.cleanup()
