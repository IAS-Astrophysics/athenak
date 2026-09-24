"""Analytic temporal convergence for ADM Boltzmann and M1 geometric sources."""
import math
import os
from pathlib import Path
import re
import subprocess

import pytest
import test_suite.testutils as testutils


@pytest.mark.parametrize("solver", ["boltzmann", "m1"])
@pytest.mark.parametrize("profile", ["linear", "exponential", "quadratic"])
def test_redshift(solver, profile, tmp_path):
    binary = Path(os.environ.get("ATHENA_BIN", testutils.ATHENAK_BUILD / "athena"))
    suffix = "_m1" if solver == "m1" else ""
    source = (testutils.ATHENAK_PATH / "tst" / "inputs" /
              f"adm_flrw_redshift{suffix}.athinput").read_text()
    source += f"\n<problem>\nscale_factor = {profile}\nquadratic = 0.3\n"
    inp = tmp_path / "input.athinput"
    inp.write_text(source)
    errors = []
    for cfl in [0.4, 0.2, 0.1]:
        dest = tmp_path / str(cfl)
        dest.mkdir()
        result = subprocess.run(
            [str(binary), "-i", str(inp), "-d", str(dest),
             f"time/cfl_number={cfl}"], capture_output=True, text=True, check=True)
        match = re.search(r"ADM_FORMAL_TEST flrw .*rel_E=(\S+)", result.stdout)
        assert match, result.stdout
        errors.append(float(match[1]))
    if solver == "boltzmann" and profile == "linear":
        # Heun happens to integrate U=1/(1+Ht) exactly. Do not infer order here.
        assert max(errors) < 1.e-12
    else:
        assert errors[-1] < 1.e-4
        assert math.log2(errors[-2] / errors[-1]) > 1.8


@pytest.mark.parametrize("solver", ["boltzmann", "m1"])
def test_transparent_fluid(solver, tmp_path):
    """Metric refresh must precede primitive recovery and both flux operators."""
    import numpy as np
    binary = Path(os.environ.get("ATHENA_BIN", testutils.ATHENAK_BUILD / "athena"))
    suffix = "_m1" if solver == "m1" else ""
    source = (testutils.ATHENAK_PATH / "tst" / "inputs" /
              f"adm_flrw_redshift{suffix}.athinput").read_text()
    source += """
<adm>
dynamic = true
<problem>
scale_factor = quadratic
quadratic = .3
<mhd>
eos = ideal
dyn_eos = ideal
dyn_error = reset_floor
reconstruct = plm
rsolver = hlle
gamma = 1.6666666666666667
<output2>
file_type = tab
variable = mhd_w
slice_x2 = .5
data_format = %.16e
dt = 1
"""
    if solver == "m1":
        source += ("\n<radiation_m1>\nopacity_type = photons\n"
                   "photon_coupled_sources = true\nmatter_sources = false\n<photons>\n")
    else:
        source += "\n<dyn_radiation>\nrad_source = false\n"
    source += "kappa_a = 0\nkappa_s = 0\nkappa_p = 0\narad = 1\n"
    inp = tmp_path / "input.athinput"
    inp.write_text(source)
    errors = []
    a = 1 + .2*.5 + .3*.5**2
    for cfl in [.4, .2, .1]:
        dest = tmp_path / str(cfl)
        dest.mkdir()
        subprocess.run([str(binary), "-i", str(inp), "-d", str(dest),
                        f"time/cfl_number={cfl}"], capture_output=True, check=True)
        data = np.loadtxt(sorted((dest / "tab").glob("*.mhd_w.*.tab"))[-1])
        assert np.max(abs(data[:, 3] / a**-3 - 1)) < 1.e-11
        errors.append(np.max(abs(data[:, 7] / a**-5 - 1)))
    assert math.log2(errors[-2] / errors[-1]) > 1.8
