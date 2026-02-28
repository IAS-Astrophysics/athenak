"""
Ambipolar diffusion MHD wave damping test (GPU).
Same physics as the CPU test but with mesh decomposition tuned for GPU execution
and higher resolutions.
"""

import pytest
import numpy as np
from numpy.polynomial import Polynomial
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import test_suite.testutils as testutils  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vis", "python"))
import athena_read  # noqa: E402

# Physical parameters (Bai & Stone 2011, Section 2.3.2)
_eta_ad = 0.01
_omega_a = 100.0
_cs = 1.0
_rho0 = 1.0
_bx0 = 1.0
_by0 = np.sqrt(2.0)
_bz0 = 0.5

_Bsq = _bx0**2 + _by0**2 + _bz0**2
_vAsq = _Bsq / _rho0
_vAxsq = _bx0**2 / _rho0
_cssq = _cs**2
_btsq = _by0**2 + _bz0**2

_tsum = _vAxsq + _btsq / _rho0 + _cssq
_tdif = _vAxsq + _btsq / _rho0 - _cssq
_cfsq = 0.5 * (_tsum + np.sqrt(_tdif**2 + 4.0 * _cssq * _btsq / _rho0))
_cssq_mhd = _cssq * _vAxsq / _cfsq

_k = 2.0 * np.pi

ANALYTIC_RATES = {
    "0": 0.5 * (_cfsq - _cssq) / (_cfsq - _cssq_mhd) * _k**2 * _vAsq / _omega_a,
    "1": 0.5 * _k**2 * _vAxsq / _omega_a,
    "2": 0.5 * (_cssq - _cssq_mhd) / (_cfsq - _cssq_mhd) * _k**2 * _vAsq / _omega_a,
}
WAVE_NAMES = {"0": "fast", "1": "Alfven", "2": "slow"}

RESOLUTIONS = [64, 128]
REL_TOL = 0.15
CONVERGENCE_RATE_MIN = 1.5

DOMAINS = {
    1: {"x1max": "1.0", "nx2": "1", "x2max": "1.0", "nx3": "1", "x3max": "1.0"},
    2: {"x1max": str(np.sqrt(5.0)), "nx2": "HALF",
        "x2max": str(np.sqrt(5.0) / 2.0), "nx3": "1", "x3max": "1.0"},
    3: {"x1max": "3.0", "nx2": "HALF", "x2max": "1.5", "nx3": "HALF", "x3max": "1.5"},
}


def build_arguments(wave_flag, dim, res, basename):
    """Assemble runtime arguments for a single GPU test run."""
    domain = DOMAINS[dim]
    nx2 = "1" if domain["nx2"] == "1" else str(res // 2)
    nx3 = "1" if domain["nx3"] == "1" else str(res // 2)

    mb_nx1 = str(min(res, 64))
    mb_nx2 = "1" if nx2 == "1" else str(min(int(nx2), 32))
    mb_nx3 = "1" if nx3 == "1" else str(min(int(nx3), 32))

    return [
        f"job/basename={basename}",
        f"time/tlim=5.0",
        f"time/integrator=rk2",
        f"time/cfl_number=0.3",
        f"mesh/nghost=2",
        f"mesh/nx1={res}",
        f"mesh/x1min=0.0",
        f"mesh/x1max={domain['x1max']}",
        f"mesh/nx2={nx2}",
        f"mesh/x2min=0.0",
        f"mesh/x2max={domain['x2max']}",
        f"mesh/nx3={nx3}",
        f"mesh/x3min=0.0",
        f"mesh/x3max={domain['x3max']}",
        f"meshblock/nx1={mb_nx1}",
        f"meshblock/nx2={mb_nx2}",
        f"meshblock/nx3={mb_nx3}",
        f"mesh_refinement/refinement=none",
        f"mhd/eos=isothermal",
        f"mhd/iso_sound_speed=1.0",
        f"mhd/reconstruct=plm",
        f"mhd/rsolver=hlld",
        f"mhd/ambipolar_diffusivity=constant",
        f"mhd/eta_ad={_eta_ad}",
        f"output1/file_type=hst",
        f"output1/dt=0.05",
        f"problem/pgen_name=linear_wave",
        f"problem/wave_flag={wave_flag}",
        f"problem/amp=1.0e-4",
        f"problem/dens=1.0",
        f"problem/pgas=0.6",
        f"problem/vx0=0.0",
        f"problem/bx0={_bx0}",
        f"problem/by0={_by0}",
        f"problem/bz0={_bz0}",
        f"problem/along_x1={'true' if dim == 1 else 'false'}",
    ]


def fit_decay_rate_from_ke(hst_file):
    """Fit exponential decay rate from the total KE history variables."""
    hst_data = athena_read.hst(hst_file)
    tt = hst_data["time"]
    ke_tot = hst_data["1-KE"] + hst_data["2-KE"] + hst_data["3-KE"]

    mask = ke_tot > 0
    tt = tt[mask]
    ke_tot = ke_tot[mask]
    if len(tt) < 5:
        return np.nan

    yy = np.log(ke_tot)
    p, _ = Polynomial.fit(tt, yy, 1, w=np.sqrt(ke_tot), full=True)
    pnormal = p.convert(domain=(-1, 1))
    slope = pnormal.coef[-1]
    return -slope / 2.0


@pytest.mark.parametrize("wave_flag", ["0", "1", "2"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_ambipolar_linwave(wave_flag, dim):
    """Test ambipolar damping rate and convergence on GPU."""
    analytic_rate = ANALYTIC_RATES[wave_flag]
    wave_name = WAVE_NAMES[wave_flag]
    errors_abs = []

    try:
        for res in RESOLUTIONS:
            basename = f"AmbLW_w{wave_flag}_{dim}d_{res}"
            args = build_arguments(wave_flag, dim, res, basename)
            testutils.run("inputs/lwave_ambipolar.athinput", args)

            hst_file = f"{basename}.mhd.hst"
            measured_rate = fit_decay_rate_from_ke(hst_file)

            error_abs = abs(analytic_rate - measured_rate)
            errors_abs.append(error_abs)
            error_rel = abs(analytic_rate / measured_rate - 1.0)

            if error_rel > REL_TOL:
                pytest.fail(
                    f"{wave_name} {dim}D N={res}: decay rate relative error "
                    f"{error_rel:.3f} exceeds tolerance {REL_TOL}"
                )

        if len(errors_abs) >= 2 and all(e > 0 for e in errors_abs):
            conv_rate = np.log(errors_abs[-2] / errors_abs[-1]) / np.log(
                RESOLUTIONS[-1] / RESOLUTIONS[-2]
            )
            if conv_rate < CONVERGENCE_RATE_MIN:
                pytest.fail(
                    f"{wave_name} {dim}D: convergence rate {conv_rate:.2f} "
                    f"< {CONVERGENCE_RATE_MIN}"
                )
    finally:
        testutils.cleanup()
