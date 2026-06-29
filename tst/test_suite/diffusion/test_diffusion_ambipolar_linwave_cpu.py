"""
Ambipolar diffusion MHD wave damping test (CPU).

Reproduces Bai & Stone (2011), Section 2.3.2: isothermal fast/Alfven/slow MHD waves damped
by ambipolar diffusion. The problem is initialized with the IDEAL MHD eigenvector and the
exponential decay rate of the (volume-integrated) kinetic energy is measured and compared to
the analytic AD damping rates from Balsara (1996) (eqs. 17-18): Gamma_fast=0.5132,
Gamma_Alfven=0.1974, Gamma_slow=0.1283 for eta_ad=0.01, omega_a=100, B0=(1, sqrt2, 0.5).

Each (wave, dimension) is run once at a single resolution N=64 (1D 64, 2D 64x32,
3D 64x32x32) and the measured damping rate is checked to be within REL_TOL of the analytic
value. N=64 gives >~20 cells/wavelength, which Bai & Stone (2011) Sec 2.3.2 identify as the
threshold for accurate AD damping. See results/wave_damping/bai_stone_resolution_study.py for
the full half/fiducial/double resolution study.
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

RESOLUTION = 64    # single base resolution (1D 64, 2D 64x32, 3D 64x32x32)
REL_TOL = 0.20     # damping rate within 20% of analytic (N=64 is the fiducial grid in 2D/3D,
                   # where the worst case, slow 3D, is ~17% per Bai & Stone 2011 Sec 2.3.2)

# Domain configurations per dimension
DOMAINS = {
    1: {"x1max": "1.0", "nx2": "1", "x2max": "1.0", "nx3": "1", "x3max": "1.0"},
    2: {"x1max": str(np.sqrt(5.0)), "nx2": "HALF",
        "x2max": str(np.sqrt(5.0) / 2.0), "nx3": "1", "x3max": "1.0"},
    3: {"x1max": "3.0", "nx2": "HALF", "x2max": "1.5", "nx3": "HALF", "x3max": "1.5"},
}


def build_arguments(wave_flag, dim, res, basename):
    """Assemble runtime arguments for a single test run."""
    domain = DOMAINS[dim]
    nx2 = "1" if domain["nx2"] == "1" else str(res // 2)
    nx3 = "1" if domain["nx3"] == "1" else str(res // 2)
    mb_nx2 = nx2
    mb_nx3 = nx3

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
        f"meshblock/nx1={res}",
        f"meshblock/nx2={mb_nx2}",
        f"meshblock/nx3={mb_nx3}",
        f"mesh_refinement/refinement=none",
        f"mhd/eos=isothermal",
        f"mhd/iso_sound_speed=1.0",
        f"mhd/reconstruct=plm",
        f"mhd/rsolver=hlld",
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
    """Fit exponential decay rate from the total KE history variables.

    The volume-integrated total KE (1-KE + 2-KE + 3-KE) decays as
    exp(-2*Gamma*t), so the fitted slope is -2*Gamma.
    Using total KE rather than a single component gives better
    convergence for fast/slow waves whose eigenvectors span all
    velocity components.
    """
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
    """Check the ambipolar damping rate is within REL_TOL of analytic at N=64."""
    analytic_rate = ANALYTIC_RATES[wave_flag]
    wave_name = WAVE_NAMES[wave_flag]

    try:
        basename = f"AmbLW_w{wave_flag}_{dim}d_{RESOLUTION}"
        testutils.run("inputs/lwave_ambipolar.athinput",
                      build_arguments(wave_flag, dim, RESOLUTION, basename))
        measured_rate = fit_decay_rate_from_ke(f"{basename}.mhd.hst")
        error_rel = abs(analytic_rate / measured_rate - 1.0)
        if error_rel > REL_TOL:
            pytest.fail(
                f"{wave_name} {dim}D N={RESOLUTION}: damping-rate relative error "
                f"{error_rel:.3f} exceeds tolerance {REL_TOL} "
                f"(measured {measured_rate:.4f}, analytic {analytic_rate:.4f})"
            )
    finally:
        testutils.cleanup()
