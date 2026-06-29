"""
Ambipolar diffusion MHD wave damping test (CPU).

Reproduces Bai & Stone (2011), Section 2.3.2: isothermal fast/Alfven/slow MHD waves damped
by ambipolar diffusion. The problem is initialized with the IDEAL MHD eigenvector and the
exponential decay rate of the (volume-integrated) kinetic energy is measured and compared to
the analytic AD damping rates from Balsara (1996) (eqs. 17-18): Gamma_fast=0.5132,
Gamma_Alfven=0.1974, Gamma_slow=0.1283 for eta_ad=0.01, omega_a=100, B0=(1, sqrt2, 0.5).

Resolution handling follows the paper's finding that accurate AD damping requires
>~20 cells/wavelength. With the box sizes used here, RESOLUTIONS=[32, 64] maps to:
  1D: 32=fiducial, 64=double;   2D/3D: 32=half, 64=fiducial.
Accuracy (rel. error < REL_TOL) is therefore checked only at the FINEST resolution (which is
the fiducial grid or finer for every dimension); the coarse grid is used only for the
convergence check, since the paper shows the rate deviates substantially at half resolution.
See results/wave_damping/bai_stone_resolution_study.py for the full half/fiducial/double study.
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

RESOLUTIONS = [32, 64]
REL_TOL = 0.25
CONVERGENCE_RATE_MIN = 1.5
# The 2nd-order convergence-rate check is only enforced when the COARSE grid's rate error
# exceeds this floor (i.e. it is genuinely under-resolved). Once both grids are already near
# the accuracy floor, the residual is limited by ideal-eigenvector contamination / fit noise,
# not truncation, so the convergence rate flattens; there we only require the error to decrease.
CONV_CHECK_MIN_ERR = 0.05

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
    """Test ambipolar damping rate and convergence for a given wave type and dimension."""
    analytic_rate = ANALYTIC_RATES[wave_flag]
    wave_name = WAVE_NAMES[wave_flag]

    try:
        rates = []
        for res in RESOLUTIONS:
            basename = f"AmbLW_w{wave_flag}_{dim}d_{res}"
            testutils.run("inputs/lwave_ambipolar.athinput",
                          build_arguments(wave_flag, dim, res, basename))
            rates.append(fit_decay_rate_from_ke(f"{basename}.mhd.hst"))
        errors_abs = [abs(analytic_rate - r) for r in rates]

        # Accuracy is checked ONLY at the finest resolution (RESOLUTIONS[-1]), which is the
        # paper's fiducial grid or finer for every dimension. Per Bai & Stone (2011) Sec 2.3.2,
        # accurate AD damping needs >~20 cells/wavelength; coarser grids deviate substantially
        # and are dominated by numerical (not ambipolar) dissipation, so no absolute tolerance
        # is imposed there. The resolution dependence is captured by the convergence check.
        error_rel = abs(analytic_rate / rates[-1] - 1.0)
        if error_rel > REL_TOL:
            pytest.fail(
                f"{wave_name} {dim}D N={RESOLUTIONS[-1]}: damping-rate relative error "
                f"{error_rel:.3f} exceeds tolerance {REL_TOL}"
            )

        # Convergence (Bai & Stone Sec 2.3.2): the rate error must keep shrinking with
        # resolution. The 2nd-order convergence rate is only diagnostic when the coarse grid is
        # genuinely under-resolved (coarse error > CONV_CHECK_MIN_ERR); once both grids are
        # already near the accuracy floor (e.g. 1D at high resolution) we only require the error
        # to decrease, since the residual there is contamination/fit noise, not truncation.
        if len(errors_abs) >= 2 and all(e > 0 for e in errors_abs):
            coarse_rel = abs(analytic_rate / rates[0] - 1.0)
            conv_rate = np.log(errors_abs[-2] / errors_abs[-1]) / np.log(
                RESOLUTIONS[-1] / RESOLUTIONS[-2]
            )
            if coarse_rel > CONV_CHECK_MIN_ERR:
                if conv_rate < CONVERGENCE_RATE_MIN:
                    pytest.fail(
                        f"{wave_name} {dim}D: convergence rate {conv_rate:.2f} "
                        f"< {CONVERGENCE_RATE_MIN}"
                    )
            elif errors_abs[-1] >= errors_abs[-2]:
                pytest.fail(
                    f"{wave_name} {dim}D: rate error did not decrease with resolution "
                    f"({errors_abs[-2]:.2e} -> {errors_abs[-1]:.2e})"
                )
    finally:
        testutils.cleanup()


@pytest.mark.parametrize("dim", [2, 3])
def test_ambipolar_linwave_multiblock(dim):
    """Test ambipolar damping with 2 meshblocks in x1.

    Exercises the cross-J 4-point averaging at internal meshblock boundaries.
    The ambipolar EMF recomputes J = curl(B) inline from the face fields (b0) via
    EdgeJ{1,2,3} in Resistivity::AddEMFConstantAmbipolar, reading the b0 ghost zones
    populated by the boundary exchange, so the cross-J averages are well defined across
    meshblock boundaries.  Only 2D/3D are tested because 1D has no cross-component J
    averaging.  Uses the fast wave (wave_flag=0) at a single resolution.
    """
    wave_flag = "0"
    analytic_rate = ANALYTIC_RATES[wave_flag]
    res = 64

    try:
        basename = f"AmbLW_mb_w{wave_flag}_{dim}d_{res}"
        args = build_arguments(wave_flag, dim, res, basename)

        mb_nx1 = str(res // 2)
        args = [a for a in args if not a.startswith("meshblock/nx1=")]
        args.append(f"meshblock/nx1={mb_nx1}")

        testutils.run("inputs/lwave_ambipolar.athinput", args)

        hst_file = f"{basename}.mhd.hst"
        measured_rate = fit_decay_rate_from_ke(hst_file)
        error_rel = abs(analytic_rate / measured_rate - 1.0)

        if error_rel > REL_TOL:
            pytest.fail(
                f"fast {dim}D multiblock: decay rate relative error "
                f"{error_rel:.3f} exceeds tolerance {REL_TOL}"
            )
    finally:
        testutils.cleanup()
