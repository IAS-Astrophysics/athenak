"""
SC coupled-dynamics test: radiatively damped acoustic wave in 1D with AMR (Davis, Stone &
Jiang 2012 sec. 5.4), the SC analogue of rad/test_rad_lwave1d_amr_cpu.

A small-amplitude acoustic wave in a radiating gas (affect_fluid on, Bo = tau = 1) is
damped and
phase-shifted by radiative heating and cooling. The mesh refines the wave crests
adaptively
(2 levels), so the run exercises the intensity and source restriction/prolongation, the
AMR
packing of ir, and the J rebuild after each remesh. After one period the pgen fits the
density
mode (volume-weighted, valid on AMR) and writes fitted vs analytic omega to
SCLinWave-davis54.dat; the full-state L1 error goes through the shared OutputErrors into
SCLinWave1D-errs.dat.

Parametrized over the sweep kernel. Over a resolution ladder (8 root blocks per run) this
asserts
  physics      fitted phase speed and damping match the analytic dispersion at the finest
  N
  convergence  the L1 rms error decreases monotonically at ~1st order (operator-split
  coupling)
SCLinWave-davis54.dat columns: Bo tau nx1 nx2 omega_r omega_i omega_r_fit omega_i_fit
"""
import math
import pytest
import test_suite.testutils as testutils

_RES = [256, 512, 1024]
_OMEGA_R, _OMEGA_I, _OMEGA_R_FIT, _OMEGA_I_FIT = 4, 5, 6, 7
_L1_RMS = 4
_KERNELS = [("plane", 0), ("tiled", 8)]


def _args(n, kernel, tile):
    return [f"mesh/nx1={n}", f"meshblock/nx1={n // 8}",
            f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}"]


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_lwave1d_amr(kernel, tile):
    testutils.cleanup()
    try:
        for n in _RES:
            assert testutils.run("inputs/sc_linwave_1d.athinput", _args(n, kernel,
                                                                        tile)), \
                f"run failed: {kernel} N={n}"
        data = testutils.athena_read.error_dat("SCLinWave-davis54.dat")
        errs = testutils.athena_read.error_dat("SCLinWave1D-errs.dat")
        rms = [row[_L1_RMS] for row in errs]
        fine = data[-1]
        assert abs(fine[_OMEGA_R_FIT] - fine[_OMEGA_R]) < 1.0e-3, \
            (f"{kernel}: phase speed off: fitted {fine[_OMEGA_R_FIT]:g} vs analytic "
             f"{fine[_OMEGA_R]:g}")
        assert abs(fine[_OMEGA_I_FIT] - fine[_OMEGA_I]) < 1.0e-3, \
            (f"{kernel}: damping off: fitted {fine[_OMEGA_I_FIT]:g} vs analytic "
             f"{fine[_OMEGA_I]:g}")
        assert all(rms[i + 1] < rms[i] for i in range(len(rms) - 1)), \
            f"{kernel}: L1 rms not monotonically decreasing: {rms}"
        order = math.log(rms[0] / rms[-1]) / math.log(_RES[-1] / _RES[0])
        assert order > 0.6, \
            f"{kernel}: insufficient convergence, observed order {order:.2f} (rms {rms})"
    finally:
        testutils.cleanup()
