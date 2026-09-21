"""
SC coupled-dynamics test: radiatively damped acoustic wave in 2D with AMR and MPI, the SC
analogue of rad/test_rad_lwave2d_amr_mpicpu.

The wave propagates along the domain diagonal (the analytic dispersion needs no grid
alignment
for this solver), 8 root meshblocks on 8 ranks, two AMR levels following the crests. This
exercises the MPI ghost exchange of intensity and source across ranks, the AMR packing of
ir
across ranks, and the allreduced residual and dispersion fit. Over two resolutions, both
sweep
kernels:
  physics      fitted phase speed and damping match the analytic dispersion at the finer N
  convergence  the L1 rms error drops by more than a factor 2 from N to 2N
  ranks        the 8-rank run at the coarser N reproduces the 1-rank run's error line to a
               tolerance (sums are reduced in a different order, so not bitwise)
"""
import pytest
import test_suite.testutils as testutils

_RES = [64, 128]
_OMEGA_R, _OMEGA_I, _OMEGA_R_FIT, _OMEGA_I_FIT = 4, 5, 6, 7
_L1_RMS = 4
_KERNELS = [("wavefront", 0), ("tiled", 8), ("plane", 0)]
_RANKS = 8


def _args(n, kernel, tile):
    return [f"mesh/nx1={n}", f"mesh/nx2={n // 2}",
            f"meshblock/nx1={n // 4}", f"meshblock/nx2={n // 8}",
            f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}"]


def _last(path):
    return open(path).read().splitlines()[-1].split()


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_lwave2d_amr_mpi(kernel, tile):
    testutils.cleanup()
    try:
        for n in _RES:
            assert testutils.mpi_run("inputs/sc_linwave.athinput", _args(n, kernel, tile),
                                     threads=_RANKS), \
                f"run failed: {kernel} N={n} on {_RANKS} ranks"
        data = testutils.athena_read.error_dat("SCLinWave-davis54.dat")
        errs = testutils.athena_read.error_dat("SCLinWave-errs.dat")
        fine = data[-1]
        assert abs(fine[_OMEGA_R_FIT] - fine[_OMEGA_R]) < 1.0e-3, \
            (f"{kernel}: phase speed off: fitted {fine[_OMEGA_R_FIT]:g} vs analytic "
             f"{fine[_OMEGA_R]:g}")
        assert abs(fine[_OMEGA_I_FIT] - fine[_OMEGA_I]) < 1.0e-3, \
            (f"{kernel}: damping off: fitted {fine[_OMEGA_I_FIT]:g} vs analytic "
             f"{fine[_OMEGA_I]:g}")
        rms = [row[_L1_RMS] for row in errs]
        assert rms[1] < 0.5 * rms[0], f"{kernel}: L1 rms converges too slowly: {rms}"

        # rank independence at the coarser resolution: 8 ranks vs 1 rank
        testutils.cleanup()
        assert testutils.mpi_run("inputs/sc_linwave.athinput", _args(_RES[0], kernel,
                                                                     tile),
                                 threads=_RANKS)
        line8 = _last("SCLinWave-errs.dat")
        testutils.cleanup()
        assert testutils.mpi_run("inputs/sc_linwave.athinput", _args(_RES[0], kernel,
                                                                     tile),
                                 threads=1)
        line1 = _last("SCLinWave-errs.dat")
        assert line8[3] == line1[3], \
            f"{kernel}: cycle counts differ: {line8[3]} vs {line1[3]}"
        for c in range(4, len(line1)):
            a, b = float(line8[c]), float(line1[c])
            assert abs(a - b) <= 1.0e-10 * max(abs(a), abs(b), 1.0e-300) + 1.0e-300, \
                f"{kernel}: column {c} differs between 8 and 1 ranks: {a:g} vs {b:g}"
    finally:
        testutils.cleanup()
