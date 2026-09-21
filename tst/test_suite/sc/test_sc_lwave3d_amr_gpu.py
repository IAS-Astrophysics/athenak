"""
SC coupled-dynamics test: radiatively damped acoustic wave in 3D with AMR on the GPU, the
SC
analogue of rad/test_rad_lwave3d_amr_gpu.

The 2D oblique wave extended uniformly along x3 (64 x 32 x 8 cells, 16 x 8 x 8
meshblocks, two
AMR levels), both sweep kernels. One resolution: this run is too costly for a CPU ladder
and the
1D/2D tests carry the convergence checks; here the 3D kernels (bilinear footpoints, the
tiled
team sweep over 3D tiles) and 3D AMR of the intensity are exercised end to end.
  physics    fitted phase speed and damping match the analytic dispersion
  error      the L1 rms error is below a floor
"""
import pytest
import test_suite.testutils as testutils

_OMEGA_R, _OMEGA_I, _OMEGA_R_FIT, _OMEGA_I_FIT = 4, 5, 6, 7
_L1_RMS = 4
_KERNELS = [("wavefront", 0), ("tiled", 8), ("plane", 0)]
_ARGS = ["mesh/nx3=8", "meshblock/nx3=8", "mesh/x3min=0.0", "mesh/x3max=0.2795"]
# the 2D run at this resolution gives 9.9e-9 (2026-09-11); set on the GPU
_L1_FLOOR = 2.0e-8


@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_lwave3d_amr(kernel, tile):
    testutils.cleanup()
    try:
        args = _ARGS + [f"nr_radiation/sweep_kernel={kernel}",
                        f"nr_radiation/tile_size={tile}"]
        assert testutils.run("inputs/sc_linwave.athinput",
                             args), f"3D run failed: {kernel}"
        fine = testutils.athena_read.error_dat("SCLinWave-davis54.dat")[-1]
        assert abs(fine[_OMEGA_R_FIT] - fine[_OMEGA_R]) < 1.0e-3, \
            (f"{kernel}: phase speed off: fitted {fine[_OMEGA_R_FIT]:g} vs analytic "
             f"{fine[_OMEGA_R]:g}")
        assert abs(fine[_OMEGA_I_FIT] - fine[_OMEGA_I]) < 1.0e-3, \
            (f"{kernel}: damping off: fitted {fine[_OMEGA_I_FIT]:g} vs analytic "
             f"{fine[_OMEGA_I]:g}")
        rms = testutils.athena_read.error_dat("SCLinWave-errs.dat")[-1][_L1_RMS]
        assert rms < _L1_FLOOR, f"{kernel}: L1 rms {rms:g} above floor {_L1_FLOOR:g}"
    finally:
        testutils.cleanup()
