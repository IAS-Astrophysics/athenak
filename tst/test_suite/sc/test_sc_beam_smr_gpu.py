"""
SC transport test: two crossing pencil beams through a statically refined mesh, after
Davis, Stone & Jiang 2012 Fig. 6 (Athena-C beam2d), with SMR added. The SC analogue of
rad/test_rad_beam_gpu.

Box [0,1] x [0,0.5], 64 x 32 root cells in 16 x 8 meshblocks, a level-1 refined band over
the
central [0.25,0.75] x [0.125,0.375] (interior, so every beam enters and leaves it). Unit
intensity enters through two ghost cells of root columns 16 and 47 of the bottom face
along
ray `iang` of octants 0 and 1: two mirror-image beams that cross at (0.5, 0.30), inside
the
fine region, and leave through the top ~0.33 from the side walls. The medium is a nearly
transparent absorber (sigma_a = 6.4e-7) with no emission; there is no fluid, the pgen
enrolls
hooks for both, and its boundary function injects the beams on the device.

The pgen's final check walks the exit face and writes sc_beam-errs.dat:
  Nx1 Nx2 Ncycle [integral/expected centroid_err/dx rms_width/dx] x 2 beams  offbeam/peak
   niter
  integral/expected  integral of J across the exit slice over w_iang * dx, what entered:
  the
                     medium is transparent, so interpolation may smear a beam but
                     conserves it.
                     Measured 2026-09-11: 0.9999996 uniform, 0.9765 with SMR. The loss is
                     the
                     coarse-to-fine prolongation of a beam that is still ~1 root cell
                     wide when
                     it reaches the fine region: 2.35% after 8 coarse rows of travel,
                     0.87% after
                     16, 0.36% after 24, independent of the band's height (the
                     fine-to-coarse
                     restriction is an average and conserves). Tolerance 5%.
  centroid_err/dx    beam centroid minus the geometric exit point, in root cells. Measured
                     6e-6 uniform, 0.10 SMR. Tolerance 1.
  offbeam/peak       largest J farther than `offbeam_margin` (16) root cells from both
  exit
                     points, over the peak: stray light. Measured 0. Threshold 1e-3.
The two beams are exact mirror images (the mesh, the refined band and the entry columns
are
symmetric about x = 0.5), so their integrals must agree and their centroid errors be
opposite,
to rounding: rays of different octants sweep in different orders yet must not interact.
Refinement sharpens the beams (rms width 2.14 -> 1.87 root cells), which none of the
checked
quantities depend on. Both sweep kernels run and must produce the identical error line.
"""
import test_suite.testutils as testutils

_INPUT = "inputs/sc_beam_smr.athinput"
_ERRS = "sc_beam-errs.dat"
_B1, _B2, _OFFBEAM = (3, 4), (6, 7), 9
_KERNELS = [("wavefront", 0), ("tiled", 8), ("plane", 0)]


def test_sc_beam_smr():
    testutils.cleanup()
    try:
        lines = {}
        for kernel, tile in _KERNELS:
            assert testutils.run(_INPUT, [f"nr_radiation/sweep_kernel={kernel}",
                                          f"nr_radiation/tile_size={tile}"]), \
                f"beam SMR run failed: {kernel}"
            row = testutils.athena_read.error_dat(_ERRS)[-1]
            for name, (c_ratio, c_cen) in (("beam 1", _B1), ("beam 2", _B2)):
                assert abs(row[c_ratio] - 1.0) < 0.05, \
                    (f"{kernel} {name}: exit integral / injected = {row[c_ratio]:.5f}, "
                     f"not conserved")
                assert abs(row[c_cen]) < 1.0, \
                    (f"{kernel} {name}: centroid {row[c_cen]:.3f} root cells from the "
                     f"geometric exit")
            assert row[_OFFBEAM] < 1.0e-3, \
                f"{kernel}: stray light off the beams, max/peak = {row[_OFFBEAM]:.2e}"
            # mirror symmetry of the two beams (independence of the octants)
            assert abs(row[_B1[0]] - row[_B2[0]]) < 1.0e-12, \
                (f"{kernel}: beam integrals differ: {row[_B1[0]]:.12g} vs "
                 f"{row[_B2[0]]:.12g}")
            assert abs(row[_B1[1]] + row[_B2[1]]) < 1.0e-12, \
                (f"{kernel}: centroid errors not mirror images: {row[_B1[1]]:.6g}, "
                 f"{row[_B2[1]]:.6g}")
            lines[kernel] = open(_ERRS).read().splitlines()[-1]
        # every kernel runs the same sweep, so their error lines must agree exactly
        for kern in lines:
            assert lines[kern] == lines["wavefront"], \
                (f"{kern} differs from wavefront:\n  wavefront: "
                 f"{lines['wavefront']}\n  {kern}: {lines[kern]}")
    finally:
        testutils.cleanup()
