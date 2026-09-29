"""
SC transport test: homogeneous emitting/absorbing sphere (Davis, Stone & Jiang 2012).

A uniform sphere with constant absorption and source sits in vacuum. The pgen enrolls
opacity
and emission hooks, the run goes through the ordinary driver for one cycle (solve, ghost
exchange, moments), and the final check compares the numeric J, H and K over the interior
of
the sphere to the analytic angular quadrature of I = b [1 - exp(-tau)] with the SAME rays
and
weights, so what remains is the transport + interpolation error. Errors go to
sc_sphere-errs.dat: Nx1 Nx2 Nx3 Ncycle RMS-L1(J) L-infty(J) RMS(H)/J RMS(K)/J.

Parametrized over
  ndim         2 (an infinite cylinder: rays keep their mu_z, see the pgen) and 3
  sweep_kernel plane and tiled (tile_size 8): same sweep, different kernel packaging
Each (ndim, kernel) runs a resolution ladder on ONE meshblock and asserts that J, H and K
converge: monotone decrease, finest-grid floor, observed order > 0.8 (the bilinear sweep
is
~1st order here, limited by the staircased sphere surface). The finest resolution is then
rerun on 2^ndim meshblocks and must reproduce the one-block error line EXACTLY: with a
fixed
source the block iteration converges to the single-block answer bit for bit (the ghost
exchange is a copy on a uniform mesh), so this certifies the exchange and the iteration.
"""

import math
import pytest
import test_suite.testutils as testutils

_RES = [32, 64, 128]
_COLS = {"J": 4, "H": 6, "K": 7}   # columns of sc_sphere-errs.dat
# finest-grid RMS errors, measured 2026-09-11: 1.6e-3/1.0e-3/1.0e-3 (2D),
# 2.9e-3/9.6e-4/1.7e-3 (3D), order ~1.2
_FLOOR = {2: {"J": 4.0e-3, "H": 2.0e-3, "K": 2.0e-3},
          3: {"J": 4.0e-3, "H": 1.5e-3, "K": 2.5e-3}}
_KERNELS = [("plane", 0), ("tiled", 8)]


def _args(res, ndim, blocks_per_dim, kernel, tile):
    nx3 = res if ndim == 3 else 1
    mb3 = res // blocks_per_dim if ndim == 3 else 1
    return [
        f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={nx3}",
        f"meshblock/nx1={res // blocks_per_dim}",
        f"meshblock/nx2={res // blocks_per_dim}",
        f"meshblock/nx3={mb3}",
        f"nr_radiation/sweep_kernel={kernel}", f"nr_radiation/tile_size={tile}",
    ]


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kernel,tile", _KERNELS)
def test_sc_sphere(ndim, kernel, tile):
    input_file = "inputs/sc_sphere.athinput"
    testutils.cleanup()
    try:
        for res in _RES:
            assert testutils.run(input_file, _args(res, ndim, 1, kernel, tile)), \
                f"run failed: {ndim}D {kernel} nx={res}"
        data = testutils.athena_read.error_dat("sc_sphere-errs.dat")
        assert len(data) == len(_RES)
        for name, col in _COLS.items():
            err = [row[col] for row in data]
            assert all(err[i + 1] < err[i] for i in range(len(err) - 1)), \
                f"{ndim}D {kernel}: {name} error not monotonically decreasing: {err}"
            assert err[-1] < _FLOOR[ndim][name], \
                f"{ndim}D {kernel}: {name} error at nx={_RES[-1]} too large: {err[-1]:g}"
            order = math.log(err[0] / err[-1]) / math.log(_RES[-1] / _RES[0])
            assert order > 0.8, \
                (f"{ndim}D {kernel}: {name} converges too slowly, observed order "
                 f"{order:.2f}")

        # the same problem on 2^ndim meshblocks must give the identical error line
        one_block = open("sc_sphere-errs.dat").read().splitlines()[-1]
        testutils.cleanup()
        assert testutils.run(input_file, _args(_RES[-1], ndim, 2, kernel, tile)), \
            f"run failed: {ndim}D {kernel} nx={_RES[-1]} on {2**ndim} blocks"
        multi_block = open("sc_sphere-errs.dat").read().splitlines()[-1]
        assert multi_block == one_block, \
            f"{ndim}D {kernel}: {2**ndim}-block result differs from 1 block:\n" \
            f"  1 block:  {one_block}\n  {2**ndim} blocks: {multi_block}"
    finally:
        testutils.cleanup()
