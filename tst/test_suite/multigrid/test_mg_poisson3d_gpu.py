"""
Multigrid Poisson solver self-gravity tests for GPU.

Tests MG defect convergence on uniform meshes with hydro and MHD,
and decomposition-independence of the final defect.

show_defect modes: 1 = final only, 2 = per-iteration (initial + each V-cycle).
"""

from test_suite.multigrid.mg_utils import (
    run_athenak, assert_solver_convergence,
    parse_final_defect, assert_defect_consistency, cleanup,
)

threshold = 1E-8


def _selfgravity_flags_gpu(res, mb=None, soe="hydro"):
    """Common flags for selfgravity defect tests on GPU."""
    if mb is None:
        mb = max(res // 4, 8)
    return [
        f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={res}",
        f"meshblock/nx1={mb}", f"meshblock/nx2={mb}", f"meshblock/nx3={mb}",
        "time/nlim=1",
        "gravity/show_defect=2",
        f"gravity/threshold={threshold}",
        "gravity/npresmooth=2",
        "gravity/npostsmooth=2",
        "gravity/full_multigrid=true",
    ]


def test_selfgravity_uniform_hydro_gpu():
    """MG defect should converge to near machine precision on 64^3 hydro mesh."""
    try:
        results = run_athenak(
            "inputs/selfgravity.athinput",
            _selfgravity_flags_gpu(64),
        )
        assert results[0], "Selfgravity hydro GPU run failed"
        assert_solver_convergence(
            results[1], threshold, max_iterations=10,
            max_avg_ratio=0.07, label="selfgravity_uniform_hydro_gpu: ")
    finally:
        cleanup()


def test_selfgravity_uniform_mhd_gpu():
    """MG defect should converge to near machine precision on 64^3 MHD mesh."""
    try:
        results = run_athenak(
            "inputs/selfgravity_mhd.athinput",
            _selfgravity_flags_gpu(64, soe="mhd"),
        )
        assert results[0], "Selfgravity MHD GPU run failed"
        assert_solver_convergence(
            results[1], threshold, max_iterations=10,
            max_avg_ratio=0.07, label="selfgravity_uniform_mhd_gpu: ")
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Decomposition-independence: final defect must match across meshblock sizes
# ---------------------------------------------------------------------------

def test_selfgravity_decomposition_consistency_gpu():
    """Final defect must be independent of meshblock decomposition (64^3 mesh)."""
    res = 64
    defects = []
    for mb in [64, 32, 16, 8]:
        try:
            results = run_athenak(
                "inputs/selfgravity.athinput",
                _selfgravity_flags_gpu(res, mb=mb) + ["gravity/show_defect=1"],
            )
            assert results[0], f"Run failed for mb={mb}"
            d = parse_final_defect(results[1])
            assert d is not None, f"No final defect for mb={mb}"
            defects.append(d)
        finally:
            cleanup()
    assert_defect_consistency(defects, max_spread=1E-8,
                              label="selfgravity_decomposition_gpu: ")
