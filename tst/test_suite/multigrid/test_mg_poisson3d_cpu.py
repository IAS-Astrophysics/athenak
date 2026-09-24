"""
Multigrid Poisson solver self-gravity tests for CPU.

Tests MG defect convergence and decomposition-independence on uniform meshes.

show_defect modes: 1 = final only, 2 = per-iteration (initial + each V-cycle).
"""

from test_suite.multigrid.mg_utils import (
    run_athenak, assert_solver_convergence,
    parse_final_defect, assert_defect_consistency, cleanup,
)

threshold = 1E-8


def _selfgravity_flags_cpu(res, mb=None, soe="hydro"):
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


res = 64


def test_selfgravity_uniform_hydro_cpu():
    """MG defect should converge to near machine precision on hydro mesh."""
    try:
        results = run_athenak(
            "inputs/selfgravity.athinput",
            _selfgravity_flags_cpu(res),
        )
        assert results[0], "Selfgravity hydro GPU run failed"
        assert_solver_convergence(
            results[1], threshold, max_iterations=10,
            max_avg_ratio=0.07, label="selfgravity_uniform_hydro_cpu: ")
    finally:
        cleanup()


def test_selfgravity_uniform_mhd_cpu():
    """MG defect should converge to near machine precision on 64^3 MHD mesh."""
    try:
        results = run_athenak(
            "inputs/selfgravity_mhd.athinput",
            _selfgravity_flags_cpu(res, soe="mhd"),
        )
        assert results[0], "Selfgravity MHD GPU run failed"
        assert_solver_convergence(
            results[1], threshold, max_iterations=10,
            max_avg_ratio=0.07, label="selfgravity_uniform_mhd_cpu: ")
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Decomposition-independence: final defect must match across meshblock sizes
# ---------------------------------------------------------------------------

def test_selfgravity_decomposition_consistency_cpu():
    """Final defect must be independent of meshblock decomposition (64^3 mesh)."""
    defects = []
    for mb in [res//4, res//2, res//1]:
        try:
            results = run_athenak(
                "inputs/selfgravity.athinput",
                _selfgravity_flags_cpu(res, mb=mb) + ["gravity/show_defect=1"],
            )
            assert results[0], f"Run failed for mb={mb}"
            d = parse_final_defect(results[1])
            assert d is not None, f"No final defect for mb={mb}"
            defects.append(d)
        finally:
            cleanup()
    assert_defect_consistency(defects, max_spread=1E-4,
                              label="selfgravity_decomposition_cpu: ")
