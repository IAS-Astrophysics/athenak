"""
Multigrid Poisson solver self-gravity tests for CPU + MPI.

Tests MG defect convergence and decomposition-independence with MPI.

show_defect modes: 1 = final only, 2 = per-iteration (initial + each V-cycle).
"""

import pytest

from test_suite.multigrid.mg_utils import (
    run_athenak, assert_solver_convergence,
    parse_final_defect, assert_defect_consistency, cleanup,
)

threshold = 1E-9


def _selfgravity_flags_mpi(res, nsmooth, fmg, mb=None):
    """Common flags for selfgravity MPI tests."""
    if mb is None:
        mb = max(res // 4, 8)
    return [
        f"mesh/nx1={res}", f"mesh/nx2={res}", f"mesh/nx3={res}",
        f"meshblock/nx1={mb}", f"meshblock/nx2={mb}", f"meshblock/nx3={mb}",
        "time/nlim=1",
        f"gravity/threshold={threshold}",
        f"gravity/npresmooth={nsmooth}",
        f"gravity/npostsmooth={nsmooth}",
        f"gravity/full_multigrid={fmg}",
    ]


# ---------------------------------------------------------------------------
# Defect convergence (MPI)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nsmooth", [1, 2])
@pytest.mark.parametrize("method", ["fmg", "mgi"])
def test_selfgravity_uniform_hydro_mpicpu(nsmooth, method):
    """MG defect should converge to near machine precision on 64^3 hydro mesh."""
    try:
        results = run_athenak(
            "inputs/selfgravity.athinput",
            _selfgravity_flags_mpi(64, nsmooth, method == "fmg") +
            ["gravity/show_defect=2"],
            mpi=True, threads=4,
        )
        assert results[0], "Selfgravity hydro MPI run failed"
        assert_solver_convergence(
            results[1], threshold, max_iterations=7,
            max_avg_ratio=0.14, label="selfgravity_uniform_hydro_mpipu: ")
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Decomposition-independence (MPI): final defect across meshblock sizes
# ---------------------------------------------------------------------------

def test_selfgravity_decomposition_consistency_mpicpu():
    """Final defect must be independent of meshblock decomposition (32^3, MPI)."""
    res = 32
    defects = []
    for mb in [32, 16]:
        nranks = (res // mb) ** 3
        try:
            results = run_athenak(
                "inputs/selfgravity.athinput",
                _selfgravity_flags_mpi(res, 1, True, mb=mb) + ["gravity/show_defect=1"],
                mpi=True, threads=nranks,
            )
            assert results[0], f"Run failed for mb={mb}, nranks={nranks}"
            d = parse_final_defect(results[1])
            assert d is not None, f"No final defect for mb={mb}"
            defects.append(d)
        finally:
            cleanup()
    assert_defect_consistency(defects, max_spread=0.01,
                              label="selfgravity_decomposition_mpicpu: ")


# ---------------------------------------------------------------------------
# Rank consistency: same decomposition, different rank counts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("res", [64])
def test_selfgravity_rank_consistency_mpicpu(res):
    """Final defect should match between 1 and 8 MPI ranks (32^3, mb=16)."""
    defects = []
    for nranks in [1, 8]:
        try:
            results = run_athenak(
                "inputs/selfgravity.athinput",
                _selfgravity_flags_mpi(res, 1, 1, mb=res//8) + ["gravity/show_defect=1"],
                mpi=True, threads=nranks,
            )
            assert results[0], f"Run failed with {nranks} ranks"
            d = parse_final_defect(results[1])
            assert d is not None, f"No final defect for {nranks} ranks"
            defects.append(d)
        finally:
            cleanup()
    assert_defect_consistency(defects, max_spread=0.03,
                              label="selfgravity_rank_consistency_mpicpu: ")
