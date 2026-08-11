"""
Binary gravity multigrid tests for CPU + MPI.

Same structure as the GPU binary gravity suite but with reduced mesh sizes
(32^3 uniform, 16^3+1-level SMR) and MPI parallelism (4 ranks).

Tests include:
  - Defect convergence per V-cycle (uniform, SMR)
  - Defect consistency: uniform 32^3 vs 1-level SMR 16^3 base (same
    effective finest resolution)
  - Rank consistency: 4 vs 16 MPI ranks on uniform and SMR meshes

show_defect modes: 1 = final only, 2 = per-iteration (initial + each V-cycle).
"""

import pytest

from test_suite.multigrid.mg_utils import (
    run_athenak, assert_solver_convergence,
    parse_final_defect, assert_defect_consistency, cleanup,
)

threshold = 1E-10

_GRAVITY_FLAGS = [
    "time/nlim=1",
    f"gravity/threshold={threshold}",
    "gravity/npresmooth=2",
    "gravity/npostsmooth=2",
    "gravity/full_multigrid=true",
    "gravity/fmg_ncycle=1",
]

_stored_defects = {}


# ---------------------------------------------------------------------------
# Defect convergence per V-cycle (uniform, SMR)
# ---------------------------------------------------------------------------

def test_binary_gravity_uniform_mpicpu():
    """Binary gravity MG defect convergence on 32^3 uniform mesh (4 MPI ranks)."""
    try:
        results = run_athenak(
            "inputs/binary_gravity.athinput",
            [
                "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=32",
                "meshblock/nx1=16", "meshblock/nx2=16", "meshblock/nx3=16",
                "mesh_refinement/refinement=none",
                "gravity/show_defect=2",
            ] + _GRAVITY_FLAGS,
            mpi=True, threads=4,
        )
        assert results[0], "Binary gravity uniform MPI run failed"
        assert_solver_convergence(results[1], threshold, max_iterations=9,
                                  max_avg_ratio=0.05,
                                  label="binary_gravity_uniform_mpicpu: ")
        d = parse_final_defect(results[1])
        if d is not None:
            _stored_defects["uniform"] = d
    finally:
        cleanup()


def test_binary_gravity_smr_mpicpu():
    """Binary gravity with 1-level SMR (16^3 base, finest = 32^3, 4 MPI ranks)."""
    try:
        results = run_athenak(
            "inputs/binary_gravity.athinput",
            [
                "mesh/nx1=16", "mesh/nx2=16", "mesh/nx3=16",
                "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
                "refined_region2/level=1",
                "refined_region3/level=1",
                "refined_region4/level=1",
                "gravity/show_defect=2",
            ] + _GRAVITY_FLAGS,
            mpi=True, threads=4,
        )
        assert results[0], "Binary gravity SMR MPI run failed"
        assert_solver_convergence(results[1], threshold, max_iterations=9,
                                  max_avg_ratio=0.05,
                                  label="binary_gravity_smr_mpicpu: ")
        d = parse_final_defect(results[1])
        if d is not None:
            _stored_defects["smr"] = d
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Defect consistency: uniform vs SMR at same effective resolution
# ---------------------------------------------------------------------------

def test_binary_gravity_defect_consistency_mpicpu():
    """Final defect should be comparable between uniform 32^3 and 1-level SMR (MPI)."""
    if "uniform" not in _stored_defects or "smr" not in _stored_defects:
        pytest.skip("Prerequisite tests did not run or failed")
    assert_defect_consistency(
        [_stored_defects["uniform"], _stored_defects["smr"]],
        max_spread=0.5,
        label="binary_gravity_defect_consistency_mpicpu: ",
    )


# ---------------------------------------------------------------------------
# Rank consistency: 4 vs 16 MPI ranks on uniform and SMR meshes
# ---------------------------------------------------------------------------

def test_binary_gravity_rank_consistency_mpicpu():
    """Final defect should match between 4 and 16 MPI ranks."""
    small_mb_flags = [
        "mesh/nx1=32", "mesh/nx2=32", "mesh/nx3=32",
        "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
        "time/nlim=1",
        "gravity/threshold=-1",
        "gravity/niteration=20",
        "gravity/npresmooth=2",
        "gravity/npostsmooth=2",
        "gravity/full_multigrid=true",
        "gravity/fmg_ncycle=1",
        "gravity/show_defect=1",
    ]

    for mesh_label, extra_flags in [
        ("uniform", ["mesh_refinement/refinement=none"]),
        ("smr", []),
    ]:
        defects = []
        for nranks in [4, 16]:
            try:
                results = run_athenak(
                    "inputs/binary_gravity.athinput",
                    small_mb_flags + extra_flags,
                    mpi=True, threads=nranks,
                )
                assert results[0], (
                    f"Run failed for {mesh_label} with {nranks} ranks")
                d = parse_final_defect(results[1])
                assert d is not None, (
                    f"No final defect for {mesh_label} with {nranks} ranks")
                defects.append(d)
            finally:
                cleanup()
        assert_defect_consistency(
            defects, max_spread=0.5,
            label=f"rank_consistency_{mesh_label}_mpicpu: ")
