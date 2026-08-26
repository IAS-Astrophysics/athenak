"""
Jeans wave growth-rate validation tests for CPU.

Tests the coupled hydro + self-gravity system by verifying that the measured
growth rate (unstable) or oscillation frequency (stable) converges to the
analytical dispersion relation omega^2 = k^2*cs^2*(1 - n_jeans^2).

Each test checks both a threshold on the relative omega error at the highest
resolution and convergence of that error across resolutions.
All tests are 3D with uniform mesh on CPU.
"""

import pytest

from test_suite.multigrid.mg_utils import (
    run_athenak, assert_jeans_growth_rate, assert_solver_convergence, cleanup,
)


def _growth_rate_flags(res, n_jeans, fmg):
    """Flags for Jeans wave growth-rate test at a given resolution."""
    mb = max(res // 4, 8)
    return [
        f"mesh/nx1={res}", f"mesh/nx2={res//2}", f"mesh/nx3={res//2}",
        f"meshblock/nx1={mb}", f"meshblock/nx2={mb}", f"meshblock/nx3={mb}",
        "mesh_refinement/refinement=none",
        "time/nlim=-1",
        "time/tlim=0.1",
        "time/cfl_number=0.3",
        f"problem/n_jeans={n_jeans}",
        "problem/amp=1.0e-6",
        "gravity/threshold=-1",
        "gravity/niteration=4",
        "gravity/npresmooth=2",
        "gravity/npostsmooth=2",
        f"gravity/full_multigrid={fmg}",
        "gravity/show_defect=0",
    ]


@pytest.mark.parametrize("method", ["fmg", "mgi"])
def test_jeans_stable_convergence_cpu(method):
    """Stable Jeans wave: oscillation frequency should converge to analytical."""
    fmg = method == "fmg"
    assert_jeans_growth_rate(
        "inputs/jeans_wave.athinput",
        lambda res: _growth_rate_flags(res, 0.5, fmg),
        res_list=[32, 64],
        max_rel_error=0.01,
        max_ratio=0.3,
        label=f"jeans_stable_{method}_cpu: ",
    )


@pytest.mark.parametrize("method", ["fmg", "mgi"])
def test_jeans_unstable_convergence_cpu(method):
    """Unstable Jeans wave: growth rate should converge to analytical."""
    fmg = method == "fmg"
    assert_jeans_growth_rate(
        "inputs/jeans_wave.athinput",
        lambda res: _growth_rate_flags(res, 2.0, fmg),
        res_list=[16, 32],
        max_rel_error=0.03,
        max_ratio=0.3,
        label=f"jeans_unstable_{method}_cpu: ",
    )


# ---------------------------------------------------------------------------
# SolveIterative: verify solver reaches target threshold with good rate
# ---------------------------------------------------------------------------

def test_jeans_solve_iterative_cpu():
    """SolveIterative should reach threshold=1e-8 with good convergence rate."""
    try:
        results = run_athenak(
            "inputs/jeans_wave.athinput",
            [
                "mesh/nx1=32", "mesh/nx2=16", "mesh/nx3=16",
                "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
                "mesh_refinement/refinement=none",
                "time/nlim=1",
                "time/tlim=1.0",
                "time/cfl_number=0.3",
                "problem/n_jeans=2.0",
                "problem/amp=1.0e-6",
                "gravity/threshold=1.0e-8",
                "gravity/niteration=40",
                "gravity/npresmooth=2",
                "gravity/npostsmooth=2",
                "gravity/full_multigrid=true",
                "gravity/fmg_ncycle=1",
                "gravity/show_defect=2",
            ],
        )
        assert results[0], "SolveIterative Jeans CPU run failed"
        assert_solver_convergence(
            results[1],
            threshold=1.0e-8,
            max_iterations=4,
            max_avg_ratio=0.05,
            label="jeans_solve_iterative_cpu: ",
        )
    finally:
        cleanup()
