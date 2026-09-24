"""
Jeans wave growth-rate validation tests for GPU.

Tests the coupled hydro + self-gravity system by verifying that the measured
growth rate (unstable) or oscillation frequency (stable) converges to the
analytical dispersion relation omega^2 = k^2*cs^2*(1 - n_jeans^2).

Parametrized over MG solver configuration (nsmooth, fmg vs mgi) and
stability regime (n_jeans < 1 stable, n_jeans > 1 unstable).

Each test checks both a threshold on the relative omega error at the highest
resolution and convergence of that error across resolutions.

An additional AMR test runs the unstable case with adaptive mesh refinement
and verifies that the mesh actually adapts during the simulation while still
recovering the correct growth rate.
"""

import pytest
from test_suite.multigrid.mg_utils import (
    run_athenak, assert_jeans_growth_rate, assert_solver_convergence,
    parse_jeans_omega, parse_amr_block_counts, cleanup,
)

input_file = "inputs/jeans_wave.athinput"


def _growth_rate_flags(res, n_jeans, nsmooth, fmg):
    """Flags for Jeans wave growth-rate test at a given resolution."""
    mb = max(res // 4, 8)
    return [
        f"mesh/nx1={res}", f"mesh/nx2={res//2}", f"mesh/nx3={res//2}",
        f"meshblock/nx1={mb}", f"meshblock/nx2={mb}", f"meshblock/nx3={mb}",
        "mesh_refinement/refinement=none",
        "time/nlim=-1",
        "time/tlim=0.06",
        "time/cfl_number=0.3",
        f"problem/n_jeans={n_jeans}",
        "problem/amp=1.0e-6",
        "gravity/threshold=-1",
        "gravity/niteration=2",
        f"gravity/npresmooth={nsmooth}",
        f"gravity/npostsmooth={nsmooth}",
        f"gravity/full_multigrid={'true' if fmg else 'false'}",
        "gravity/show_defect=0",
    ]


@pytest.mark.parametrize("njeans", [0.5, 2.0])
@pytest.mark.parametrize("method", ["fmg", "mgi"])
def test_jeans_growth_rate_gpu(njeans, method):
    """Jeans wave growth rate / frequency should converge to analytical."""
    fmg = method == "fmg"
    assert_jeans_growth_rate(
        input_file,
        lambda res: _growth_rate_flags(res, njeans, 2, fmg),
        res_list=[32, 64],
        max_rel_error=0.01,
        max_ratio=0.35,
        label=f"jeans_growth_njeans{njeans}_{method}_gpu: ",
    )


# ---------------------------------------------------------------------------
# SolveIterative: verify solver reaches target threshold with good rate
# ---------------------------------------------------------------------------

def test_jeans_solve_iterative_gpu():
    """SolveIterative should reach threshold=1e-8 with good convergence rate."""
    try:
        results = run_athenak(
            f"{input_file}",
            [
                "mesh/nx1=64", "mesh/nx2=32", "mesh/nx3=32",
                "meshblock/nx1=16", "meshblock/nx2=16", "meshblock/nx3=16",
                "mesh_refinement/refinement=none",
                "time/nlim=1",
                "time/tlim=1.0",
                "time/cfl_number=0.3",
                "problem/n_jeans=2.0",
                "problem/amp=1.0e-6",
                "gravity/threshold=1.0e-8",
                "gravity/niteration=-1",
                "gravity/npresmooth=1",
                "gravity/npostsmooth=1",
                "gravity/full_multigrid=true",
                "gravity/show_defect=2",
            ],
        )
        assert results[0], "SolveIterative Jeans GPU run failed"
        assert_solver_convergence(
            results[1],
            threshold=1.0e-8,
            max_iterations=6,
            max_avg_ratio=0.11,
            label="jeans_solve_iterative_gpu: ",
        )
    finally:
        cleanup()

# ---------------------------------------------------------------------------
# AMR
# ---------------------------------------------------------------------------


def test_jeans_amr_gpu():
    """Jeans wave with AMR and background velocity: mesh should adapt
    (creating and destroying blocks as the wave sweeps) and growth rate converge."""
    try:
        results = run_athenak(
            f"{input_file}",
            [
                "mesh/nx1=64", "mesh/nx2=32", "mesh/nx3=32",
                "meshblock/nx1=8", "meshblock/nx2=8", "meshblock/nx3=8",
                "mesh_refinement/refinement=adaptive",
                "mesh_refinement/num_levels=2",
                "time/nlim=-1",
                "time/tlim=0.5",
                "time/cfl_number=0.3",
                "problem/n_jeans=1.01",
                "problem/amp=1.0e-3",
                "problem/v0=0.4",
                "amr_criterion0/value_max=1.0005",
                "gravity/threshold=-1",
                "gravity/niteration=5",
                "gravity/npresmooth=2",
                "gravity/npostsmooth=2",
                "gravity/full_multigrid=true",
                "gravity/show_defect=0",
                "output1/dt=-1",
                "output2/dt=-1",
            ],
        )
        assert results[0], "Jeans AMR GPU run failed"

        amr = parse_amr_block_counts(results[1])
        assert amr is not None, "No AMR block counts in output"
        assert amr["created"] > 0, (
            f"AMR did not adapt: {amr['created']} blocks created")
        assert amr["deleted"] > 0, (
            f"AMR did not destroy blocks: {amr['deleted']} blocks deleted")

        omega_data = parse_jeans_omega(results[1])
        assert omega_data is not None, "No omega data in output"
        rel_err = (abs(omega_data["measured"] - omega_data["analytical"])
                   / omega_data["analytical"])
        max_rel_error = 0.1
        assert rel_err < max_rel_error, (
            f"Omega relative error {rel_err:.4e} exceeds {max_rel_error:.4e}")
    finally:
        cleanup()
