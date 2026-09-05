"""Two-rank version of the prescribed Lagrangian MC counterflow regression."""

import pytest

from test_suite.particles.test_particles_lagrangian_mc_counterflow_gpu import (
    _run_counterflow_case,
)


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize("outward_fraction", (0.25, 0.5), ids=("balanced", "unbalanced"))
def test_particle_lagrangian_mc_counterflow_mpicpu(
    tmp_path, monkeypatch, dimension, outward_fraction
):
    """Exercise opposing transfers across an actual MPI rank boundary."""
    _run_counterflow_case(tmp_path, monkeypatch, dimension, outward_fraction, mpi=True)
