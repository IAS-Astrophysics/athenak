"""Two-rank versions of the particle startup and snapshot-selection checks."""

import pytest

from test_suite.particles.test_particles_startup_gpu import (
    _check_particle_amr_gate,
    _check_particle_vtk_gid_gate,
)


@pytest.mark.parametrize("input_name", ("particle_migration", "particle_lagrangian_mc"))
@pytest.mark.parametrize("refinement", ("none", "static", "adaptive"))
def test_particle_amr_gate_mpicpu(tmp_path, monkeypatch, input_name, refinement):
    """Reject unsupported refinement without stranding another rank at startup."""
    _check_particle_amr_gate(tmp_path, monkeypatch, input_name, refinement, mpi=True)


@pytest.mark.parametrize(
    "gid", (None, -1, 0, 1), ids=("default", "all", "block0", "block1")
)
def test_particle_vtk_gid_gate_mpicpu(tmp_path, monkeypatch, gid):
    """Keep all-block snapshots working on two ranks and reject specific block IDs."""
    _check_particle_vtk_gid_gate(tmp_path, monkeypatch, gid, mpi=True)
