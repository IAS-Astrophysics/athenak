"""Particle startup restrictions and supported output/refinement settings."""

from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_suite.particles.test_particles_snapshot_gpu import (
    _read_particle_vtk,
    _snapshot_input,
)


def _run_particle_startup(tmp_path, monkeypatch, input_file, arguments, mpi=False):
    input_file = Path(input_file).resolve()
    executable = Path("athena").resolve()
    monkeypatch.chdir(tmp_path)
    command = [
        str(executable), "-i", str(input_file),
        "job/basename=particle_startup", "time/nlim=0", *arguments,
    ]
    if mpi:
        command = ["mpirun", "-np", "2", *command]
    return subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30,
    )


def _check_particle_amr_gate(tmp_path, monkeypatch, input_name, refinement, mpi=False):
    input_file = tmp_path / f"{input_name}_refinement.athinput"
    input_file.write_text(
        Path(f"inputs/{input_name}.athinput").read_text()
        + f"\n<mesh_refinement>\nrefinement = {refinement}\n"
        + "num_levels = 2\nmax_nmb_per_rank = 32\n"
    )
    result = _run_particle_startup(
        tmp_path, monkeypatch, input_file, [], mpi=mpi,
    )
    output = result.stdout + result.stderr
    if refinement == "adaptive":
        assert result.returncode != 0, output
        assert "Particles do not support adaptive mesh refinement" in output
        assert "Setup complete" not in output
    else:
        assert result.returncode == 0, output
        assert Path("particle_startup.user.hst").exists()


def _check_particle_vtk_gid_gate(tmp_path, monkeypatch, gid, mpi=False):
    # Use two real MeshBlocks so the common parser accepts both requested GIDs.
    input_file = Path(_snapshot_input(
        tmp_path, "inputs/particle_migration.athinput", "particle_vtk_gid",
    ))
    if gid is not None:
        input_file.write_text(input_file.read_text() + f"gid = {gid}\n")
    result = _run_particle_startup(tmp_path, monkeypatch, input_file, [], mpi)
    output = result.stdout + result.stderr
    if gid is not None and gid >= 0:
        assert result.returncode != 0, output
        assert (
            "file_type=pvtk does not support gid filtering; omit gid or use gid=-1"
        ) in output
        assert not list(Path("pvtk").glob("*.part.vtk"))
    else:
        assert result.returncode == 0, output
        points, fields, _ = _read_particle_vtk(
            Path("pvtk/particle_startup.prtcl_all.00000.part.vtk")
        )
        assert points.shape == (8, 3)
        np.testing.assert_array_equal(np.sort(fields["ptag"]), np.arange(8))


@pytest.mark.parametrize("input_name", ("particle_migration", "particle_lagrangian_mc"))
@pytest.mark.parametrize("refinement", ("none", "static", "adaptive"))
def test_particle_amr_gate_gpu(tmp_path, monkeypatch, input_name, refinement):
    """Both pushers accept fixed meshes and reject adaptive refinement at startup."""
    _check_particle_amr_gate(tmp_path, monkeypatch, input_name, refinement)


@pytest.mark.parametrize("gid", (None, -1, 0, 1), ids=("default", "all", "block0", "block1"))
def test_particle_vtk_gid_gate_gpu(tmp_path, monkeypatch, gid):
    """Reject block selections while preserving default and explicit all-block output."""
    _check_particle_vtk_gid_gate(tmp_path, monkeypatch, gid)
