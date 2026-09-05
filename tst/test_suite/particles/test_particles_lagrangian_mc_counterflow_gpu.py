"""Prescribed-flux regression for Lagrangian MC coarse/fine counterflow."""

from pathlib import Path

import numpy as np
import pytest

import test_suite.testutils as testutils
from test_suite.particles.test_particles_snapshot_gpu import _read_particle_vtk


def _move_draw(tag, cycle):
    """Evaluate the tag/cycle RNG independently of any flux or routing code."""
    mask = (1 << 64) - 1
    key = 5 ^ ((int(tag)*0xD2B74407B1CE6E93) & mask)
    key ^= (cycle*0x9E3779B97F4A7C15) & mask
    value = (key + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30))*0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27))*0x94D049BB133111EB) & mask
    value ^= value >> 31
    return (value >> 11)/(1 << 53)


def _read_snapshot(basename, number, count):
    path = Path(f"pvtk/{basename}.prtcl_all.{number:05d}.part.vtk")
    assert path.exists(), f"missing counterflow snapshot: {path}"
    points, fields, _ = _read_particle_vtk(path)
    assert points.shape == (count, 3)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(fields["owner_rank"]))
    assert np.all(fields["owner_rank"] >= 0), "particle is outside its owning block"
    np.testing.assert_array_equal(fields["status"], np.zeros(count, dtype=int))
    order = np.argsort(fields["ptag"])
    tags = fields["ptag"][order]
    np.testing.assert_array_equal(tags, np.arange(count))
    return points[order], fields["owner_rank"][order], tags


def _run_counterflow_case(tmp_path, monkeypatch, dimension, outward_fraction, mpi=False):
    """Check each tag against independently specified coarse/fine transfer probabilities.

    Three initially populated cells have density one and equal-mass tracers. The
    coarse cell has 2**dimension times the volume and particle count of either fine
    cell. Exactly one fine face carries flow in each direction; other faces carry
    zero. This tests the particle operator, not the fluid solver's evolution.
    """
    input_file = Path("inputs/particle_lagrangian_mc_counterflow.athinput").resolve()
    executable = Path("athena").resolve()
    # Keep outputs separate from other tests and from existing user output files.
    monkeypatch.chdir(tmp_path)
    Path("athena").symlink_to(executable)
    basename = "particle_counterflow"
    arguments = [
        f"job/basename={basename}",
        f"problem/outward_fraction={outward_fraction}",
    ]
    if dimension == 3:
        arguments += [
            "mesh/nx3=8", "meshblock/nx3=4", "refined_region1/x3max=0.0",
        ]
    if mpi:
        assert testutils.mpi_run(str(input_file), arguments, threads=2)
    else:
        assert testutils.run(str(input_file), arguments)

    volume_ratio = 2**dimension
    count = (volume_ratio + 2)*256
    # VTK places the inactive 2D coordinate at mesh/x3min, not the stored particle z.
    positions = np.array([
        [0.5, -1.75, -1.75 if dimension == 3 else -2.0],
        [-0.25, -1.875, -1.875 if dimension == 3 else -2.0],
        [-0.25, -1.625, -1.625 if dimension == 3 else -2.0],
    ])
    initial, initial_ranks, tags = _read_snapshot(basename, 0, count)
    cells = np.where(initial[:, 0] > 0.0, 0, np.where(initial[:, 1] < -1.75, 1, 2))
    np.testing.assert_array_equal(initial, positions[cells])
    np.testing.assert_array_equal(np.bincount(cells), [volume_ratio*256, 256, 256])
    cell_ranks = np.array([initial_ranks[cells == cell][0] for cell in range(3)])
    np.testing.assert_array_equal(initial_ranks, cell_ranks[cells])
    if mpi:
        assert cell_ranks[0] != cell_ranks[1], "interface must cross MPI ranks"
        assert cell_ranks[0] != cell_ranks[2], "interface must cross MPI ranks"

    for cycle in range(2):
        # Q/M_coarse = (Q/M_fine)/2**dimension. In particular, balanced
        # transfers still require coarse departures even though the net is zero.
        outward = outward_fraction if cycle == 0 else 0.5*outward_fraction
        fine_receiver, fine_donor = (1, 2) if cycle == 0 else (2, 1)
        probabilities = np.where(
            cells == 0, outward/volume_ratio, np.where(cells == fine_donor, 0.25, 0.0)
        )
        draws = np.array([_move_draw(tag, cycle) for tag in tags])
        moved = draws < probabilities
        coarse_departures = moved & (cells == 0)
        fine_departures = moved & (cells == fine_donor)
        assert np.any(coarse_departures), "fixture must exercise coarse departures"
        assert np.any(fine_departures), "fixture must exercise fine departures"
        cells[coarse_departures] = fine_receiver
        cells[fine_departures] = 0

        points, ranks, _ = _read_snapshot(basename, cycle+1, count)
        np.testing.assert_array_equal(
            points, positions[cells], err_msg=f"wrong particle moves on cycle {cycle}"
        )
        np.testing.assert_array_equal(ranks, cell_ranks[cells])


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize("outward_fraction", (0.25, 0.5), ids=("balanced", "unbalanced"))
def test_particle_lagrangian_mc_counterflow_gpu(
    tmp_path, monkeypatch, dimension, outward_fraction
):
    """Preserve both transfer directions, correct cell volumes, and final ownership."""
    _run_counterflow_case(tmp_path, monkeypatch, dimension, outward_fraction)
