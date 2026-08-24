"""Two-rank CPU/MPI regression for particle snapshot output."""

from pathlib import Path
import shutil

import numpy as np

import test_suite.testutils as testutils
from test_suite.particles.test_particles_snapshot_gpu import (
    _read_particle_vtk,
    _snapshot_input,
)


def test_particle_snapshot_mpicpu(tmp_path):
    """Write one valid snapshot from ranks with unequal particle counts."""
    basename = "particle_snapshot_mpicpu"
    history = Path(f"{basename}.user.hst")
    shutil.rmtree("pvtk", ignore_errors=True)
    history.unlink(missing_ok=True)

    try:
        input_file = _snapshot_input(
            tmp_path, "inputs/particle_migration.athinput", "particle_snapshot_mpi"
        )
        assert testutils.mpi_run(
            input_file,
            [f"job/basename={basename}", "problem/delete_tag=3"],
            threads=2,
        ), "two-rank particle snapshot run failed"

        snapshot = Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        assert snapshot.exists(), "two-rank particle snapshot was not written"
        points, fields, types = _read_particle_vtk(snapshot)

        assert points.shape == (7, 3)
        assert list(fields) == [
            "ptag", "status", "vx", "vy", "vz", "radius_squared", "vtk_y",
        ]
        assert types == {
            "ptag": "int",
            "status": "int",
            "vx": "float",
            "vy": "float",
            "vz": "float",
            "radius_squared": "float",
            "vtk_y": "float",
        }

        order = np.argsort(fields["ptag"])
        tags = fields["ptag"][order]
        np.testing.assert_array_equal(tags, [0, 1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(fields["status"][order], np.zeros(7, dtype=np.int32))

        vx = np.full(7, 0.01)
        vx[tags == 4] = -0.20
        vy = -0.08 + 0.005*tags
        vz = 0.03 - 0.002*tags
        np.testing.assert_allclose(fields["vx"][order], vx)
        np.testing.assert_allclose(fields["vy"][order], vy)
        np.testing.assert_allclose(fields["vz"][order], vz)

        initial_x = -0.70 + 0.20*tags
        initial_x[tags == 4] = 0.004
        move_time = 0.05
        np.testing.assert_allclose(points[order, 0], initial_x + move_time*vx)
        np.testing.assert_allclose(points[order, 1], -0.20 + 0.04*tags + move_time*vy)
        np.testing.assert_allclose(points[order, 2], -0.10 + 0.02*tags + move_time*vz)
        np.testing.assert_allclose(
            fields["radius_squared"][order], np.sum(points[order]**2, axis=1), atol=1e-7
        )
        np.testing.assert_allclose(fields["vtk_y"][order], points[order, 1])
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        history.unlink(missing_ok=True)


def test_particle_snapshot_empty_rank_mpicpu(tmp_path):
    """Write a valid snapshot when one MPI rank owns no particles."""
    basename = "particle_snapshot_empty_rank_mpicpu"
    history = Path(f"{basename}.user.hst")
    shutil.rmtree("pvtk", ignore_errors=True)
    history.unlink(missing_ok=True)

    try:
        input_path = Path(
            _snapshot_input(
                tmp_path,
                "inputs/particle_migration.athinput",
                "particle_snapshot_empty_rank",
            )
        )
        input_path.write_text(
            input_path.read_text().replace(
                "pgen_name = particle_drift\n",
                "pgen_name = particle_drift\nexpected_particles = 1\n",
                1,
            )
        )
        assert testutils.mpi_run(
            str(input_path),
            [
                f"job/basename={basename}",
                "mesh/nx1=12",
                "mesh/x1max=0.8",
                "particles/ppc=0.001953125",
                "problem/migration_test=false",
                "time/nlim=0",
            ],
            threads=2,
        ), "empty-rank particle snapshot run failed"

        snapshot = Path(f"pvtk/{basename}.prtcl_all.00000.part.vtk")
        assert snapshot.exists(), "empty-rank particle snapshot was not written"
        points, fields, types = _read_particle_vtk(snapshot)

        assert points.shape == (1, 3)
        assert list(fields) == [
            "ptag", "status", "vx", "vy", "vz", "radius_squared", "vtk_y",
        ]
        assert types == {
            "ptag": "int",
            "status": "int",
            "vx": "float",
            "vy": "float",
            "vz": "float",
            "radius_squared": "float",
            "vtk_y": "float",
        }
        np.testing.assert_array_equal(fields["ptag"], [0])
        np.testing.assert_array_equal(fields["status"], [0])
        np.testing.assert_allclose(fields["vx"], [0.10])
        np.testing.assert_allclose(fields["vy"], [-0.08])
        np.testing.assert_allclose(fields["vz"], [0.03])
        np.testing.assert_allclose(points, [[-0.35, -0.20, -0.10]])
        np.testing.assert_allclose(fields["radius_squared"], np.sum(points**2, axis=1))
        np.testing.assert_allclose(fields["vtk_y"], points[:, 1])
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        history.unlink(missing_ok=True)
