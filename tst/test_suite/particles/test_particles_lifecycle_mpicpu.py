"""Two-rank CPU/MPI regression for deferred particle deletion after migration."""

from pathlib import Path
import shutil

import numpy as np

import test_suite.testutils as testutils
from test_suite.particles.test_particles_lifecycle_gpu import (
    PACTIVE,
    PDELETE_AFTER_SNAPSHOT,
    _lifecycle_input,
    _sorted_snapshot,
)


def test_particle_lifecycle_mpicpu(tmp_path):
    """Migrate a deferred particle, snapshot it, and purge it on its new rank."""
    basename = "particle_lifecycle_mpicpu"
    history = Path(f"{basename}.user.hst")
    shutil.rmtree("pvtk", ignore_errors=True)
    history.unlink(missing_ok=True)
    try:
        input_file = _lifecycle_input(
            tmp_path, "inputs/particle_migration.athinput", "particle_lifecycle_mpi"
        )
        assert testutils.mpi_run(
            input_file,
            [
                f"job/basename={basename}",
                "problem/delete_after_snapshot_tag=3",
                "time/nlim=2",
                "time/tlim=1.0",
                "output1/dcycle=0",
            ],
            threads=2,
        ), "two-rank deferred particle lifecycle run failed"

        deferred_points, deferred_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        )
        np.testing.assert_array_equal(deferred_fields["ptag"], np.arange(8))
        expected_status = np.full(8, PACTIVE, dtype=np.int32)
        expected_status[3] = PDELETE_AFTER_SNAPSHOT
        np.testing.assert_array_equal(deferred_fields["status"], expected_status)
        assert deferred_points[3, 0] > 0.0

        _, final_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00002.part.vtk")
        )
        np.testing.assert_array_equal(final_fields["ptag"], [0, 1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(
            final_fields["status"], np.full(7, PACTIVE, dtype=np.int32)
        )
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        history.unlink(missing_ok=True)


def test_particle_boundary_lifecycle_mpicpu(tmp_path):
    """Retain an off-mesh deferred particle on its last rank for one snapshot."""
    basename = "particle_boundary_lifecycle_mpicpu"
    history = Path(f"{basename}.user.hst")
    shutil.rmtree("pvtk", ignore_errors=True)
    history.unlink(missing_ok=True)
    try:
        input_file = _lifecycle_input(
            tmp_path,
            "inputs/particle_migration.athinput",
            "particle_boundary_lifecycle_mpi",
        )
        assert testutils.mpi_run(
            input_file,
            [
                f"job/basename={basename}",
                "problem/delete_after_snapshot_tag=7",
                "mesh/x1min=-0.7005",
                "mesh/x1max=0.7005",
                "mesh/ix1_bc=outflow",
                "mesh/ox1_bc=outflow",
                "time/nlim=2",
                "time/tlim=1.0",
                "output1/dcycle=0",
            ],
            threads=2,
        ), "two-rank deferred particle boundary deletion run failed"

        deferred_points, deferred_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        )
        _, final_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00002.part.vtk")
        )

        np.testing.assert_array_equal(deferred_fields["ptag"], np.arange(8))
        expected_status = np.full(8, PACTIVE, dtype=np.int32)
        expected_status[7] = PDELETE_AFTER_SNAPSHOT
        np.testing.assert_array_equal(deferred_fields["status"], expected_status)
        assert deferred_points[7, 0] >= 0.7005
        np.testing.assert_array_equal(final_fields["ptag"], np.arange(7))
        np.testing.assert_array_equal(
            final_fields["status"], np.full(7, PACTIVE, dtype=np.int32)
        )
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        history.unlink(missing_ok=True)
