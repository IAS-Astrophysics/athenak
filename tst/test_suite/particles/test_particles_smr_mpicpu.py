"""Two-rank CPU/MPI regression for particle routing across an SMR interface."""

from pathlib import Path
import shutil

import numpy as np

import test_suite.testutils as testutils
from test_suite.particles.test_particles_snapshot_gpu import _read_particle_vtk


def test_particle_smr_mpicpu():
    """Route a fine-to-coarse diagonal particle onto a different MPI rank."""
    basename = "particle_smr_mpicpu"
    shutil.rmtree("pvtk", ignore_errors=True)
    try:
        assert testutils.mpi_run(
            "inputs/particle_smr.athinput",
            [
                f"job/basename={basename}",
                "refined_region1/x2max=0.0",
                "problem/particle_y=-1.001",
                "problem/particle_vy=1.0",
            ],
            threads=2,
        ), "two-rank particle SMR run failed"

        initial = Path(f"pvtk/{basename}.prtcl_all.00000.part.vtk")
        final = Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        initial_points, initial_fields, _ = _read_particle_vtk(initial)
        final_points, final_fields, _ = _read_particle_vtk(final)

        assert initial_points.shape == (1, 3)
        assert final_points.shape == (1, 3)
        np.testing.assert_allclose(initial_points[0, :2], [-0.001, -1.001])
        np.testing.assert_allclose(final_points[0, :2], [0.0615, -0.9385])
        np.testing.assert_array_equal(initial_fields["ptag"], [0])
        np.testing.assert_array_equal(final_fields["ptag"], [0])
        np.testing.assert_allclose(initial_fields["owner_error"], [0.0])
        np.testing.assert_allclose(final_fields["owner_error"], [0.0])
        np.testing.assert_allclose(initial_fields["owner_level"], [1.0])
        np.testing.assert_allclose(final_fields["owner_level"], [0.0])
        assert initial_fields["owner_rank"][0] != final_fields["owner_rank"][0]
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
