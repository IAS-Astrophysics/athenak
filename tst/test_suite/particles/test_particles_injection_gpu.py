"""GPU contract regression for passive particle creation hooks."""

from pathlib import Path
import shutil

import numpy as np

import test_suite.testutils as testutils
from test_suite.particles.test_particles_snapshot_gpu import _read_particle_vtk


def test_particle_injection_gpu():
    """Create particles initially and after a step without moving newborn particles."""
    basename = "particle_injection_gpu"
    shutil.rmtree("pvtk", ignore_errors=True)
    try:
        assert testutils.run(
            "inputs/particle_injection.athinput",
            [f"job/basename={basename}"],
        ), "particle injection run failed"

        initial = Path(f"pvtk/{basename}.prtcl_all.00000.part.vtk")
        final = Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        initial_points, initial_fields, _ = _read_particle_vtk(initial)
        final_points, final_fields, _ = _read_particle_vtk(final)

        np.testing.assert_array_equal(initial_fields["ptag"], [0])
        np.testing.assert_allclose(initial_points, [[-0.5, 0.0, 0.0]])

        order = np.argsort(final_fields["ptag"])
        np.testing.assert_array_equal(final_fields["ptag"][order], [0, 1])
        np.testing.assert_array_equal(final_fields["status"][order], [0, 0])
        np.testing.assert_allclose(final_fields["vx"][order], [0.2, 0.2])
        np.testing.assert_allclose(
            final_points[order], [[-0.4875, 0.0, 0.0], [-0.5, 0.0, 0.0]]
        )
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
