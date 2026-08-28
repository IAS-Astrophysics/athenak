"""GPU regression tests for particle routing across static-refinement boundaries."""

from pathlib import Path
import shutil

import numpy as np
import pytest

import test_suite.testutils as testutils
from test_suite.particles.test_particles_snapshot_gpu import _read_particle_vtk


@pytest.mark.parametrize(
    "name,arguments,initial_xy,final_xy,z_values,initial_level,final_level",
    [
        (
            "fine_to_coarse_face",
            [],
            (-0.001, -0.75),
            (0.0615, -0.75),
            None,
            1,
            0,
        ),
        (
            "coarse_to_fine_face",
            ["problem/particle_x=0.001", "problem/particle_vx=-1.0"],
            (0.001, -0.75),
            (-0.0615, -0.75),
            None,
            0,
            1,
        ),
        (
            "coarse_to_fine_periodic_exact_upper",
            ["problem/particle_x=3.9375"],
            (3.9375, -0.75),
            (-4.0, -0.75),
            None,
            0,
            1,
        ),
        (
            "fine_to_coarse_diagonal",
            [
                "refined_region1/x2max=0.0",
                "problem/particle_y=-0.001",
                "problem/particle_vy=1.0",
            ],
            (-0.001, -0.001),
            (0.0615, 0.0615),
            None,
            1,
            0,
        ),
        (
            "fine_to_coarse_internal_diagonal",
            ["problem/particle_y=-1.001", "problem/particle_vy=1.0"],
            (-0.001, -1.001),
            (0.0615, -0.9385),
            None,
            1,
            0,
        ),
        (
            "fine_to_coarse_periodic_diagonal",
            [
                "mesh/nx2=24",
                "problem/particle_y=1.999",
                "problem/particle_vy=1.0",
            ],
            (-0.001, 1.999),
            (0.04066666666666667, -1.9593333333333334),
            None,
            1,
            0,
        ),
        (
            "fine_to_coarse_corner_3d",
            [
                "mesh/nx3=8",
                "meshblock/nx3=4",
                "refined_region1/x2max=0.0",
                "refined_region1/x3max=0.0",
                "problem/particle_y=-0.001",
                "problem/particle_z=-0.001",
                "problem/particle_vy=1.0",
                "problem/particle_vz=1.0",
            ],
            (-0.001, -0.001),
            (0.03025, 0.03025),
            (-0.001, 0.03025),
            1,
            0,
        ),
    ],
)
def test_particle_smr_gpu(
    name, arguments, initial_xy, final_xy, z_values, initial_level, final_level
):
    """Preserve the particle and assign the leaf block containing its new position."""
    basename = f"particle_smr_{name}_gpu"
    shutil.rmtree("pvtk", ignore_errors=True)
    try:
        assert testutils.run(
            "inputs/particle_smr.athinput",
            [f"job/basename={basename}", *arguments],
        ), f"particle SMR {name} run failed"

        initial = Path(f"pvtk/{basename}.prtcl_all.00000.part.vtk")
        final = Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        initial_points, initial_fields, _ = _read_particle_vtk(initial)
        final_points, final_fields, _ = _read_particle_vtk(final)

        assert initial_points.shape == (1, 3)
        assert final_points.shape == (1, 3)
        np.testing.assert_allclose(initial_points[0, :2], initial_xy, atol=1.0e-7)
        np.testing.assert_allclose(final_points[0, :2], final_xy, atol=1.0e-7)
        if z_values is not None:
            np.testing.assert_allclose(initial_points[0, 2], z_values[0], atol=1.0e-7)
            np.testing.assert_allclose(final_points[0, 2], z_values[1], atol=1.0e-7)
        np.testing.assert_array_equal(initial_fields["ptag"], [0])
        np.testing.assert_array_equal(final_fields["ptag"], [0])
        np.testing.assert_array_equal(initial_fields["status"], [0])
        np.testing.assert_array_equal(final_fields["status"], [0])
        np.testing.assert_allclose(initial_fields["owner_error"], [0.0])
        np.testing.assert_allclose(final_fields["owner_error"], [0.0])
        np.testing.assert_allclose(initial_fields["owner_level"], [initial_level])
        np.testing.assert_allclose(final_fields["owner_level"], [final_level])
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)


def test_particle_smr_nonperiodic_diagonal_deferred_gpu():
    """Do not route or wrap a deferred particle exiting at an SMR corner."""
    basename = "particle_smr_nonperiodic_diagonal_deferred_gpu"
    shutil.rmtree("pvtk", ignore_errors=True)
    try:
        assert testutils.run(
            "inputs/particle_smr.athinput",
            [
                f"job/basename={basename}",
                "mesh/ix1_bc=outflow",
                "mesh/ox1_bc=outflow",
                "refined_region1/x2max=0.0",
                "problem/particle_x=-3.999",
                "problem/particle_y=-0.001",
                "problem/particle_vx=-1.0",
                "problem/particle_vy=1.0",
                "problem/defer_after_push=true",
            ],
        ), "deferred particle SMR boundary run failed"

        initial = Path(f"pvtk/{basename}.prtcl_all.00000.part.vtk")
        final = Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        initial_points, initial_fields, _ = _read_particle_vtk(initial)
        final_points, final_fields, _ = _read_particle_vtk(final)

        np.testing.assert_allclose(initial_points[0, :2], [-3.999, -0.001])
        np.testing.assert_allclose(final_points[0, :2], [-4.0615, 0.0615])
        np.testing.assert_array_equal(initial_fields["status"], [0])
        np.testing.assert_array_equal(final_fields["status"], [3])
        np.testing.assert_allclose(initial_fields["owner_error"], [0.0])
        np.testing.assert_allclose(final_fields["owner_error"], [1.0])
        np.testing.assert_allclose(initial_fields["owner_level"], [1.0])
        np.testing.assert_allclose(final_fields["owner_level"], [1.0])
        np.testing.assert_allclose(
            initial_fields["owner_rank"], final_fields["owner_rank"]
        )
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
