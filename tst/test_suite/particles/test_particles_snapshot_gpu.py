"""GPU regression test for complete particle snapshot output."""

from pathlib import Path
import shutil

import numpy as np

import test_suite.testutils as testutils


def _snapshot_input(tmp_path, source, name):
    """Add particle VTK output to an existing deterministic particle input."""
    path = tmp_path / f"{name}.athinput"
    path.write_text(
        Path(source).read_text()
        + """

<output2>
file_type = pvtk
variable = prtcl_all
dcycle = 1
"""
    )
    return str(path)


def _next_nonempty_line(file):
    """Read the next nonempty ASCII line from a mixed text/binary VTK file."""
    while True:
        line = file.readline()
        if not line:
            return ""
        line = line.decode("ascii").strip()
        if line:
            return line


def _read_particle_vtk(path):
    """Read points and scalar fields from an AthenaK particle VTK snapshot."""
    with path.open("rb") as file:
        assert file.readline().startswith(b"# vtk DataFile Version")
        file.readline()
        assert file.readline().strip() == b"BINARY"
        assert file.readline().strip() == b"DATASET UNSTRUCTURED_GRID"

        point_header = _next_nonempty_line(file).split()
        assert point_header[0] == "POINTS"
        npart = int(point_header[1])
        points = np.frombuffer(file.read(3*npart*4), dtype=">f4").reshape(npart, 3)

        point_data_header = _next_nonempty_line(file).split()
        assert point_data_header == ["POINT_DATA", str(npart)]

        fields = {}
        types = {}
        while True:
            scalar_header = _next_nonempty_line(file)
            if not scalar_header:
                break
            words = scalar_header.split()
            assert words[0] == "SCALARS"
            name, vtk_type = words[1], words[2]
            assert _next_nonempty_line(file) == "LOOKUP_TABLE default"
            dtype = ">i4" if vtk_type == "int" else ">f4"
            fields[name] = np.frombuffer(file.read(npart*4), dtype=dtype).copy()
            types[name] = vtk_type
    return points, fields, types


def test_particle_snapshot_gpu(tmp_path):
    """Write stored, shared-callback, and VTK-only fields with their local names."""
    shutil.rmtree("pvtk", ignore_errors=True)
    histories = [
        Path("particle_snapshot_cosmic_ray.user.hst"),
        Path("particle_snapshot_lagrangian_mc.user.hst"),
    ]
    for history in histories:
        history.unlink(missing_ok=True)

    try:
        cosmic_input = _snapshot_input(
            tmp_path, "inputs/particle_drift.athinput", "cosmic_ray_snapshot"
        )
        assert testutils.run(
            cosmic_input,
            ["job/basename=particle_snapshot_cosmic_ray", "time/nlim=0"],
        ), "cosmic-ray particle snapshot run failed"

        cosmic_path = Path("pvtk/particle_snapshot_cosmic_ray.prtcl_all.00000.part.vtk")
        assert cosmic_path.exists(), "cosmic-ray particle snapshot was not written"
        points, fields, types = _read_particle_vtk(cosmic_path)
        assert points.shape == (8, 3)
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
        np.testing.assert_array_equal(np.sort(fields["ptag"]), np.arange(8))
        np.testing.assert_array_equal(fields["status"], np.zeros(8, dtype=np.int32))
        np.testing.assert_allclose(fields["vx"], 0.10 + 0.01*fields["ptag"])
        np.testing.assert_allclose(fields["vy"], -0.08 + 0.005*fields["ptag"])
        np.testing.assert_allclose(fields["vz"], 0.03 - 0.002*fields["ptag"])
        np.testing.assert_allclose(fields["radius_squared"], np.sum(points**2, axis=1))
        np.testing.assert_allclose(fields["vtk_y"], points[:, 1])

        shutil.rmtree("pvtk")
        lmc_input = _snapshot_input(
            tmp_path,
            "inputs/particle_lagrangian_mc.athinput",
            "lagrangian_mc_snapshot",
        )
        assert testutils.run(
            lmc_input,
            ["job/basename=particle_snapshot_lagrangian_mc", "time/nlim=0"],
        ), "Lagrangian-MC particle snapshot run failed"

        lmc_path = Path(
            "pvtk/particle_snapshot_lagrangian_mc.prtcl_all.00000.part.vtk"
        )
        assert lmc_path.exists(), "Lagrangian-MC particle snapshot was not written"
        points, fields, types = _read_particle_vtk(lmc_path)
        assert points.shape == (8, 3)
        assert list(fields) == [
            "ptag", "status", "x_min", "y_min", "z_min", "t_min",
        ]
        assert types == {
            "ptag": "int",
            "status": "int",
            "x_min": "float",
            "y_min": "float",
            "z_min": "float",
            "t_min": "float",
        }
        np.testing.assert_array_equal(np.sort(fields["ptag"]), np.arange(8))
        np.testing.assert_array_equal(fields["status"], np.zeros(8, dtype=np.int32))
        np.testing.assert_allclose(fields["x_min"], points[:, 0])
        np.testing.assert_allclose(fields["y_min"], points[:, 1])
        np.testing.assert_allclose(fields["z_min"], 0.5)
        np.testing.assert_allclose(fields["t_min"], 0.0)
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        for history in histories:
            history.unlink(missing_ok=True)
