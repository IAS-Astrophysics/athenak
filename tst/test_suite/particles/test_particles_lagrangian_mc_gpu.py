"""Deterministic GPU regression test for Lagrangian Monte Carlo particles."""

from pathlib import Path
import shutil
import subprocess

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_lagrangian_mc_gpu.user.hst")


def _read_mesh_vtk_scalar(path):
    """Read the sole scalar field from an AthenaK mesh VTK output."""
    with path.open("rb") as file:
        while True:
            line = file.readline()
            if not line:
                pytest.fail(f"CELL_DATA header not found in {path}")
            if line.startswith(b"CELL_DATA "):
                ncells = int(line.split()[1])
                break

        scalar_header = file.readline().split()
        while not scalar_header:
            scalar_header = file.readline().split()
        assert scalar_header == [b"SCALARS", b"pdens", b"float"]
        assert file.readline().strip() == b"LOOKUP_TABLE default"
        return np.frombuffer(file.read(4*ncells), dtype=">f4").copy()


def test_particle_lagrangian_mc_gpu():
    """Use RK2 mass fluxes to move a known set of particles across cell faces."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.run(
            "inputs/particle_lagrangian_mc.athinput",
            ["particles/check_flux_probabilities=true"],
        ), "Lagrangian MC particle run failed"

        if not HISTORY.exists():
            pytest.fail(f"Lagrangian MC particle history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        error_fields = (
            "fluid_err",
            "flux_err",
            "pos_err",
            "owner_err",
            "status_err",
        )
        for field in error_fields:
            if not np.all(np.isfinite(data[field])):
                pytest.fail(f"{field} contains non-finite values")
            if data[field][-1] > 1.0e-13:
                pytest.fail(f"{field} is too large: {data[field][-1]:g}")

        expected = {
            "moved": 4.0,
            "moved_tags": 9.0,
            "migrated": 1.0,
            "npart": 8.0,
            "tag_sum": 28.0,
        }
        for field, value in expected.items():
            if data[field][-1] != pytest.approx(value):
                pytest.fail(f"unexpected {field}: {data[field][-1]:g}")
    finally:
        HISTORY.unlink(missing_ok=True)


def test_particle_lagrangian_mc_mhd_parity_gpu():
    """Match the deterministic Hydro particle path with zero-field MHD."""
    histories = []
    paths = []
    try:
        for fluid in ("hydro", "mhd"):
            basename = f"particle_lagrangian_mc_{fluid}_parity_gpu"
            history = Path(f"{basename}.user.hst")
            histories.append(history)
            history.unlink(missing_ok=True)
            input_name = (
                "particle_lagrangian_mc_mhd"
                if fluid == "mhd"
                else "particle_lagrangian_mc"
            )
            assert testutils.run(
                f"inputs/{input_name}.athinput", [f"job/basename={basename}"]
            ), f"Lagrangian MC {fluid.upper()} parity run failed"
            paths.append(athena_read.hst(str(history)))

        hydro, mhd = paths
        for field in (
            "time",
            "fluid_err",
            "flux_err",
            "pos_err",
            "owner_err",
            "moved",
            "moved_tags",
            "migrated",
            "npart",
            "tag_sum",
            "status_err",
        ):
            np.testing.assert_allclose(hydro[field], mhd[field], rtol=0.0, atol=1.0e-13)
    finally:
        for history in histories:
            history.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "dimension, mesh_overrides, cfl_number, should_succeed",
    (
        (2, (), "0.5", True),
        (2, (), "0.5001", False),
        (
            3,
            ("mesh/nx3=4", "meshblock/nx3=4", "particles/ppc=0.0625"),
            "0.3333333333333333",
            True,
        ),
        (3, ("mesh/nx3=4", "meshblock/nx3=4"), "0.34", False),
    ),
)
def test_particle_lagrangian_mc_cfl_gate(
    dimension, mesh_overrides, cfl_number, should_succeed
):
    """Accept the dimensional CFL limit and reject values above it at startup."""
    basename = f"particle_lagrangian_mc_cfl_{dimension}d_{cfl_number}"
    history = Path(f"{basename}.user.hst")
    try:
        result = subprocess.run(
            [
                "./athena",
                "-i",
                "inputs/particle_lagrangian_mc.athinput",
                f"job/basename={basename}",
                "time/nlim=0",
                f"time/cfl_number={cfl_number}",
                *mesh_overrides,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        if should_succeed:
            assert result.returncode == 0, output
        else:
            assert result.returncode != 0
            assert (
                "Lagrangian MC particles require time/cfl_number "
                f"<= 1/{dimension} in {dimension}D"
            ) in output
            assert "particles/check_flux_probabilities=true" in output
    finally:
        history.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "direction_sign, expected_counts",
    (
        (
            1,
            {
                "none": 1.0,
                "x1_left": 0.0,
                "x1_right": 5.0,
                "x2_left": 0.0,
                "x2_right": 9.0,
                "x3_left": 0.0,
                "x3_right": 1.0,
            },
        ),
        (
            -1,
            {
                "none": 1.0,
                "x1_left": 5.0,
                "x1_right": 0.0,
                "x2_left": 9.0,
                "x2_right": 0.0,
                "x3_left": 1.0,
                "x3_right": 0.0,
            },
        ),
    ),
)
def test_particle_lagrangian_mc_directions_gpu(direction_sign, expected_counts):
    """Exercise stay and all six one-cell moves using 3D RK2 Hydro fluxes."""
    basename = f"particle_lagrangian_mc_directions_{direction_sign}_gpu"
    history = Path(f"{basename}.user.hst")
    try:
        history.unlink(missing_ok=True)
        assert testutils.run(
            "inputs/particle_lagrangian_mc.athinput",
            [
                f"job/basename={basename}",
                "mesh/nx3=4",
                "meshblock/nx3=4",
                "time/cfl_number=0.3",
                "time/tlim=1.0",
                "particles/ppc=0.125",
                "problem/test_case=directions",
                f"problem/direction_sign={direction_sign}",
            ],
        ), "Lagrangian MC direction run failed"

        data = athena_read.hst(str(history))
        for field in ("pos_err", "owner_err", "status_err", "min_err"):
            assert np.all(np.isfinite(data[field]))
            assert data[field][-1] <= 1.0e-13
        for field, expected in expected_counts.items():
            assert data[field][-1] == pytest.approx(expected)
        assert data["npart"][-1] == pytest.approx(16.0)
        assert data["tag_sum"][-1] == pytest.approx(120.0)
    finally:
        history.unlink(missing_ok=True)


def test_particle_lagrangian_mc_runtime_probability_check_gpu():
    """Reject an invalid integrated flux when the optional check is enabled."""
    basename = "particle_lagrangian_mc_runtime_probability_check_gpu"
    history = Path(f"{basename}.user.hst")
    try:
        result = subprocess.run(
            [
                "./athena",
                "-i",
                "inputs/particle_lagrangian_mc.athinput",
                f"job/basename={basename}",
                "particles/check_flux_probabilities=true",
                "problem/test_case=guard",
                "problem/user_srcs=true",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "Invalid Lagrangian MC outgoing probability" in output
        assert "density=" in output
        assert "outward fluxes=" in output
        assert "time/cfl_number <=" not in output
    finally:
        history.unlink(missing_ok=True)


def test_particle_lagrangian_mc_reproducibility_gpu():
    """Preserve each tag's multi-cycle path across array order and block migration."""
    histories = []
    paths = []
    try:
        for reverse in (False, True):
            basename = f"particle_lagrangian_mc_repro_{int(reverse)}_gpu"
            history = Path(f"{basename}.user.hst")
            histories.append(history)
            history.unlink(missing_ok=True)
            assert testutils.run(
                "inputs/particle_lagrangian_mc.athinput",
                [
                    f"job/basename={basename}",
                    "time/nlim=3",
                    "time/tlim=1.0",
                    "particles/ppc=0.5",
                    "problem/test_case=reproducibility",
                    f"problem/reverse_particle_order={str(reverse).lower()}",
                ],
            ), "Lagrangian MC reproducibility run failed"
            paths.append(athena_read.hst(str(history)))

        ordered, reversed_order = paths
        np.testing.assert_array_equal(ordered["time"], reversed_order["time"])
        for tag in range(16):
            field = f"x{tag}"
            np.testing.assert_array_equal(ordered[field], reversed_order[field])
        positions = np.stack([ordered[f"x{tag}"] for tag in range(16)])
        moved = np.diff(positions, axis=1) > 0.0
        assert np.any(np.any(moved, axis=1) & np.any(~moved, axis=1))
        for data in paths:
            assert len(data["time"]) == 4
            assert np.all(data["owner_err"] == 0.0)
            assert data["migrated"][-1] > 0.0
            assert data["npart"][-1] == pytest.approx(16.0)
            assert data["tag_sum"][-1] == pytest.approx(120.0)
    finally:
        for history in histories:
            history.unlink(missing_ok=True)


def test_particle_density_output_collocated_gpu(tmp_path):
    """Count collocated particles without losing device updates to a write race."""
    basename = "particle_density_collocated_gpu"
    history = Path(f"{basename}.user.hst")
    vtk_path = Path(f"vtk/{basename}.prtcl_d.00000.vtk")
    input_path = tmp_path / "particle_density_collocated.athinput"
    input_path.write_text(
        Path("inputs/particle_lagrangian_mc.athinput").read_text()
        + """

<output2>
file_type = vtk
variable = prtcl_d
dcycle = 1
"""
    )

    try:
        shutil.rmtree("vtk", ignore_errors=True)
        history.unlink(missing_ok=True)
        assert testutils.run(
            str(input_path),
            [
                f"job/basename={basename}",
                "time/nlim=0",
                "particles/ppc=0.5",
                "problem/test_case=reproducibility",
            ],
        ), "collocated particle-density output run failed"

        assert vtk_path.exists(), "collocated particle-density VTK was not written"
        density = _read_mesh_vtk_scalar(vtk_path)
        assert np.count_nonzero(density) == 1
        assert np.max(density) == pytest.approx(16.0)
        assert np.sum(density) == pytest.approx(16.0)
    finally:
        shutil.rmtree("vtk", ignore_errors=True)
        history.unlink(missing_ok=True)
