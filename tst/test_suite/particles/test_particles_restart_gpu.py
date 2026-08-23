"""GPU contract regression for particle sidecar restart."""

from pathlib import Path
import shutil
import struct
import subprocess

import athena_read
import numpy as np

import test_suite.testutils as testutils
from test_suite.particles.test_particles_snapshot_gpu import _read_particle_vtk


def _restart_input(tmp_path):
    """Add a restart output to the deterministic drift input."""
    path = tmp_path / "particle_restart.athinput"
    path.write_text(
        Path("inputs/particle_drift.athinput").read_text()
        + """

<output2>
file_type = rst
dcycle = 0
"""
    )
    return str(path)


def _lagrangian_mc_restart_input(tmp_path):
    """Add a restart output to the deterministic Lagrangian-MC input."""
    path = tmp_path / "particle_lagrangian_mc_restart.athinput"
    input_text = Path("inputs/particle_lagrangian_mc.athinput").read_text()
    input_text = input_text.replace(
        "<particles>\n", "<particles>\nrestart_sort_by_tag = false\n", 1
    )
    path.write_text(
        input_text
        + """

<output2>
file_type = rst
dcycle = 0
"""
    )
    return str(path)


def _injection_restart_input(tmp_path):
    """Add a restart output to the particle injection input."""
    path = tmp_path / "particle_injection_restart.athinput"
    path.write_text(
        Path("inputs/particle_injection.athinput").read_text()
        + """

<output2>
file_type = rst
dcycle = 1
"""
    )
    return str(path)


def _sidecar_layout(path):
    """Return the payload layout of the current single-population sidecar."""
    contents = bytearray(path.read_bytes())
    assert contents[:8] == b"ATHPRST1"
    version, endian, real_size, int_size, npop = struct.unpack_from(
        "=5I", contents, 8
    )
    assert version == 1
    assert endian == 0x01020304
    assert npop == 1
    nmb = struct.unpack_from("=Q", contents, 28)[0]

    entry_offset = 44
    _, _, nrdata, nidata, metadata_size = struct.unpack_from(
        "=4IQ", contents, entry_offset + 64
    )
    directory_offset = entry_offset + 88
    counts = struct.unpack_from(f"={nmb}Q", contents, directory_offset)
    payload_offset = directory_offset + nmb * 8 + metadata_size
    return {
        "contents": contents,
        "counts": counts,
        "directory_offset": directory_offset,
        "payload_offset": payload_offset,
        "real_size": real_size,
        "int_size": int_size,
        "nrdata": nrdata,
        "nidata": nidata,
    }


def _write_position_sidecar(source, destination, x1):
    """Replace one stored particle's x1 coordinate."""
    layout = _sidecar_layout(source)
    offset = layout["payload_offset"]
    for count in layout["counts"]:
        if count:
            real_format = "=d" if layout["real_size"] == 8 else "=f"
            struct.pack_into(real_format, layout["contents"], offset, x1)
            destination.write_bytes(layout["contents"])
            return
        offset += count * (
            layout["nrdata"] * layout["real_size"]
            + (layout["nidata"] - 1) * layout["int_size"]
        )
    raise AssertionError("particle restart contains no particle to corrupt")


def _write_empty_sidecar(source, destination):
    """Replace the per-MeshBlock counts and payload with an empty distribution."""
    layout = _sidecar_layout(source)
    for gid in range(len(layout["counts"])):
        struct.pack_into(
            "=Q", layout["contents"], layout["directory_offset"] + gid * 8, 0
        )
    destination.write_bytes(layout["contents"][: layout["payload_offset"]])


def _final_history(basename):
    data = athena_read.hst(f"{basename}.user.hst")
    fields = (
        "time",
        "max_err",
        "npart",
        "tag_sum",
        "owner_err",
        "migrated",
        "status_err",
        "count_err",
    )
    return np.asarray([data[field][-1] for field in fields])


def _final_lagrangian_mc_history(basename):
    data = athena_read.hst(f"{basename}.user.hst")
    fields = (
        "time",
        *(f"x{tag}" for tag in range(16)),
        "owner_err",
        "migrated",
        "npart",
        "tag_sum",
    )
    return np.asarray([data[field][-1] for field in fields])


def test_particle_drift_restart_gpu(tmp_path):
    """A sidecar restart must reproduce an uninterrupted drift run exactly."""
    direct = "particle_restart_direct_gpu"
    split = "particle_restart_split_gpu"
    resumed = "particle_restart_resumed_gpu"
    histories = [Path(f"{name}.user.hst") for name in (direct, split, resumed)]
    shutil.rmtree("rst", ignore_errors=True)
    for history in histories:
        history.unlink(missing_ok=True)
    try:
        input_file = _restart_input(tmp_path)
        assert testutils.run(
            input_file,
            [
                f"job/basename={direct}",
                "time/nlim=2",
                "time/tlim=1.0",
                "output2/dcycle=0",
            ],
        ), "uninterrupted particle drift run failed"
        direct_state = _final_history(direct)

        assert testutils.run(
            input_file,
            [
                f"job/basename={split}",
                "time/nlim=1",
                "time/tlim=1.0",
                "output2/dcycle=1",
            ],
        ), "particle drift checkpoint run failed"

        fluid_restart = Path("rst") / f"{split}.00001.rst"
        particle_restart = Path("rst") / f"{split}.00001.part_rst"
        assert fluid_restart.exists(), "fluid restart was not written"
        assert particle_restart.exists(), "particle restart sidecar was not written"
        selected_particle_restart = tmp_path / "independent_particles.part_rst"
        shutil.move(particle_restart, selected_particle_restart)

        assert testutils.run_command(
            [
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(selected_particle_restart),
                f"job/basename={resumed}",
                "time/nlim=2",
                "time/tlim=1.0",
                "output2/dcycle=0",
            ]
        ), "particle drift restart run failed"

        np.testing.assert_array_equal(_final_history(resumed), direct_state)
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        for history in histories:
            history.unlink(missing_ok=True)


def test_particle_restart_custom_distribution_gpu(tmp_path):
    """Enforce GID ownership while accepting a structurally valid empty sidecar."""
    split = "particle_restart_custom_split_gpu"
    invalid = "particle_restart_invalid_position_gpu"
    upper_face = "particle_restart_upper_face_gpu"
    empty = "particle_restart_empty_gpu"
    histories = [
        Path(f"{name}.user.hst")
        for name in (split, invalid, upper_face, empty)
    ]
    shutil.rmtree("rst", ignore_errors=True)
    for history in histories:
        history.unlink(missing_ok=True)
    try:
        input_file = _restart_input(tmp_path)
        assert testutils.run(
            input_file,
            [
                f"job/basename={split}",
                "time/nlim=1",
                "time/tlim=1.0",
                "meshblock/nx1=4",
                "output2/dcycle=1",
            ],
        ), "particle drift checkpoint run failed"

        fluid_restart = Path("rst") / f"{split}.00001.rst"
        particle_restart = Path("rst") / f"{split}.00001.part_rst"
        invalid_restart = tmp_path / "invalid_position.part_rst"
        upper_face_restart = tmp_path / "upper_face.part_rst"
        empty_restart = tmp_path / "empty.part_rst"
        _write_position_sidecar(particle_restart, invalid_restart, 1.0e30)
        _write_position_sidecar(particle_restart, upper_face_restart, 0.0)
        _write_empty_sidecar(particle_restart, empty_restart)

        result = subprocess.run(
            [
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(invalid_restart),
                f"job/basename={invalid}",
                "time/nlim=1",
                "output2/dcycle=0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "position outside its MeshBlock section" in result.stdout + result.stderr

        result = subprocess.run(
            [
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(upper_face_restart),
                f"job/basename={upper_face}",
                "time/nlim=1",
                "output2/dcycle=0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "position outside its MeshBlock section" in result.stdout + result.stderr

        assert testutils.run_command(
            [
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(empty_restart),
                f"job/basename={empty}",
                "time/nlim=1",
                "output2/dcycle=0",
            ]
        ), "empty particle sidecar did not load"
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        for history in histories:
            history.unlink(missing_ok=True)


def test_particle_restart_sort_by_tag_gpu(tmp_path):
    """Opt-in tag sorting must make equivalent checkpoints byte-identical."""
    basenames = []
    sidecars = []
    shutil.rmtree("rst", ignore_errors=True)
    try:
        input_file = _lagrangian_mc_restart_input(tmp_path)
        for reverse in (False, True):
            basename = f"particle_restart_sorted_{int(reverse)}_gpu"
            basenames.append(basename)
            Path(f"{basename}.user.hst").unlink(missing_ok=True)
            assert testutils.run(
                input_file,
                [
                    f"job/basename={basename}",
                    "time/nlim=1",
                    "time/tlim=1.0",
                    "particles/ppc=0.5",
                    "particles/restart_sort_by_tag=true",
                    "problem/test_case=reproducibility",
                    f"problem/reverse_particle_order={str(reverse).lower()}",
                    "output2/dcycle=1",
                ],
            ), "sorted particle checkpoint run failed"
            sidecars.append(
                (Path("rst") / f"{basename}.00001.part_rst").read_bytes()
            )

        assert sidecars[0] == sidecars[1]
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        for basename in basenames:
            Path(f"{basename}.user.hst").unlink(missing_ok=True)


def test_particle_lagrangian_mc_restart_gpu(tmp_path):
    """A sidecar restart must preserve the tag-keyed random continuation."""
    direct = "particle_lmc_restart_direct_gpu"
    split = "particle_lmc_restart_split_gpu"
    resumed = "particle_lmc_restart_resumed_gpu"
    histories = [Path(f"{name}.user.hst") for name in (direct, split, resumed)]
    shutil.rmtree("rst", ignore_errors=True)
    for history in histories:
        history.unlink(missing_ok=True)
    try:
        input_file = _lagrangian_mc_restart_input(tmp_path)
        common = [
            "time/tlim=1.0",
            "particles/ppc=0.5",
            "problem/test_case=reproducibility",
        ]
        assert testutils.run(
            input_file,
            [f"job/basename={direct}", "time/nlim=3", *common, "output2/dcycle=0"],
        ), "uninterrupted Lagrangian-MC run failed"
        direct_state = _final_lagrangian_mc_history(direct)

        assert testutils.run(
            input_file,
            [f"job/basename={split}", "time/nlim=1", *common, "output2/dcycle=1"],
        ), "Lagrangian-MC checkpoint run failed"

        fluid_restart = Path("rst") / f"{split}.00001.rst"
        particle_restart = Path("rst") / f"{split}.00001.part_rst"
        assert testutils.run_command(
            [
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(particle_restart),
                f"job/basename={resumed}",
                "time/nlim=3",
                "time/tlim=1.0",
                "output2/dcycle=0",
            ]
        ), "Lagrangian-MC restart run failed"

        np.testing.assert_array_equal(
            _final_lagrangian_mc_history(resumed), direct_state
        )
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        for history in histories:
            history.unlink(missing_ok=True)


def test_particle_injection_tag_high_water_gpu(tmp_path):
    """Restarted injection must not reuse tags removed from a custom sidecar."""
    split = "particle_injection_restart_split_gpu"
    resumed = "particle_injection_restart_resumed_gpu"
    shutil.rmtree("pvtk", ignore_errors=True)
    shutil.rmtree("rst", ignore_errors=True)
    try:
        input_file = _injection_restart_input(tmp_path)
        assert testutils.run(
            input_file,
            [f"job/basename={split}"],
        ), "particle injection checkpoint run failed"

        fluid_restart = Path("rst") / f"{split}.00001.rst"
        particle_restart = Path("rst") / f"{split}.00001.part_rst"
        empty_restart = tmp_path / "empty_injection.part_rst"
        _write_empty_sidecar(particle_restart, empty_restart)

        assert testutils.run_command(
            [
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(empty_restart),
                f"job/basename={resumed}",
                "problem/injection_cycle=2",
                "time/nlim=2",
                "output2/dcycle=0",
            ]
        ), "particle injection restart run failed"

        snapshot = Path(f"pvtk/{resumed}.prtcl_all.00002.part.vtk")
        _, fields, _ = _read_particle_vtk(snapshot)
        np.testing.assert_array_equal(fields["ptag"], [2])
        np.testing.assert_array_equal(fields["status"], [0])
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        shutil.rmtree("rst", ignore_errors=True)
