"""GPU regressions for deferred particle deletion after a complete snapshot."""

from pathlib import Path
import shutil
import sys

import numpy as np

import test_suite.testutils as testutils
from test_suite.particles.test_particles_snapshot_gpu import _read_particle_vtk


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "vis" / "python"))
from read_particle_track import read_particle_track  # noqa: E402


PACTIVE = 0
PDELETE_PENDING = 2
PDELETE_AFTER_SNAPSHOT = 3


def _lifecycle_input(
    tmp_path,
    source,
    name,
    *,
    pvtk_dcycle=1,
    track_dcycle=None,
    restart_dcycle=None,
):
    """Add the requested lifecycle outputs to a deterministic particle input."""
    output_blocks = []
    if track_dcycle is not None:
        output_blocks.append(
            f"""

<output20>
file_type = particle_track
id = all
tags = all
dcycle = {track_dcycle}
"""
        )
    if pvtk_dcycle is not None:
        output_blocks.append(
            f"""

<output21>
file_type = pvtk
variable = prtcl_all
dcycle = {pvtk_dcycle}
"""
        )
    if restart_dcycle is not None:
        output_blocks.append(
            f"""

<output30>
file_type = rst
dcycle = {restart_dcycle}
"""
        )
    path = tmp_path / f"{name}.athinput"
    path.write_text(Path(source).read_text() + "".join(output_blocks))
    return str(path)


def _sorted_snapshot(path):
    """Read a particle snapshot and sort its records by durable tag."""
    points, fields, types = _read_particle_vtk(path)
    order = np.argsort(fields["ptag"])
    return points[order], {name: values[order] for name, values in fields.items()}, types


def _assert_drift_fields(points, fields):
    """Check stored and derived fields that must survive lifecycle compaction."""
    tags = fields["ptag"]
    np.testing.assert_allclose(fields["vx"], 0.10 + 0.01*tags)
    np.testing.assert_allclose(fields["vy"], -0.08 + 0.005*tags)
    np.testing.assert_allclose(fields["vz"], 0.03 - 0.002*tags)
    np.testing.assert_allclose(fields["radius_squared"], np.sum(points**2, axis=1))
    np.testing.assert_allclose(fields["vtk_y"], points[:, 1])


def test_particle_lifecycle_gpu(tmp_path):
    """Snapshot a deferred particle once, then remove it at the next purge."""
    basename = "particle_lifecycle_gpu"
    history = Path(f"{basename}.user.hst")
    shutil.rmtree("pvtk", ignore_errors=True)
    history.unlink(missing_ok=True)
    try:
        input_file = _lifecycle_input(
            tmp_path, "inputs/particle_drift.athinput", "particle_lifecycle"
        )
        assert testutils.run(
            input_file,
            [
                f"job/basename={basename}",
                "problem/delete_after_snapshot_tag=3",
                "time/nlim=2",
                "time/tlim=1.0",
                "output1/dcycle=0",
            ],
        ), "deferred particle lifecycle run failed"

        initial_points, initial_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00000.part.vtk")
        )
        deferred_points, deferred_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        )
        final_points, final_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00002.part.vtk")
        )

        np.testing.assert_array_equal(initial_fields["ptag"], np.arange(8))
        np.testing.assert_array_equal(
            initial_fields["status"], np.full(8, PACTIVE, dtype=np.int32)
        )
        np.testing.assert_array_equal(deferred_fields["ptag"], np.arange(8))
        expected_status = np.full(8, PACTIVE, dtype=np.int32)
        expected_status[3] = PDELETE_AFTER_SNAPSHOT
        np.testing.assert_array_equal(deferred_fields["status"], expected_status)
        assert not np.array_equal(deferred_points[3], initial_points[3])

        np.testing.assert_array_equal(final_fields["ptag"], [0, 1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(
            final_fields["status"], np.full(7, PACTIVE, dtype=np.int32)
        )
        _assert_drift_fields(deferred_points, deferred_fields)
        _assert_drift_fields(final_points, final_fields)
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        history.unlink(missing_ok=True)


def test_particle_lifecycle_output_cadence_gpu(tmp_path):
    """Particle-track output must not acknowledge a deferred particle snapshot."""
    basename = "particle_lifecycle_cadence_gpu"
    history = Path(f"{basename}.user.hst")
    track_path = Path(f"particle_track/{basename}.all.part_track")
    shutil.rmtree("particle_track", ignore_errors=True)
    shutil.rmtree("pvtk", ignore_errors=True)
    history.unlink(missing_ok=True)
    try:
        input_file = _lifecycle_input(
            tmp_path,
            "inputs/particle_drift.athinput",
            "particle_lifecycle_cadence",
            pvtk_dcycle=2,
            track_dcycle=1,
        )
        assert testutils.run(
            input_file,
            [
                f"job/basename={basename}",
                "problem/delete_after_snapshot_tag=3",
                "time/nlim=3",
                "time/tlim=1.0",
                "output1/dcycle=0",
            ],
        ), "particle lifecycle output-cadence run failed"

        track = read_particle_track(track_path)
        cycle1 = track["cycle"] == 1
        cycle2 = track["cycle"] == 2
        cycle3 = track["cycle"] == 3
        np.testing.assert_array_equal(np.sort(track["ptag"][cycle1]), np.arange(8))
        tag3_cycle1 = track["status"][cycle1][track["ptag"][cycle1] == 3]
        assert tag3_cycle1.size == 1
        assert tag3_cycle1.item() == PDELETE_AFTER_SNAPSHOT
        np.testing.assert_array_equal(np.sort(track["ptag"][cycle2]), np.arange(8))
        tag3_cycle2 = track["status"][cycle2][track["ptag"][cycle2] == 3]
        assert tag3_cycle2.size == 1
        assert tag3_cycle2.item() in (PDELETE_AFTER_SNAPSHOT, PDELETE_PENDING)
        assert 3 not in track["ptag"][cycle3]

        _, snapshot_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{basename}.prtcl_all.00001.part.vtk")
        )
        np.testing.assert_array_equal(snapshot_fields["ptag"], np.arange(8))
        assert snapshot_fields["status"][3] == PDELETE_AFTER_SNAPSHOT
    finally:
        shutil.rmtree("particle_track", ignore_errors=True)
        shutil.rmtree("pvtk", ignore_errors=True)
        history.unlink(missing_ok=True)


def test_particle_lifecycle_restart_gpu(tmp_path):
    """Restart a snapshot-acknowledged particle and purge it without reclassification."""
    split = "particle_lifecycle_restart_split_gpu"
    resumed = "particle_lifecycle_restart_resumed_gpu"
    histories = [Path(f"{name}.user.hst") for name in (split, resumed)]
    shutil.rmtree("pvtk", ignore_errors=True)
    shutil.rmtree("rst", ignore_errors=True)
    for history in histories:
        history.unlink(missing_ok=True)
    try:
        input_file = _lifecycle_input(
            tmp_path,
            "inputs/particle_drift.athinput",
            "particle_lifecycle_restart",
            restart_dcycle=1,
        )
        assert testutils.run(
            input_file,
            [
                f"job/basename={split}",
                "problem/delete_after_snapshot_tag=3",
                "time/nlim=1",
                "time/tlim=1.0",
                "output1/dcycle=0",
            ],
        ), "deferred particle checkpoint run failed"

        _, split_fields, _ = _sorted_snapshot(
            Path(f"pvtk/{split}.prtcl_all.00001.part.vtk")
        )
        assert split_fields["status"][3] == PDELETE_AFTER_SNAPSHOT
        fluid_restart = Path("rst") / f"{split}.00001.rst"
        particle_restart = Path("rst") / f"{split}.00001.part_rst"
        assert fluid_restart.exists(), "fluid restart was not written"
        assert particle_restart.exists(), "particle restart sidecar was not written"

        assert testutils.run_command(
            [
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(particle_restart),
                f"job/basename={resumed}",
                "time/nlim=2",
                "time/tlim=1.0",
                "output1/dcycle=0",
                "output21/dcycle=1",
                "output30/dcycle=0",
            ]
        ), "deferred particle restart run failed"

        resumed_snapshots = sorted(
            Path("pvtk").glob(f"{resumed}.prtcl_all.*.part.vtk")
        )
        assert resumed_snapshots, "restarted particle snapshot was not written"
        for snapshot in resumed_snapshots:
            _, fields, _ = _sorted_snapshot(snapshot)
            np.testing.assert_array_equal(fields["ptag"], [0, 1, 2, 4, 5, 6, 7])
            np.testing.assert_array_equal(
                fields["status"], np.full(7, PACTIVE, dtype=np.int32)
            )
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
        shutil.rmtree("rst", ignore_errors=True)
        for history in histories:
            history.unlink(missing_ok=True)
