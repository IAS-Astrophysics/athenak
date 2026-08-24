"""GPU regressions for configurable particle track output."""

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

import test_suite.testutils as testutils


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "vis" / "python"))
from read_particle_track import BLOCK_END, read_particle_track  # noqa: E402


def _track_input(tmp_path, source, name, selections, dcycle=1):
    """Add particle track outputs to an existing deterministic particle input."""
    output_blocks = []
    for number, (output_id, tags) in enumerate(selections, start=20):
        output_blocks.append(
            f"""
<output{number}>
file_type = particle_track
id = {output_id}
tags = {tags}
dcycle = {dcycle}
"""
        )
    path = tmp_path / f"{name}.athinput"
    path.write_text(Path(source).read_text() + "".join(output_blocks))
    return str(path)


def _sorted(data):
    """Sort all columns into particle-tag order."""
    order = np.argsort(data["ptag"])
    return {name: values[order] for name, values in data.items()}


def test_particle_track_gpu(tmp_path):
    """Select tags, preserve empty output, and evaluate shared and track-only callbacks."""
    shutil.rmtree("particle_track", ignore_errors=True)
    histories = [
        Path("particle_track_cosmic_ray.user.hst"),
        Path("particle_track_lagrangian_mc.user.hst"),
    ]
    for history in histories:
        history.unlink(missing_ok=True)

    try:
        input_file = _track_input(
            tmp_path,
            "inputs/particle_drift.athinput",
            "cosmic_ray_track",
            [
                ("list", "1,2,3,19"),
                ("slice", "::2"),
                ("all", "all"),
                ("missing", "19"),
            ],
        )
        log_path = Path(testutils.LOG_FILE_PATH)
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        assert testutils.run(
            input_file,
            ["job/basename=particle_track_cosmic_ray", "time/nlim=0"],
        ), "cosmic-ray particle track run failed"
        with log_path.open() as log:
            log.seek(log_offset)
            new_log = log.read()
        assert "requested tags not present at startup: 19" in new_log

        expected_columns = [
            "time", "cycle", "ptag", "status", "x", "y", "z", "vx", "vy", "vz",
            "radius_squared", "track_x",
        ]
        expected_tags = {
            "list": [1, 2, 3],
            "slice": [0, 2, 4, 6],
            "all": range(8),
            "missing": [],
        }
        for output_id, tags in expected_tags.items():
            path = Path(
                f"particle_track/particle_track_cosmic_ray.{output_id}.part_track"
            )
            data = _sorted(read_particle_track(path))
            tags = np.repeat(np.asarray(list(tags)), 2)
            assert list(data) == expected_columns
            np.testing.assert_array_equal(data["ptag"], tags)
            np.testing.assert_array_equal(data["status"], np.zeros(len(tags), dtype=int))
            np.testing.assert_array_equal(data["cycle"], np.zeros(len(tags), dtype=int))
            np.testing.assert_allclose(data["time"], 0.0)
            np.testing.assert_allclose(data["x"], -0.35 + 0.08*data["ptag"])
            np.testing.assert_allclose(data["y"], -0.20 + 0.04*data["ptag"], atol=1.0e-15)
            np.testing.assert_allclose(data["z"], -0.10 + 0.02*data["ptag"], atol=1.0e-15)
            np.testing.assert_allclose(data["vx"], 0.10 + 0.01*data["ptag"])
            np.testing.assert_allclose(data["vy"], -0.08 + 0.005*data["ptag"])
            np.testing.assert_allclose(data["vz"], 0.03 - 0.002*data["ptag"])
            np.testing.assert_allclose(
                data["radius_squared"], data["x"]**2 + data["y"]**2 + data["z"]**2
            )
            np.testing.assert_allclose(data["track_x"], data["x"])

        input_file = _track_input(
            tmp_path,
            "inputs/particle_lagrangian_mc.athinput",
            "lagrangian_mc_track",
            [("all", "all")],
        )
        assert testutils.run(
            input_file,
            ["job/basename=particle_track_lagrangian_mc", "time/nlim=0"],
        ), "Lagrangian-MC particle track run failed"
        data = _sorted(
            read_particle_track(
                "particle_track/particle_track_lagrangian_mc.all.part_track"
            )
        )
        assert list(data) == ["time", "cycle", "ptag", "status", "x", "y", "z"]
        np.testing.assert_array_equal(data["ptag"], np.repeat(np.arange(8), 2))
        np.testing.assert_array_equal(data["status"], np.zeros(16, dtype=int))
    finally:
        shutil.rmtree("particle_track", ignore_errors=True)
        for history in histories:
            history.unlink(missing_ok=True)


def test_particle_track_recovery_gpu(tmp_path):
    """Preserve damaged bytes and recover complete blocks before and after them."""
    basename = "particle_track_recovery"
    history = Path(f"{basename}.user.hst")
    track_path = Path(f"particle_track/{basename}.all.part_track")
    shutil.rmtree("particle_track", ignore_errors=True)
    history.unlink(missing_ok=True)

    try:
        input_file = _track_input(
            tmp_path,
            "inputs/particle_drift.athinput",
            "particle_track_recovery",
            [("all", "all")],
        )
        assert testutils.run(
            input_file,
            [f"job/basename={basename}", "time/nlim=0"],
        ), "initial particle track recovery run failed"

        complete = track_path.read_bytes()
        assert complete.endswith(BLOCK_END)
        initial = _sorted(read_particle_track(track_path))
        np.testing.assert_array_equal(initial["ptag"], np.repeat(np.arange(8), 2))

        missing_footer = tmp_path / "particle_track_missing_footer.part_track"
        missing_footer.write_bytes(complete[:-len(BLOCK_END)])
        with pytest.warns(RuntimeWarning, match="incomplete particle track data"):
            data = _sorted(read_particle_track(missing_footer))
        np.testing.assert_array_equal(data["ptag"], np.arange(8))

        damaged = complete[:-(len(BLOCK_END)+1)]
        damaged_tail = tmp_path / "particle_track_damaged_tail.part_track"
        damaged_tail.write_bytes(damaged)
        with pytest.warns(RuntimeWarning, match="incomplete particle track data"):
            data = _sorted(read_particle_track(damaged_tail))
        np.testing.assert_array_equal(data["ptag"], np.arange(8))

        track_path.write_bytes(damaged)
        log_path = Path(testutils.LOG_FILE_PATH)
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        assert testutils.run(
            input_file,
            [f"job/basename={basename}", "time/nlim=0"],
        ), "particle track recovery continuation failed"
        with log_path.open() as log:
            log.seek(log_offset)
            new_log = log.read()
        assert "ends with an incomplete output block" in new_log

        recovered = track_path.read_bytes()
        assert recovered.startswith(damaged)
        assert recovered.endswith(BLOCK_END)
        with pytest.warns(RuntimeWarning, match="skipping malformed particle track data"):
            data = _sorted(read_particle_track(track_path))
        np.testing.assert_array_equal(data["ptag"], np.repeat(np.arange(8), 3))
    finally:
        shutil.rmtree("particle_track", ignore_errors=True)
        history.unlink(missing_ok=True)
