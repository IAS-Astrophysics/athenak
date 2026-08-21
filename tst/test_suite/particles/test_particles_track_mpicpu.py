"""Two-rank CPU/MPI regression for particle track output."""

from pathlib import Path
import shutil
import sys

import numpy as np

import test_suite.testutils as testutils
from test_suite.particles.test_particles_track_gpu import _track_input


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "vis" / "python"))
from read_particle_track import read_particle_track  # noqa: E402


def test_particle_track_mpicpu(tmp_path):
    """Append one valid shared history from ranks with unequal particle counts."""
    basename = "particle_track_mpicpu"
    history = Path(f"{basename}.user.hst")
    shutil.rmtree("particle_track", ignore_errors=True)
    history.unlink(missing_ok=True)

    try:
        input_file = _track_input(
            tmp_path,
            "inputs/particle_migration.athinput",
            "particle_track_mpi",
            [("all", "all")],
            dcycle=2,
        )
        assert testutils.mpi_run(
            input_file,
            [f"job/basename={basename}", "problem/delete_tag=3"],
            threads=2,
        ), "two-rank particle track run failed"

        data = read_particle_track(
            f"particle_track/{basename}.all.part_track"
        )
        final_cycle = data["cycle"].max()
        final = {
            name: values[data["cycle"] == final_cycle] for name, values in data.items()
        }
        order = np.argsort(final["ptag"])
        tags = final["ptag"][order]
        np.testing.assert_array_equal(tags, [0, 1, 2, 4, 5, 6, 7])
        np.testing.assert_array_equal(final["status"][order], np.zeros(7, dtype=int))

        vx = np.full(7, 0.01)
        vx[tags == 4] = -0.20
        np.testing.assert_allclose(final["vx"][order], vx)
        np.testing.assert_allclose(final["vy"][order], -0.08 + 0.005*tags)
        np.testing.assert_allclose(final["vz"][order], 0.03 - 0.002*tags)
    finally:
        shutil.rmtree("particle_track", ignore_errors=True)
        history.unlink(missing_ok=True)
