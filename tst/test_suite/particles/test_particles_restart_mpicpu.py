"""Changed-rank CPU MPI regression for particle sidecar restart."""

from pathlib import Path
import shutil

import athena_read
import numpy as np

import test_suite.testutils as testutils


def _restart_input(tmp_path):
    """Add a restart output to the deterministic migration input."""
    path = tmp_path / "particle_restart_mpi.athinput"
    path.write_text(
        Path("inputs/particle_migration.athinput").read_text()
        + """

<output2>
file_type = rst
dcycle = 0
"""
    )
    return str(path)


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


def test_particle_restart_changed_rank_count_mpicpu(tmp_path):
    """Read a one-rank sidecar on two ranks and preserve the drift solution."""
    direct = "particle_restart_mpi_direct"
    split = "particle_restart_mpi_split"
    resumed = "particle_restart_mpi_resumed"
    histories = [Path(f"{name}.user.hst") for name in (direct, split, resumed)]
    shutil.rmtree("rst", ignore_errors=True)
    for history in histories:
        history.unlink(missing_ok=True)
    try:
        input_file = _restart_input(tmp_path)
        assert testutils.run_command(
            [
                "mpirun",
                "-np",
                "2",
                "./athena",
                "-i",
                input_file,
                f"job/basename={direct}",
                "time/nlim=2",
                "time/tlim=1.0",
                "output2/dcycle=0",
            ]
        ), "uninterrupted two-rank particle run failed"
        direct_state = _final_history(direct)

        assert testutils.run_command(
            [
                "mpirun",
                "-np",
                "1",
                "./athena",
                "-i",
                input_file,
                f"job/basename={split}",
                "time/nlim=1",
                "time/tlim=1.0",
                "output2/dcycle=1",
            ]
        ), "one-rank particle checkpoint run failed"

        fluid_restart = Path("rst") / f"{split}.00001.rst"
        particle_restart = Path("rst") / f"{split}.00001.part_rst"
        assert testutils.run_command(
            [
                "mpirun",
                "-np",
                "2",
                "./athena",
                "-r",
                str(fluid_restart),
                "-p",
                str(particle_restart),
                f"job/basename={resumed}",
                "time/nlim=2",
                "time/tlim=1.0",
                "output2/dcycle=0",
            ]
        ), "two-rank particle restart run failed"

        np.testing.assert_array_equal(_final_history(resumed), direct_state)
    finally:
        shutil.rmtree("rst", ignore_errors=True)
        for history in histories:
            history.unlink(missing_ok=True)
