"""Two-rank CPU/MPI regressions for Lagrangian Monte Carlo particles."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


@pytest.mark.parametrize(
    ("fluid", "input_name"),
    (
        ("hydro", "particle_lagrangian_mc"),
        ("mhd", "particle_lagrangian_mc_mhd"),
    ),
)
def test_particle_lagrangian_mc_migration_mpicpu(fluid, input_name):
    """Move one Lagrangian MC particle across a two-rank MeshBlock boundary."""
    basename = f"particle_lagrangian_mc_{fluid}_mpicpu"
    history = Path(f"{basename}.user.hst")
    try:
        history.unlink(missing_ok=True)
        assert testutils.mpi_run(
            f"inputs/{input_name}.athinput",
            [f"job/basename={basename}"],
            threads=2,
        ), f"two-rank Lagrangian MC {fluid.upper()} run failed"

        if not history.exists():
            pytest.fail(f"Lagrangian MC history was not written: {history}")
        data = athena_read.hst(str(history))

        for field in (
            "fluid_err",
            "flux_err",
            "pos_err",
            "owner_err",
            "status_err",
        ):
            if not np.all(np.isfinite(data[field])):
                pytest.fail(f"{fluid} {field} contains non-finite values")
            if data[field][-1] > 1.0e-13:
                pytest.fail(f"{fluid} {field} is too large: {data[field][-1]:g}")

        expected = {
            "moved": 4.0,
            "moved_tags": 9.0,
            "migrated": 1.0,
            "npart": 8.0,
            "tag_sum": 28.0,
        }
        for field, value in expected.items():
            if data[field][-1] != pytest.approx(value):
                pytest.fail(f"unexpected {fluid} {field}: {data[field][-1]:g}")
    finally:
        history.unlink(missing_ok=True)
