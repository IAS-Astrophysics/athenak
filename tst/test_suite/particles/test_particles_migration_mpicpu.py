"""Deterministic two-rank CPU test for particle MPI migration."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_migration_mpicpu.user.hst")


def test_particle_migration_mpicpu():
    """Exchange particles across a MeshBlock boundary between two MPI ranks."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.mpi_run(
            "inputs/particle_migration.athinput",
            ["job/basename=particle_migration_mpicpu"],
            threads=2,
        ), "particle MPI migration run failed"

        if not HISTORY.exists():
            pytest.fail(f"particle MPI migration history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        if not np.all(np.isfinite(data["max_err"])):
            pytest.fail("particle MPI migration error contains non-finite values")
        if data["max_err"][-1] > 1.0e-13:
            pytest.fail(
                f"particle MPI migration error is too large: {data['max_err'][-1]:g}"
            )
        if data["npart"][-1] != pytest.approx(8.0):
            pytest.fail(f"particle count changed: {data['npart'][-1]:g}")
        if data["tag_sum"][-1] != pytest.approx(28.0):
            pytest.fail(f"particle identifiers changed: tag sum={data['tag_sum'][-1]:g}")
        if data["owner_err"][-1] != pytest.approx(0.0):
            pytest.fail(f"particle owners are incorrect: {data['owner_err'][-1]:g}")
        if data["migrated"][-1] != pytest.approx(2.0):
            pytest.fail(f"wrong migration count: {data['migrated'][-1]:g}")
    finally:
        HISTORY.unlink(missing_ok=True)
