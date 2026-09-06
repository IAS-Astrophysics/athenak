"""Deterministic two-rank CPU test for deletion during particle migration."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_delete_mpicpu.user.hst")


def test_particle_delete_mpicpu():
    """Delete one would-be migrant while another crosses in the opposite direction."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.mpi_run(
            "inputs/particle_migration.athinput",
            ["job/basename=particle_delete_mpicpu", "problem/delete_tag=3"],
            threads=2,
        ), "particle MPI deletion run failed"

        if not HISTORY.exists():
            pytest.fail(f"particle MPI deletion history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        if not np.all(np.isfinite(data["max_err"])):
            pytest.fail("particle MPI deletion error contains non-finite values")
        if data["max_err"][-1] > 1.0e-13:
            pytest.fail(
                f"surviving particle data changed: error={data['max_err'][-1]:g}"
            )
        if data["npart"][-1] != pytest.approx(7.0):
            pytest.fail(f"delete-pending particle remains: count={data['npart'][-1]:g}")
        if data["tag_sum"][-1] != pytest.approx(25.0):
            pytest.fail(f"wrong particle was removed: tag sum={data['tag_sum'][-1]:g}")
        if data["owner_err"][-1] != pytest.approx(0.0):
            pytest.fail(f"particle owners are incorrect: {data['owner_err'][-1]:g}")
        if data["migrated"][-1] != pytest.approx(1.0):
            pytest.fail(f"wrong migration count: {data['migrated'][-1]:g}")
        if data["status_err"][-1] != pytest.approx(0.0):
            pytest.fail(
                f"surviving particle statuses changed: {data['status_err'][-1]:g}"
            )
        if data["count_err"][-1] != pytest.approx(0.0):
            pytest.fail(f"particle counts are inconsistent: {data['count_err'][-1]:g}")
    finally:
        HISTORY.unlink(missing_ok=True)
