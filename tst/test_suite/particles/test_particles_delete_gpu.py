"""GPU regression test for pending particle deletion."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_delete_gpu.user.hst")


def test_particle_delete_gpu():
    """Remove a delete-pending particle without corrupting the survivors."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.run(
            "inputs/particle_drift.athinput",
            ["job/basename=particle_delete_gpu", "problem/delete_tag=3"],
        ), "particle deletion run failed"

        if not HISTORY.exists():
            pytest.fail(f"particle deletion history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        if not np.all(np.isfinite(data["max_err"])):
            pytest.fail("particle deletion error contains non-finite values")
        if data["max_err"][-1] > 1.0e-13:
            pytest.fail(f"surviving particle data changed: error={data['max_err'][-1]:g}")
        if data["npart"][-1] != pytest.approx(7.0):
            pytest.fail(f"delete-pending particle remains: count={data['npart'][-1]:g}")
        if data["tag_sum"][-1] != pytest.approx(25.0):
            pytest.fail(f"wrong particle was removed: tag sum={data['tag_sum'][-1]:g}")
        if data["owner_err"][-1] != pytest.approx(0.0):
            pytest.fail(f"particle owners are incorrect: {data['owner_err'][-1]:g}")
        if data["status_err"][-1] != pytest.approx(0.0):
            pytest.fail(
                f"surviving particle statuses changed: {data['status_err'][-1]:g}"
            )
    finally:
        HISTORY.unlink(missing_ok=True)
