"""GPU regression test for frozen particle pusher behavior."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_frozen_gpu.user.hst")


def test_particle_frozen_gpu():
    """Keep a frozen particle fixed while active particles drift on CUDA."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.run(
            "inputs/particle_drift.athinput",
            ["job/basename=particle_frozen_gpu", "problem/frozen_tag=3"],
        ), "frozen particle drift run failed"

        if not HISTORY.exists():
            pytest.fail(f"frozen particle history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        if not np.all(np.isfinite(data["max_err"])):
            pytest.fail("frozen particle drift error contains non-finite values")
        if data["max_err"][-1] > 1.0e-13:
            pytest.fail(f"frozen particle moved: error={data['max_err'][-1]:g}")
        if data["npart"][-1] != pytest.approx(8.0):
            pytest.fail(f"particle count changed: {data['npart'][-1]:g}")
        if data["tag_sum"][-1] != pytest.approx(28.0):
            pytest.fail(f"particle identifiers changed: tag sum={data['tag_sum'][-1]:g}")
        if data["status_err"][-1] != pytest.approx(0.0):
            pytest.fail(f"particle statuses changed: {data['status_err'][-1]:g}")
    finally:
        HISTORY.unlink(missing_ok=True)
