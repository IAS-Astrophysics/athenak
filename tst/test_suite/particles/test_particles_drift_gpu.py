"""Deterministic GPU characterization test for the existing particle drift pusher."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_drift_gpu.user.hst")


def test_particle_drift_gpu():
    """Drift known particles on CUDA without changing their count or identifiers."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.run("inputs/particle_drift.athinput"), "particle drift run failed"

        if not HISTORY.exists():
            pytest.fail(f"particle drift history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        if not np.all(np.isfinite(data["max_err"])):
            pytest.fail("particle drift error contains non-finite values")
        if data["max_err"][-1] > 1.0e-13:
            pytest.fail(f"particle drift error is too large: {data['max_err'][-1]:g}")
        if data["npart"][-1] != pytest.approx(8.0):
            pytest.fail(f"particle count changed: {data['npart'][-1]:g}")
        if data["tag_sum"][-1] != pytest.approx(28.0):
            pytest.fail(f"particle identifiers changed: tag sum={data['tag_sum'][-1]:g}")
    finally:
        HISTORY.unlink(missing_ok=True)
