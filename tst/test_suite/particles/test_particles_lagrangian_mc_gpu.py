"""Deterministic GPU regression test for Lagrangian Monte Carlo particles."""

from pathlib import Path

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_lagrangian_mc_gpu.user.hst")


def test_particle_lagrangian_mc_gpu():
    """Use RK2 mass fluxes to move a known set of particles across cell faces."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.run(
            "inputs/particle_lagrangian_mc.athinput"
        ), "Lagrangian MC particle run failed"

        if not HISTORY.exists():
            pytest.fail(f"Lagrangian MC particle history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        error_fields = (
            "fluid_err",
            "flux_err",
            "pos_err",
            "owner_err",
            "status_err",
        )
        for field in error_fields:
            if not np.all(np.isfinite(data[field])):
                pytest.fail(f"{field} contains non-finite values")
            if data[field][-1] > 1.0e-13:
                pytest.fail(f"{field} is too large: {data[field][-1]:g}")

        expected = {
            "moved": 4.0,
            "moved_tags": 9.0,
            "migrated": 1.0,
            "npart": 8.0,
            "tag_sum": 28.0,
        }
        for field, value in expected.items():
            if data[field][-1] != pytest.approx(value):
                pytest.fail(f"unexpected {field}: {data[field][-1]:g}")
    finally:
        HISTORY.unlink(missing_ok=True)
