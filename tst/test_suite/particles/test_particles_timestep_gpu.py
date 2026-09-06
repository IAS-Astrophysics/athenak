"""GPU regression test for a user-enrolled particle timestep limit."""

from pathlib import Path

import athena_read
import pytest

import test_suite.testutils as testutils


HISTORY = Path("particle_timestep_gpu.user.hst")


def test_particle_timestep_gpu():
    """Apply the user limit when it is tighter than the built-in drift limit."""
    try:
        HISTORY.unlink(missing_ok=True)
        assert testutils.run(
            "inputs/particle_drift.athinput",
            ["job/basename=particle_timestep_gpu", "time/tlim=1.0"],
        ), "particle timestep run failed"

        if not HISTORY.exists():
            pytest.fail(f"particle timestep history was not written: {HISTORY}")
        data = athena_read.hst(str(HISTORY))

        if data["time"][-1] != pytest.approx(0.125):
            pytest.fail(f"particle timestep was not applied: time={data['time'][-1]:g}")
    finally:
        HISTORY.unlink(missing_ok=True)
