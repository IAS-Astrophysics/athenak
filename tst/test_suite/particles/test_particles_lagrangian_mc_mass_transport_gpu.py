"""End-to-end scientific-equivalence target for Wong-Medeiros-Stone tracers."""

from pathlib import Path
import shutil
import sys

import athena_read
import numpy as np
import pytest

import test_suite.testutils as testutils


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "vis" / "python"))
from read_particle_track import read_particle_track  # noqa: E402


HISTORY = Path("particle_lagrangian_mc_mass_transport_gpu.user.hst")
TRACK = Path(
    "particle_track/particle_lagrangian_mc_mass_transport_gpu.all.part_track"
)


def _assert_monotone(values, name, atol=0.0):
    """Require a cumulative history quantity to be nondecreasing."""
    difference = np.diff(values)
    if np.any(difference < -atol):
        pytest.fail(f"{name} is not monotone: {values}")


def test_particle_lagrangian_mc_mass_transport_gpu():
    """Recover conserved mass transport through SMR and an absorbing horizon.

    This is an outside-in acceptance test for the particle path used by Wong,
    Medeiros, and Stone.  The problem generator must initialize equal-mass
    tracers from unequal zone coordinate masses, evolve them from integrated
    MHD mass fluxes across a fixed coarse/fine interface, and retire them at
    the horizon.
    Individual trajectories need not match the historical random sequence;
    the tracer and fluid accreted fractions must agree statistically.
    """
    shutil.rmtree("particle_track", ignore_errors=True)
    shutil.rmtree("pvtk", ignore_errors=True)
    HISTORY.unlink(missing_ok=True)

    try:
        assert testutils.run(
            "inputs/particle_lagrangian_mc_mass_transport.athinput"
        ), "Wong-Medeiros-Stone Lagrangian MC acceptance run failed"

        if not HISTORY.exists():
            pytest.fail(f"particle mass-transport history was not written: {HISTORY}")
        history = athena_read.hst(str(HISTORY))
        required_history = {
            "init_mass",
            "fluid_acc",
            "tracer_ini",
            "tracer_acc",
            "cell_mmin",
            "cell_mmax",
            "crs_mass",
            "fin_mass",
            "crs_n",
            "fin_n",
            "init_zmax",
            "c2f_count",
            "owner_err",
            "status_err",
            "count_err",
            "min_err",
        }
        missing = required_history.difference(history)
        if missing:
            pytest.fail(f"missing particle mass-transport histories: {sorted(missing)}")

        for name in required_history:
            if not np.all(np.isfinite(history[name])):
                pytest.fail(f"{name} contains non-finite values")
        for name in ("owner_err", "status_err", "count_err", "min_err"):
            if np.max(np.abs(history[name])) > 1.0e-12:
                pytest.fail(f"{name} is too large: {np.max(np.abs(history[name])):g}")

        initial_mass = history["init_mass"][0]
        initial_count = history["tracer_ini"][0]
        assert initial_mass > 0.0
        assert initial_count >= 8192
        np.testing.assert_allclose(history["init_mass"], initial_mass)
        np.testing.assert_allclose(history["tracer_ini"], initial_count)

        # The initialization must genuinely test mass weighting rather than
        # assigning a uniform particle count to equal-mass zones.  Cell mass
        # is measured only over the sampled region outside the horizon.
        cell_mass_min = history["cell_mmin"][0]
        cell_mass_max = history["cell_mmax"][0]
        assert cell_mass_min > 0.0
        assert cell_mass_max/cell_mass_min >= 16.0
        np.testing.assert_allclose(history["cell_mmin"], cell_mass_min)
        np.testing.assert_allclose(history["cell_mmax"], cell_mass_max)

        coarse_mass = history["crs_mass"][0]
        fine_mass = history["fin_mass"][0]
        coarse_count = history["crs_n"][0]
        fine_count = history["fin_n"][0]
        assert min(coarse_mass, fine_mass, coarse_count, fine_count) > 0.0
        np.testing.assert_allclose(coarse_mass + fine_mass, initial_mass)
        np.testing.assert_allclose(coarse_count + fine_count, initial_count)
        for name in (
            "crs_mass",
            "fin_mass",
            "crs_n",
            "fin_n",
        ):
            np.testing.assert_allclose(history[name], history[name][0])

        fine_mass_fraction = fine_mass/initial_mass
        fine_tracer_fraction = fine_count/initial_count
        refinement_sampling_error = 5.0*np.sqrt(
            fine_mass_fraction*(1.0-fine_mass_fraction)/initial_count
        )
        assert fine_tracer_fraction == pytest.approx(
            fine_mass_fraction, abs=max(refinement_sampling_error, 0.01)
        )

        # The problem generator also bins the initial fluid mass and tracers
        # jointly by radius and refinement level.  No populated bin may differ
        # by more than five sampling standard deviations.
        initialization_zmax = history["init_zmax"][0]
        np.testing.assert_allclose(history["init_zmax"], initialization_zmax)
        assert initialization_zmax <= 5.0

        _assert_monotone(history["fluid_acc"], "fluid_acc", 1.0e-12)
        _assert_monotone(history["tracer_acc"], "tracer_acc")
        _assert_monotone(history["c2f_count"], "c2f_count")
        assert history["c2f_count"][-1] > 0.0

        fluid_fraction = history["fluid_acc"][-1]/initial_mass
        tracer_fraction = history["tracer_acc"][-1]/initial_count
        assert 0.02 < fluid_fraction < 0.5

        # Equal-mass tracers sample the conserved mass flow.  Permit five
        # binomial standard deviations plus a two-percent discretization floor
        # for the Cartesian representation of the spherical horizon.
        sampling_error = 5.0*np.sqrt(
            fluid_fraction*(1.0-fluid_fraction)/initial_count
        )
        tolerance = max(sampling_error, 0.02)
        assert tracer_fraction == pytest.approx(fluid_fraction, abs=tolerance)

        if not TRACK.exists():
            pytest.fail(f"particle trajectory output was not written: {TRACK}")
        track = read_particle_track(TRACK)
        required_track = {
            "time", "cycle", "ptag", "status", "x", "y", "z",
            "x_min", "y_min", "z_min", "t_min",
        }
        missing = required_track.difference(track)
        if missing:
            pytest.fail(f"missing particle trajectory fields: {sorted(missing)}")

        radius = np.sqrt(track["x"]**2 + track["y"]**2 + track["z"]**2)
        minimum_radius = np.sqrt(
            track["x_min"]**2 + track["y_min"]**2 + track["z_min"]**2
        )
        assert np.all(np.isfinite(minimum_radius))
        assert np.all(np.isfinite(radius))
        assert np.all(track["t_min"] >= 0.0)
        assert np.all(track["t_min"] <= track["time"] + 1.0e-12)

        minimum_decreased = False
        for tag in np.unique(track["ptag"]):
            selected = np.flatnonzero(track["ptag"] == tag)
            order = np.lexsort((track["time"][selected], track["cycle"][selected]))
            tag_minimum = minimum_radius[selected][order]
            assert np.all(np.diff(tag_minimum) <= 1.0e-12)
            if tag_minimum[-1] < tag_minimum[0] - 1.0e-12:
                minimum_decreased = True
        assert minimum_decreased
    finally:
        HISTORY.unlink(missing_ok=True)
        shutil.rmtree("particle_track", ignore_errors=True)
        shutil.rmtree("pvtk", ignore_errors=True)
