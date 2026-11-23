"""
Regression test: grey M1 single zone with constant emission and absorption.

The box sits inside the emitting sphere of the rad_m1_spheretest toy opacity, so every
cell has eta = kappa_abs = 10 in flat space with the fluid at rest, and the radiation
stays uniform.  E then obeys dE/dt = eta - kappa E with E(0) = 0, and each source
treatment must reproduce its own discrete update exactly:

  explicit sources               2-stage midpoint:  E1 = E + dt/2 S(E), E += dt S(E1)
  implicit, solver forced on     backward Euler:    E = (E + dt eta) / (1 + dt kappa)
  implicit, default thin limit   forward Euler:     E += dt S(E)   (kappa dt < 1/3)

The time step is cfl * dx = 0.025, which divides the output cadence exactly.  On top
of the recurrences the test checks that E stays uniform, F stays zero, and the error
against the exact solution 1 - exp(-10 t) converges at the expected order.
"""

# Modules
import glob
import re
import numpy as np
import pytest
import test_suite.testutils as testutils

input_file = "inputs/rad_m1_singlezone_toy.athinput"
ETA = KAPPA = 10.0
DX = 0.5 / 8


def step(E, dt, mode):
    """One time step of the discrete update used by each source treatment."""
    if mode == "explicit":
        E1 = E + 0.5 * dt * (ETA - KAPPA * E)
        return E + dt * (ETA - KAPPA * E1)
    if mode == "backward_euler":
        return (E + dt * ETA) / (1.0 + dt * KAPPA)
    return E + dt * (ETA - KAPPA * E)  # forward Euler


def flags_for(mode, cfl):
    flags = [f"time/cfl_number={cfl}"]
    if mode == "explicit":
        flags.append("radiation_m1/src_update=explicit")
    elif mode == "backward_euler":
        flags += ["radiation_m1/source_thin_limit=0", "radiation_m1/source_epsrel=1e-13"]
    return flags


def read_outputs():
    """Return a list of (time, E array, F array) for every tab output.

    Columns are found by name, so E and F hold every species (1 without bns_nurates,
    4 with it); with the toy opacity all species evolve identically.
    """
    out = []
    for f in sorted(glob.glob("tab/m1_singlezone.rad_m1_moments.*.tab")):
        with open(f) as fh:
            t = float(re.search(r"time=([0-9.eE+-]+)", fh.readline()).group(1))
            names = fh.readline().split()
        names = names[names.index("x3v") + 1:]
        d = np.loadtxt(f, ndmin=2)[:, -len(names):]
        E = d[:, [n.startswith("E:") for n in names]]
        F = d[:, [n.startswith("F") for n in names]]
        out.append((t, E, F))
    return out


def run_case(mode, cfl):
    """Run one configuration; return (max recurrence error, max error vs exact)."""
    testutils.cleanup()
    assert testutils.run(input_file, flags_for(mode, cfl)), f"{mode} cfl={cfl} failed"
    outputs = read_outputs()
    assert len(outputs) == 11, f"expected 11 outputs, found {len(outputs)}"
    dt = cfl * DX
    err_rec = err_exact = 0.0
    for t, E, F in outputs:
        assert np.ptp(E) == 0.0, f"{mode}: E not uniform at t={t}"
        assert np.abs(F).max() == 0.0, f"{mode}: F nonzero at t={t}"
        n = int(round(t / dt))
        Erec = 0.0
        for _ in range(n):
            Erec = step(Erec, dt, mode)
        err_rec = max(err_rec, np.abs(E - Erec).max())
        err_exact = max(err_exact, np.abs(E - (1.0 - np.exp(-KAPPA * t))).max())
    return err_rec, err_exact


# (mode, tolerance on the recurrence, expected convergence ratio when dt is halved)
cases = [("explicit", 1e-13, 0.25),
         ("backward_euler", 1e-11, 0.5),
         ("forward_euler", 1e-13, 0.5)]


@pytest.mark.parametrize("mode,tol,ratio", cases)
def test_m1_singlezone(mode, tol, ratio):
    try:
        err_rec0, err_exact0 = run_case(mode, 0.4)
        err_rec1, err_exact1 = run_case(mode, 0.2)
        for err in (err_rec0, err_rec1):
            if err > tol:
                pytest.fail(f"{mode}: E differs from the discrete update by {err:g}")
        measured = err_exact1 / err_exact0
        if abs(measured - ratio) > 0.1:
            pytest.fail(f"{mode}: error ratio {measured:g} when halving dt, "
                        f"expected about {ratio:g}")
    finally:
        testutils.cleanup()
