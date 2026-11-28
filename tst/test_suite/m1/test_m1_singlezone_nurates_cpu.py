"""
Regression test: grey M1 + bns_nurates single zone equilibration.

Five identical cells of hot, neutron-rich matter (rho = 1.619e-5, T = 50 MeV, Y_e = 0.1,
SFHo test table) start with no neutrinos and exchange energy and lepton number with the
four neutrino species until they reach equilibrium, with the production M1 and
bns_nurates settings (the code defaults plus NEPS).  The space is flat and the fluid is
at rest, so the test checks that

  * every cell stays identical;
  * total energy tau + sum_s E_s is conserved;
  * lepton number is conserved: D Y_e changes by -m_b (N_nue - N_anue), with one m_b;
  * the final state is stationary, with nu_x and anti-nu_x equal;
  * the equilibrium (Y_e, T, E_s) matches the reference.  The zone is optically thick,
    so the rates use Kirchhoff's law and the equilibrium is set by thermodynamics, not
    by the size of the rates: switching weak magnetism, medium corrections, NEPS or the
    distribution choice on or off moves it by less than 1e-4.

Needs a build with -D Athena_ENABLE_NURATES=ON and bns_nurates in src/bns_nurates; it
is skipped otherwise.
"""

# Modules
import glob
import re
import numpy as np
import pytest
import test_suite.testutils as testutils

input_file = "inputs/rad_m1_singlezone_nurates.athinput"

# equilibrium reached by the run, recorded with bns_nurates main (22114c7)
REFERENCE = {"Y_e": 0.27601, "T": 42.488,
             "E": [1.9621e-07, 6.2414e-07, 7.0484e-07, 7.0484e-07]}
REF_RTOL = 2e-3


def nurates_enabled():
    """True if the current build (tst/build) was configured with bns_nurates."""
    try:
        with open("../CMakeCache.txt") as fh:
            return "Athena_ENABLE_NURATES:BOOL=ON" in fh.read()
    except OSError:
        return False


def read_tab(fname):
    """Return (time, {column name: array}) for one tab output."""
    with open(fname) as fh:
        line0, line1 = fh.readline(), fh.readline()
    t = float(re.search(r"time=([0-9.eE+-]+)", line0).group(1))
    names = line1.split()
    names = names[names.index("x3v") + 1:]
    d = np.loadtxt(fname, ndmin=2)[:, -len(names):]
    return t, dict(zip(names, d.T))


def read_outputs():
    """Return a list of (time, {variable: array}) for every dump time.

    The run ends on the cycle limit, which writes the last dump a second time; repeated
    times are skipped.
    """
    out = []
    for f in sorted(glob.glob("tab/singlezone.mhd_u.*.tab")):
        t, data = read_tab(f)
        if out and t == out[-1][0]:
            continue
        for v in ("mhd_w", "rad_m1_E", "rad_m1_N"):
            data.update(read_tab(f.replace("mhd_u", v))[1])
        out.append((t, data))
    return out


def test_m1_singlezone_nurates():
    if not nurates_enabled():
        pytest.skip("build does not have Athena_ENABLE_NURATES=ON")
    try:
        testutils.cleanup()
        assert testutils.run(input_file), "single zone run failed"
        outputs = read_outputs()
        assert len(outputs) == 9, f"expected 9 dump times, found {len(outputs)}"

        E = [f"E:{s}" for s in range(4)]
        _, d0 = outputs[0]
        etot0 = d0["ener"][0] + sum(d0[e][0] for e in E)
        mb = []
        for t, d in outputs:
            for name, val in d.items():
                assert np.all(np.isfinite(val)), f"{name} not finite at t={t}"
                assert np.ptp(val) == 0.0, f"{name} differs between cells at t={t}"
            etot = d["ener"][0] + sum(d[e][0] for e in E)
            if abs(etot - etot0) > 1e-12 * abs(etot0):
                pytest.fail(f"total energy changed by {(etot - etot0) / etot0:g} "
                            f"at t={t}")
            if t > 0:
                dlep = d["N:0"][0] - d["N:1"][0]
                mb.append(-(d["r_00"][0] - d0["r_00"][0]) / dlep)
        mb = np.array(mb)
        if np.ptp(mb) > 1e-8 * abs(mb.mean()):
            pytest.fail(f"lepton number not conserved: D Y_e / (N_nue - N_anue) "
                        f"varies by {np.ptp(mb) / mb.mean():g}")

        _, prev = outputs[-3]
        _, last = outputs[-1]
        column = {"Y_e": "s_00", "T": "temperature"}
        got = {key: last[col][0] for key, col in column.items()}
        for key, col in column.items():
            change = abs(got[key] - prev[col][0])
            if change > 1e-3 * got[key]:
                pytest.fail(f"{key} still changing at the end of the run: {change:g}")
        nux, anux = last["E:2"][0], last["E:3"][0]
        if abs(nux - anux) > 1e-6 * nux:
            pytest.fail(f"nu_x and anti-nu_x differ: {nux:g} vs {anux:g}")

        for key in got:
            if abs(got[key] - REFERENCE[key]) > REF_RTOL * REFERENCE[key]:
                pytest.fail(f"equilibrium {key} = {got[key]:g}, "
                            f"reference {REFERENCE[key]:g}")
        for s, ref in enumerate(REFERENCE["E"]):
            val = last[f"E:{s}"][0]
            if abs(val - ref) > REF_RTOL * ref:
                pytest.fail(f"equilibrium E_{s} = {val:g}, reference {ref:g}")
    finally:
        testutils.cleanup()
