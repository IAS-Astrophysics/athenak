"""
SC restart test: a coupled radiation-hydro run resumed from its own restart file must
reproduce the uninterrupted run bit for bit.

The 2D radiatively damped acoustic wave (sc_linwave.athinput, affect_fluid on, the
radiation
relaxation rate sets dt) writes restart files every 0.5 time units. This test runs it to
t=1,
keeps the final error line (shared OutputErrors) and the dispersion-fit line, then
restarts from
the t=0.5 file and runs to t=1 again. Both lines must be identical strings.

What this certifies: the intensity is checkpointed and restored exactly, J is rebuilt
from it so
the first solve after the restart continues the iteration rather than starting from zero,
and
the relaxation timestep is a function of the current state alone (it refreshes the opacity
before evaluating the rate), so the restarted run takes exactly the same steps. There is
no
restart test elsewhere in the suite; this one is deliberately strict because the property
holds
exactly (verified 2026-09-11 on the conserved-variable and J dumps as well).

The deck is otherwise the AMR deck of the 2D linwave tests; here refinement is switched
off.
With AMR a restart is exact only if no remesh falls within refinement_interval cycles
after
the checkpoint: athenak writes outputs before the remesh of a cycle and does not
checkpoint the
per-block refinement clocks (ncyc_since_ref), so the restarted run remeshes later than the
continuous one. That is a property of the athenak restart, independent of this module (the
hydro-only wave shows it too), and is not what this test is about.
"""

import glob
import os
import shutil
import test_suite.testutils as testutils

_INPUT = "inputs/sc_linwave.athinput"
_UNIFORM = ["mesh_refinement/num_levels=1"]   # see the docstring
_ERRS = "SCLinWave-errs.dat"
_DISP = "SCLinWave-davis54.dat"


def _last_line(path):
    with open(path) as f:
        return f.read().splitlines()[-1]


def test_sc_restart():
    testutils.cleanup()
    shutil.rmtree("rst", ignore_errors=True)
    try:
        assert testutils.run(_INPUT, _UNIFORM), "uninterrupted run failed"
        full_errs, full_disp = _last_line(_ERRS), _last_line(_DISP)
        rst_files = sorted(glob.glob("rst/SCLinWave.*.rst"))
        assert len(rst_files) >= 2, \
            f"expected restart files at t=0 and t=0.5, found {rst_files}"
        half = rst_files[1]   # 00000 is t=0, 00001 the first dump at t >= 0.5

        os.remove(_ERRS)
        os.remove(_DISP)
        assert testutils.run_command(["./athena", "-r", half] + _UNIFORM), \
            f"restart from {half} failed"
        rst_errs, rst_disp = _last_line(_ERRS), _last_line(_DISP)

        assert rst_errs == full_errs, \
            (f"restarted run differs from the full run:\n  full:    {full_errs}\n  "
             f"restart: {rst_errs}")
        assert rst_disp == full_disp, \
            (f"restarted dispersion fit differs:\n  full:    {full_disp}\n  restart: "
             f"{rst_disp}")
    finally:
        testutils.cleanup()
        shutil.rmtree("rst", ignore_errors=True)
