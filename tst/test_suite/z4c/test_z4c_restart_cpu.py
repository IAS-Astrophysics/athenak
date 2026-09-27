"""Restart equivalence for evolved gauge, controllers and extraction state."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from .overhaul_utils import run_case, table


@pytest.mark.parametrize("controller", ["oscillator", "pid", "relaxation", "dob", "bdob"])
@pytest.mark.parametrize("per_rank", [False, True])
def test_restart_equivalence(tmp_path, controller, per_rank):
    settings = f"""
<mesh>
x1min=-2
x1max=2
x2min=-2
x2max=2
x3min=-2
x3max=2
<problem>
pgen_name=z4c_superposed_punctures
punc_1_rest_mass=0.1
punc_2_rest_mass=0.1
punc_1_velocity_x1=0.1
punc_2_velocity_x1=-0.1
<z4c>
spatial_order=4
telegraph_lapse=true
enable_driftcontrol=true
dc_variety={controller}
co_0_type=BH
co_0_x=0.15
co_0_y=0.1
co_0_z=0.05
dump_horizon_0=true
co_0_dump_radius=0.2
horizon_0_Nx=3
horizon_dt=0.001
<cce>
num_radii=1
rin_0=0.3
rout_0=0.5
num_l_modes=2
num_radial_modes=2
cce_dt=0.001
<time>
cfl_number=0.01
tlim=1
<output1>
variable=z4c
dt=0.0001
<output2>
file_type=rst
dt=0.0001
single_file_per_rank={str(per_rank).lower()}
"""
    full, split = tmp_path / "full", tmp_path / "split"
    run_case(full, settings + "\n<time>\nnlim=4\n")
    run_case(split, settings + "\n<time>\nnlim=2\n")
    checkpoint = sorted((split / "rst").rglob("*.rst"))[-1]
    tracker = next(split.glob("*.co_0.txt"))
    prefix = tracker.read_text()
    executable = str(Path(os.environ.get("ATHENA_OVERHAUL_EXE", "./athena")).resolve())
    result = subprocess.run([executable, "-r", str(checkpoint), "time/nlim=4"],
                            cwd=split, capture_output=True, text=True, timeout=90)
    (split / "restart.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert tracker.read_text().startswith(prefix)
    for name, expected in table(full).items():
        np.testing.assert_allclose(table(split)[name], expected, rtol=1e-11, atol=1e-13,
                                   err_msg=name)
    # Every output has the same path and contents after a split evolution.
    for subdir in ("horizon_0", "cce"):
        full_files = sorted(p.relative_to(full) for p in (full / subdir).rglob("*")
                            if p.is_file())
        split_files = sorted(p.relative_to(split) for p in (split / subdir).rglob("*")
                             if p.is_file())
        assert full_files == split_files
        assert full_files
        for path in full_files:
            assert (full / path).read_bytes() == (split / path).read_bytes(), path
