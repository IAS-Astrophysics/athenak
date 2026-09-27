"""Bounded integration check using upstream's single-puncture pgen and FastFlow."""
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np

from .overhaul_utils import ROOT


def test_fastflow(tmp_path):
    executable = str(Path(os.environ.get("ATHENA_OVERHAUL_EXE", "./athena")).resolve())
    args = [executable, "-i", str(ROOT / "tst/inputs/z4c_boosted.athinput"),
            "time/nlim=2", "mesh_refinement/refinement=none",
            "problem/punc_velocity_x1=0", "job/basename=fastflow",
            "fastflow/use_puncture_0=-1"]
    for axis in range(1, 4):
        args += [f"mesh/nx{axis}=32", f"meshblock/nx{axis}=16",
                 f"mesh/x{axis}min=-2", f"mesh/x{axis}max=2"]
    result = subprocess.run(shlex.split(os.environ.get("ATHENA_TEST_LAUNCHER", ""))+args,
                            cwd=tmp_path, capture_output=True, text=True, timeout=90)
    (tmp_path / "run.log").write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    path = tmp_path / "fastflow.horizon_summary_0.txt"
    data = np.atleast_2d(np.loadtxt(path))
    assert len(data) > 0 and np.isfinite(data).all()
    # Read the horizon expansion residual by name using the upstream reader.
    import importlib.util
    spec = importlib.util.spec_from_file_location("athena_read", ROOT / "vis/python/athena_read.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    horizon = reader.horizon(str(path))
    assert np.max(horizon["hrms"]) < 0.03
