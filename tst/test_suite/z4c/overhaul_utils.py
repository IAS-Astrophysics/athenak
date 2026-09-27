"""Small, bounded feature tests usable with serial, MPI, and CUDA executables."""
import os
from pathlib import Path
import subprocess
import shlex

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def run_case(directory, extra="", executable=None, success=True, slices=True):
    directory.mkdir(parents=True, exist_ok=True)
    executable = executable or os.environ.get("ATHENA_OVERHAUL_EXE", "./athena")
    executable = str(Path(executable).resolve())
    text = (ROOT / "tst/inputs/lwave_z4c.athinput").read_text()
    text += """
<mesh>
nx1=8
nx2=8
nx3=8
nghost=4
<meshblock>
nx1=8
nx2=8
nx3=8
<mesh_refinement>
refinement=none
<problem>
amp=0
<time>
nlim=0
tlim=0.01
<output1>
file_type=tab
variable=adm
dt=0.001
slice_x2=0.5
slice_x3=0.5
data_format=%24.16e
"""
    if not slices:
        text = "\n".join(line for line in text.splitlines() if not line.startswith("slice_"))
    text += extra
    if os.environ.get("ATHENA_TEST_LAUNCHER"):
        text += "\n<mesh>\nnx1=16\n"

    source = directory / "test.athinput"
    source.write_text(text)
    result = subprocess.run(shlex.split(os.environ.get("ATHENA_TEST_LAUNCHER", "")) +
                            [executable, "-i", str(source)], cwd=directory,
                            capture_output=True, text=True, timeout=90)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


def table(directory, last=True):
    files = sorted((directory / "tab").glob("*.tab"))
    assert files
    path = files[-1] if last else files[0]
    names = path.read_text().splitlines()[1].lstrip("# ").split()
    data = np.atleast_2d(np.loadtxt(path))
    assert np.isfinite(data).all(), path
    return dict(zip(names, data.T))
