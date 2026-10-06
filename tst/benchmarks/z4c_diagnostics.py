"""Compare full diagnostics and CPU/Serial kernel timings between two binaries.

Build both binaries with identical Release flags and the pinned Kokkos submodule.
The baseline should be PR #790 head 22baa243; the candidate includes the optimization.
From the repository root:
  c++ -O2 -std=c++17 -shared -fPIC tst/benchmarks/z4c_diagnostics_timer.cpp -o timer.so
  python tst/benchmarks/z4c_diagnostics.py --baseline /path/to/baseline/athena \
      --candidate /path/to/candidate/athena --profiler ./timer.so

Timing is only valid for a synchronous CPU/Serial backend. It excludes output I/O,
initialization and evolution kernels, discards the first invocation per process,
and alternates baseline/candidate processes to reduce ordering bias. All 17
double-precision fields on a 1D output slice are compared; the timed kernel
still computes the full 3D grid. No threshold
is enforced: performance depends on backend, compiler, grid and stencil.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def run(executable, profiler, directory, order, problem, size, steps):
    directory.mkdir()
    text = (ROOT / "tst/inputs/lwave_z4c.athinput").read_text()
    text += f"""
<mesh>
nghost=4
nx1={size}
nx2={size}
nx3={size}
<meshblock>
nx1={size}
nx2={size}
nx3={size}
<mesh_refinement>
refinement=none
<z4c>
spatial_order={order}
<time>
nlim={steps}
tlim=1
cfl_number=0.001
<problem>
pgen_name={problem}
amp=0.001
punc_1_velocity_x1=0.3
punc_2_velocity_x1=-0.2
<output1>
file_type=tab
variable=z4c_diag
dt=1e-10
slice_x2=0.5
slice_x3=0.5
data_format=%24.16e
"""
    source = directory / "test.athinput"
    source.write_text(text)
    env = dict(os.environ, KOKKOS_TOOLS_LIBS=str(profiler))
    result = subprocess.run([str(executable), "-i", str(source)], cwd=directory,
                            env=env, capture_output=True, text=True, timeout=180)
    (directory / "run.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    times = [int(x) for x in re.findall(r"Z4C_DIAGNOSTIC_NS (\d+)", result.stdout)]
    if len(times) < 2:
        raise RuntimeError("No repeated kernel timings; use a Serial build with Kokkos Tools")
    files = sorted((directory / "tab").glob("*.tab"))
    data = np.atleast_2d(np.loadtxt(files[-1]))
    if not np.isfinite(data).all():
        raise RuntimeError("Nonfinite diagnostic output")
    return times[1:], data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "profiler"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for order in (2, 4, 6):
            for problem in ("z4c_linear_wave", "z4c_superposed_punctures"):
                times = {"baseline": [], "candidate": []}
                max_abs = 0.0
                for repeat in range(args.repeats):
                    outputs = {}
                    labels = ("baseline", "candidate") if repeat % 2 == 0 else (
                        "candidate", "baseline")
                    for label in labels:
                        directory = Path(tmp) / f"{order}-{problem}-{repeat}-{label}"
                        samples, data = run(getattr(args, label).resolve(),
                                            args.profiler.resolve(), directory, order,
                                            problem, args.size, args.steps)
                        times[label].extend(samples)
                        outputs[label] = data
                    np.testing.assert_allclose(outputs["candidate"], outputs["baseline"],
                                               rtol=2e-11, atol=2e-13)
                    max_abs = max(max_abs, float(np.max(np.abs(
                        outputs["candidate"] - outputs["baseline"]))))
                medians = {key: statistics.median(value) / args.size**3
                           for key, value in times.items()}
                row = {"order": order, "problem": problem, "ns_per_cell": medians,
                       "speedup": medians["baseline"] / medians["candidate"],
                       "samples_per_binary": len(times["baseline"]),
                       "max_abs_difference": max_abs}
                results.append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps({"size": args.size, "steps": args.steps,
                      "repeats": args.repeats, "results": results}, indent=2))


if __name__ == "__main__":
    main()
