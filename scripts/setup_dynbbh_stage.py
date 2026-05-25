#!/usr/bin/env python3
"""Create chained dynbbh run directories with conservative staged updates.

The helper is intentionally small: it copies a previous run's parfile,
executable, and launch wrapper, applies one named stage of parameter changes,
then writes a PBS script.  It is meant for restart-to-restart workflows where
the physics change should be explicit in the generated parfile diff.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path


CPU_BIND = (
    "list:1-8:9-16:17-24:25-32:33-40:41-48:"
    "53-60:61-68:69-76:77-84:85-92:93-100"
)


def latest_restart(run_dir):
    rank0 = run_dir / "rst" / "rank_00000000"
    restarts = sorted(rank0.glob("torus.*.rst"))
    if not restarts:
        raise SystemExit(f"no restart files found under {rank0}")
    return restarts[-1]


def infer_last_time(run_dir):
    stdout = run_dir / "stdout.log"
    if not stdout.exists():
        return None
    last = None
    pattern = re.compile(r"time=([0-9.eE+-]+)")
    for line in stdout.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            last = match.group(1)
    return last


def set_param(text, block, key, value):
    block_pat = re.compile(rf"(?m)^<{re.escape(block)}>\s*$")
    match = block_pat.search(text)
    if not match:
        return text.rstrip() + f"\n\n<{block}>\n{key} = {value}\n"

    next_block = re.search(r"(?m)^<[^>\n]+>\s*$", text[match.end():])
    end = match.end() + next_block.start() if next_block else len(text)
    body = text[match.end():end]
    key_pat = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*)[^#\n]*(.*)$")
    if key_pat.search(body):
        body = key_pat.sub(rf"\g<1>{value}\2", body, count=1)
    else:
        if body and not body.endswith("\n"):
            body += "\n"
        body += f"{key} = {value}\n"
    return text[:match.end()] + body + text[end:]


def apply_stage(text, args, source_run):
    if args.stage == "burnin-schwarzschild":
        updates = {
            ("problem", "a1"): "0.0",
            ("problem", "a2"): "0.0",
            ("problem", "spin_ramp"): "false",
            ("coord", "excise_to_horizon"): "false",
            ("coord", "excise_shrink_to_horizon"): "false",
        }
    elif args.stage == "shrink-to-horizon":
        updates = {
            ("coord", "excise_to_horizon"): "false",
            ("coord", "excise_shrink_to_horizon"): "true",
            ("coord", "excise_shrink_timescale"): str(args.shrink_timescale),
            ("problem", "spin_ramp"): "false",
        }
    elif args.stage == "horizon-start":
        updates = {
            ("coord", "excise_to_horizon"): "true",
            ("coord", "excise_shrink_to_horizon"): "false",
            ("problem", "spin_ramp"): "false",
        }
    elif args.stage == "spin-ramp":
        start = args.spin_ramp_start_time
        if start is None and source_run is not None:
            start = infer_last_time(source_run)
        if start is None:
            raise SystemExit("--spin-ramp-start-time is required without --source-run")
        updates = {
            ("problem", "a1"): str(args.spin_target),
            ("problem", "a2"): str(args.spin_target),
            ("problem", "spin_ramp"): "true",
            ("problem", "spin_ramp_timescale"): str(args.spin_ramp_timescale),
            ("problem", "spin_ramp_start_time"): str(start),
            ("coord", "excise_to_horizon"): "true",
            ("coord", "excise_shrink_to_horizon"): "false",
        }
    elif args.stage == "spin-static":
        updates = {
            ("problem", "a1"): str(args.spin_target),
            ("problem", "a2"): str(args.spin_target),
            ("problem", "spin_ramp"): "false",
            ("coord", "excise_to_horizon"): "true",
            ("coord", "excise_shrink_to_horizon"): "false",
        }
    else:
        raise SystemExit(f"unknown stage {args.stage}")

    for (block, key), value in updates.items():
        text = set_param(text, block, key, value)

    for override in args.set:
        try:
            lhs, value = override.split("=", 1)
            block, key = lhs.split("/", 1)
        except ValueError:
            raise SystemExit(
                f"bad --set '{override}', expected block/key=value"
            ) from None
        text = set_param(text, block.strip(), key.strip(), value.strip())

    return text


def write_launch(run_dir):
    launch = run_dir / "launch.sh"
    launch.write_text(f"""#!/bin/bash -l
set -u
set -o pipefail
RUN_DIR="$1"
EXE="$2"
RESTART="$3"
RUNTIME="$4"
cd "$RUN_DIR"
rank=${{PMI_RANK:-${{PMIX_RANK:-${{PALS_RANK:-0}}}}}}
if [[ "$rank" == "0" ]]; then
  exec >> "$RUN_DIR/stdout.log" 2>&1
else
  exec > /dev/null 2>&1
fi
echo "[{run_dir.name}] restart: $RESTART"
echo "[{run_dir.name}] exe: $EXE"
echo "[{run_dir.name}] parfile: $RUN_DIR/parfile.par"
echo "[{run_dir.name}] output dir: $RUN_DIR"
echo "[{run_dir.name}] wall-clock stop: $RUNTIME"
gpu_tile_compact.sh "$EXE" -d "$RUN_DIR" -t "$RUNTIME" -r "$RESTART" -i "$RUN_DIR/parfile.par"
""")
    launch.chmod(0o755)


def write_pbs(path, args, run_dir, exe_path, restart):
    path.write_text(f"""#!/bin/bash -l
#PBS -A {args.account}
#PBS -q {args.queue}
#PBS -N {args.job_name}
#PBS -l select={args.nodes}
#PBS -l walltime={args.walltime}
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -M {args.mail}
#PBS -m abe
#PBS -j oe
set -u
set -o pipefail
set -x
RUN_DIR={run_dir}
EXE={exe_path}
RESTART={restart}
RUNTIME={args.runtime}
RANKS={args.ranks}
RANKS_PER_NODE={args.ranks_per_node}
CPU_BIND="{args.cpu_bind}"
source /home/hzhu/athenak_env
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export FI_MR_CACHE_MONITOR=disabled
export FI_LOG_LEVEL=ERROR
if [[ ! -x "$EXE" ]]; then echo "missing exe $EXE" >&2; exit 3; fi
if [[ ! -f "$RESTART" ]]; then echo "missing restart $RESTART" >&2; exit 4; fi
rm -f "$RUN_DIR/stdout.log"
mkdir -p "$RUN_DIR/bin" "$RUN_DIR/rst"
mpiexec -n "$RANKS" -ppn "$RANKS_PER_NODE" --cpu-bind="$CPU_BIND" --wdir="$RUN_DIR" \\
  /bin/bash "$RUN_DIR/launch.sh" "$RUN_DIR" "$EXE" "$RESTART" "$RUNTIME"
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--stage", required=True, choices=[
        "burnin-schwarzschild",
        "shrink-to-horizon",
        "horizon-start",
        "spin-ramp",
        "spin-static",
    ])
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--template-parfile", type=Path)
    parser.add_argument("--exe", type=Path, default=Path("/home/hzhu/athenak/build_cb/src/athena"))
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--spin-target", type=float, default=0.9)
    parser.add_argument("--spin-ramp-timescale", type=float, default=785.0)
    parser.add_argument("--spin-ramp-start-time")
    parser.add_argument("--shrink-timescale", type=float, default=785.0)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--queue", default="debug-scaling")
    parser.add_argument("--account", default="BBHGRMHD")
    parser.add_argument("--nodes", type=int, default=22)
    parser.add_argument("--ranks", type=int, default=264)
    parser.add_argument("--ranks-per-node", type=int, default=12)
    parser.add_argument("--walltime", default="01:00:00")
    parser.add_argument("--runtime", default="00:50:00")
    parser.add_argument("--cpu-bind", default=CPU_BIND)
    parser.add_argument("--mail", default="hz0693@princeton.edu")
    parser.add_argument("--job-name")
    parser.add_argument("--pbs-name")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.source_run is None and args.template_parfile is None:
        raise SystemExit("provide --source-run or --template-parfile")

    run_dir = args.base_dir / args.case / "run_0"
    if run_dir.exists() and not args.force:
        raise SystemExit(f"{run_dir} already exists; use --force to replace")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "bin").mkdir()
    (run_dir / "rst").mkdir()

    source_run = args.source_run
    parfile = args.template_parfile or source_run / "parfile.par"
    restart = args.restart or latest_restart(source_run)
    if not parfile.exists():
        raise SystemExit(f"missing parfile {parfile}")
    if not restart.exists():
        raise SystemExit(f"missing restart {restart}")
    if not args.exe.exists():
        raise SystemExit(f"missing executable {args.exe}")

    shutil.copy2(args.exe, run_dir / "athena")
    (run_dir / "athena").chmod(0o755)
    write_launch(run_dir)

    text = apply_stage(parfile.read_text(), args, source_run)
    (run_dir / "parfile.par").write_text(text)

    args.job_name = args.job_name or args.case[:15]
    pbs_name = args.pbs_name or f"{args.case}.pbs"
    pbs_path = args.base_dir / pbs_name
    write_pbs(pbs_path, args, run_dir, run_dir / "athena", restart)

    print(f"run_dir={run_dir}")
    print(f"parfile={run_dir / 'parfile.par'}")
    print(f"pbs={pbs_path}")
    print(f"restart={restart}")
    if args.submit:
        result = subprocess.run(["qsub", str(pbs_path)], check=True,
                                text=True, capture_output=True)
        print(f"job_id={result.stdout.strip()}")


if __name__ == "__main__":
    main()
