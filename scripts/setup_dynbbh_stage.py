#!/usr/bin/env python3
"""Create chained dynbbh run directories with conservative staged updates.

The helper is intentionally small: it copies a previous run's parfile,
executable, and launch wrapper, applies one named stage of parameter changes,
then writes a PBS script.  It is meant for restart-to-restart workflows where
the physics change should be explicit in the generated parfile diff.
"""

import argparse
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
from pathlib import Path


CPU_BIND = (
    "list:1-8:9-16:17-24:25-32:33-40:41-48:"
    "53-60:61-68:69-76:77-84:85-92:93-100"
)
ORBIT_DURATION = 785.398163397
DEFAULT_RANKS = 264
DEFAULT_RANKS_PER_NODE = 12
DEFAULT_NODES_PER_RUN = 22
SPIN_CASES = (
    ("spin_m0p7", 0.7, 180.0, "anti-aligned"),
    ("spin_0", 0.0, 0.0, "zero"),
    ("spin_p0p7", 0.7, 0.0, "aligned"),
    ("spin_p0p9", 0.9, 0.0, "aligned"),
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
        def repl(key_match):
            suffix = key_match.group(2)
            if suffix.startswith("#"):
                suffix = " " + suffix
            return f"{key_match.group(1)}{value}{suffix}"
        body = key_pat.sub(repl, body, count=1)
    else:
        if body and not body.endswith("\n"):
            body += "\n"
        body += f"{key} = {value}\n"
    return text[:match.end()] + body + text[end:]


def format_real(value):
    return f"{value:.12g}"


def iter_blocks(text):
    matches = list(re.finditer(r"(?m)^<([^>\n]+)>\s*$", text))
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        yield match.group(1), match.end(), end


def find_output_block(text, file_type):
    file_type_pat = re.compile(
        rf"(?m)^\s*file_type\s*=\s*{re.escape(file_type)}(?:\s|#|$)"
    )
    for block, start, end in iter_blocks(text):
        if block.startswith("output") and file_type_pat.search(text[start:end]):
            return block
    return None


def set_restart_output_policy(text, restart_dt):
    block = find_output_block(text, "rst")
    if block is None:
        raise SystemExit("no restart output block with file_type=rst found")
    text = set_param(text, block, "file_type", "rst")
    text = set_param(text, block, "dt", format_real(restart_dt))
    text = set_param(text, block, "single_file_per_rank", "true")
    return strip_output_runtime_counters(text)


def strip_output_runtime_counters(text):
    blocks = list(iter_blocks(text))
    for block, start, end in reversed(blocks):
        if not block.startswith("output"):
            continue
        body = text[start:end]
        body = re.sub(r"(?m)^\s*(?:file_number|last_time)\s*=.*(?:\n|$)", "", body)
        text = text[:start] + body + text[end:]
    return text


def parse_case_source(spec):
    try:
        case, rest = spec.split("=", 1)
    except ValueError:
        raise SystemExit(
            f"bad --case-source '{spec}', expected CASE=template.par,restart.rst"
        ) from None
    parts = [part.strip() for part in rest.split(",")]
    if len(parts) not in (2, 3):
        raise SystemExit(
            f"bad --case-source '{spec}', expected CASE=template.par,restart.rst[,label]"
        )
    case = case.strip().upper()
    if not case:
        raise SystemExit(f"bad --case-source '{spec}', empty case name")
    label = parts[2] if len(parts) == 3 else ""
    return {
        "case": case,
        "template": Path(parts[0]).expanduser(),
        "restart": Path(parts[1]).expanduser(),
        "label": label,
    }


def restart_root_and_name(rank0_restart):
    rank0_restart = Path(rank0_restart)
    if rank0_restart.parent.name != "rank_00000000":
        raise SystemExit(
            f"{rank0_restart} is not inside a rank_00000000 directory; "
            "single_file_per_rank restarts must be specified by their rank-0 file"
        )
    return rank0_restart.parent.parent, rank0_restart.name


def check_restart_set(rank0_restart, expected_ranks):
    root, name = restart_root_and_name(rank0_restart)
    missing = []
    for rank in range(expected_ranks):
        candidate = root / f"rank_{rank:08d}" / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            missing.append(candidate)
            if len(missing) >= 5:
                break
    if missing:
        first = "\n  ".join(str(path) for path in missing)
        raise SystemExit(
            f"restart set for {rank0_restart} is incomplete; missing/empty examples:\n"
            f"  {first}"
        )
    return root, name


def read_restart_state(rank0_restart):
    data = Path(rank0_restart).read_bytes()[:50000]
    marker = b"<par_end>"
    loc = data.find(marker)
    if loc < 0:
        raise SystemExit(f"<par_end> not found in first 50 KB of {rank0_restart}")
    header = loc + 10
    for real_size, fmt in ((8, "<d"), (4, "<f")):
        time_offset = header + 2 * 4 + 9 * real_size + 2 * 19 * 4
        try:
            time = struct.unpack_from(fmt, data, time_offset)[0]
            dt = struct.unpack_from(fmt, data, time_offset + real_size)[0]
            cycle = struct.unpack_from("<i", data, time_offset + 2 * real_size)[0]
        except struct.error:
            continue
        if math.isfinite(time) and time >= 0.0 and math.isfinite(dt) and dt > 0.0:
            return {"time": time, "dt": dt, "cycle": cycle, "real_size": real_size}
    raise SystemExit(f"could not decode restart mesh time from {rank0_restart}")


def copy_restart_set(src_rank0, dst_rst, expected_ranks):
    src_root, name = check_restart_set(src_rank0, expected_ranks)
    dst_rst.mkdir(parents=True, exist_ok=True)
    for rank in range(expected_ranks):
        rank_dir = f"rank_{rank:08d}"
        dst_dir = dst_rst / rank_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / rank_dir / name, dst_dir / name)
        if rank == 0 or (rank + 1) % 32 == 0 or rank + 1 == expected_ranks:
            print(f"copied {rank + 1}/{expected_ranks} restart ranks for {name}",
                  flush=True)
    return dst_rst / "rank_00000000" / name


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


def write_restart_time_helper(path):
    path.write_text("""#!/usr/bin/env python3
import math
import struct
import sys
from pathlib import Path

data = Path(sys.argv[1]).read_bytes()[:50000]
loc = data.find(b"<par_end>")
if loc < 0:
    raise SystemExit(f"<par_end> not found in {sys.argv[1]}")
header = loc + 10
for real_size, fmt in ((8, "<d"), (4, "<f")):
    offset = header + 2 * 4 + 9 * real_size + 2 * 19 * 4
    try:
        time = struct.unpack_from(fmt, data, offset)[0]
        dt = struct.unpack_from(fmt, data, offset + real_size)[0]
    except struct.error:
        continue
    if math.isfinite(time) and time >= 0.0 and math.isfinite(dt) and dt > 0.0:
        print(f"{time:.17g}")
        raise SystemExit(0)
raise SystemExit(f"could not decode restart time from {sys.argv[1]}")
""")
    path.chmod(0o755)


def write_zoom_launch(path, *, label, run_dir, exe, input_file, runtime,
                      initial_restart="", initial_restart_root="", spin_ramp=False,
                      helper="", require_source_advanced=False):
    initial_restart = shlex.quote(str(initial_restart)) if initial_restart else ""
    initial_restart_root = (
        shlex.quote(str(initial_restart_root)) if initial_restart_root else ""
    )
    helper = shlex.quote(str(helper)) if helper else ""
    path.write_text(f"""#!/bin/bash -l
set -euo pipefail

RUN_DIR={shlex.quote(str(run_dir))}
EXE={shlex.quote(str(exe))}
INPUT={shlex.quote(str(input_file))}
INITIAL_RESTART={initial_restart}
INITIAL_RESTART_ROOT={initial_restart_root}
RESTART_TIME_HELPER={helper}
RUNTIME="${{1:-{runtime}}}"
SPIN_RAMP={"1" if spin_ramp else "0"}
REQUIRE_SOURCE_ADVANCED={"1" if require_source_advanced else "0"}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/rst"
cd "$RUN_DIR"

rank=${{PMI_RANK:-${{PMIX_RANK:-${{PALS_RANK:-0}}}}}}
if [[ "$rank" == "0" ]]; then
  exec >> "$RUN_DIR/stdout.log" 2>&1
else
  exec > /dev/null 2>&1
fi

select_latest_restart() {{
  local root="$1"
  if [[ -n "$root" && -d "$root/rank_00000000" ]] &&
     compgen -G "$root/rank_00000000/*.rst" >/dev/null; then
    ls -1t "$root"/rank_00000000/*.rst | head -n1
  elif [[ -n "$root" && -d "$root" ]] &&
       compgen -G "$root/*.rst" >/dev/null; then
    ls -1t "$root"/*.rst | head -n1
  fi
}}

RESTART="$(select_latest_restart "$RUN_DIR/rst" || true)"
RESTART_SOURCE=own
if [[ -z "$RESTART" ]]; then
  RESTART="$(select_latest_restart "$INITIAL_RESTART_ROOT" || true)"
  RESTART_SOURCE=source
fi
if [[ -z "$RESTART" ]]; then
  RESTART="$INITIAL_RESTART"
  RESTART_SOURCE=initial
fi
if [[ -z "$RESTART" || ! -s "$RESTART" ]]; then
  echo "missing restart for {label}: $RESTART" >&2
  exit 4
fi
if [[ "$REQUIRE_SOURCE_ADVANCED" == "1" && "$RESTART_SOURCE" == "source" ]]; then
  source_count=0
  if [[ -d "$INITIAL_RESTART_ROOT/rank_00000000" ]]; then
    source_count=$(find "$INITIAL_RESTART_ROOT/rank_00000000" -maxdepth 1 -name '*.rst' | wc -l)
  fi
  if (( source_count < 2 )); then
    echo "[{label}] stage-2 source root has only $source_count restart(s): $INITIAL_RESTART_ROOT" >&2
    echo "[{label}] refusing to start spin survey before a post-horizon stage-2 restart exists" >&2
    exit 5
  fi
fi
test -x "$EXE"
test -s "$INPUT"

extra_args=()
if [[ "$SPIN_RAMP" == "1" ]]; then
  test -x "$RESTART_TIME_HELPER"
  restart_time="$(python "$RESTART_TIME_HELPER" "$RESTART")"
  extra_args+=("problem/spin_ramp_start_time=$restart_time")
  echo "[{label}] spin ramp start time override: $restart_time"
fi

echo "[{label}] restart: $RESTART"
echo "[{label}] restart source: $RESTART_SOURCE"
echo "[{label}] input: $INPUT"
echo "[{label}] executable: $EXE"
echo "[{label}] run dir: $RUN_DIR"
echo "[{label}] runtime: $RUNTIME"
gpu_tile_compact.sh "$EXE" -d "$RUN_DIR" -t "$RUNTIME" -r "$RESTART" -i "$INPUT" "${{extra_args[@]}}"
""")
    path.chmod(0o755)


def write_bundle_pbs(path, *, root, runs, account, queue, job_name, walltime,
                     runtime, max_iters, ranks, ranks_per_node, nodes_per_run,
                     cpu_bind, auto_resubmit=False):
    run_lines = "\n".join(
        f'  "{run["name"]}|{run["run_dir"]}|{run["launch"]}"' for run in runs
    )
    total_nodes = len(runs) * nodes_per_run
    path.write_text(f"""#!/bin/bash -l
#PBS -A {account}
#PBS -q {queue}
#PBS -N {job_name}
#PBS -l select={total_nodes}
#PBS -l walltime={walltime}
#PBS -l filesystems=flare:home
#PBS -l place=scatter
#PBS -j oe
#PBS -o {root}/logs/{path.stem}.log

set -u
set -o pipefail
set -x

ROOT={root}
RUNTIME={runtime}
MAX_ITERS={max_iters}
RANKS={ranks}
RANKS_PER_NODE={ranks_per_node}
NODES_PER_RUN={nodes_per_run}
CPU_BIND="{cpu_bind}"
AUTO_RESUBMIT={1 if auto_resubmit else 0}
SELF={path}

RUN_SPECS=(
{run_lines}
)

source /home/hzhu/athenak_env
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export FI_MR_CACHE_MONITOR=disabled
export FI_LOG_LEVEL=ERROR

mkdir -p "$ROOT/logs"
mapfile -t HOSTS < <(sort -u "$PBS_NODEFILE")
NNODES=${{#HOSTS[@]}}
TOTAL_NODES_NEEDED=$(( ${{#RUN_SPECS[@]}} * NODES_PER_RUN ))
if (( NNODES < TOTAL_NODES_NEEDED )); then
  echo "Need $TOTAL_NODES_NEEDED nodes, allocated $NNODES" >&2
  exit 2
fi

for iter in $(seq 1 "$MAX_ITERS"); do
  echo "==== Starting internal iteration $iter of $MAX_ITERS ===="
  for spec in "${{RUN_SPECS[@]}}"; do
    IFS='|' read -r name run_dir launch <<< "$spec"
    mkdir -p "$run_dir"
    if [[ -f "$run_dir/stdout.log" ]]; then
      cat "$run_dir/stdout.log" >> "$run_dir/stdout_history.log"
      > "$run_dir/stdout.log"
    fi
  done

  pids=()
  idx=0
  for spec in "${{RUN_SPECS[@]}}"; do
    IFS='|' read -r name run_dir launch <<< "$spec"
    start=$(( idx * NODES_PER_RUN ))
    job_hosts=( "${{HOSTS[@]:$start:$NODES_PER_RUN}}" )
    job_hosts_str="$(IFS=,; echo "${{job_hosts[*]}}")"
    test -x "$launch"
    echo "Launching $name on $job_hosts_str"
    (
      mpiexec -n "$RANKS" -ppn "$RANKS_PER_NODE" --hosts "$job_hosts_str" \\
        --wdir="$run_dir" --cpu-bind="$CPU_BIND" \\
        /bin/bash "$launch" "$RUNTIME"
    ) &
    pids+=("$!")
    idx=$(( idx + 1 ))
  done

  status=0
  for pid in "${{pids[@]}}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "At least one run failed in iteration $iter" >&2
    exit "$status"
  fi

  need_restart=0
  for spec in "${{RUN_SPECS[@]}}"; do
    IFS='|' read -r name run_dir launch <<< "$spec"
    if [[ -s "$run_dir/stdout.log" ]] &&
       grep -Fq "Terminating on wall clock limit" "$run_dir/stdout.log"; then
      need_restart=1
      break
    fi
  done
  if (( need_restart == 0 )); then
    echo "All runs finished naturally in iteration $iter."
    break
  fi
done

need_resubmit=0
for spec in "${{RUN_SPECS[@]}}"; do
  IFS='|' read -r name run_dir launch <<< "$spec"
  if [[ -s "$run_dir/stdout.log" ]] &&
     grep -Fq "Terminating on wall clock limit" "$run_dir/stdout.log"; then
    need_resubmit=1
    echo "Detected wall-clock termination in $name."
    break
  fi
done

if (( need_resubmit && AUTO_RESUBMIT )); then
  qsub -W depend=afterany:${{PBS_JOBID}} "$SELF"
elif (( need_resubmit )); then
  echo "Additional work remains. Resubmit manually with: qsub $SELF"
else
  echo "All runs finished without time-limit termination."
fi
""")
    path.chmod(0o755)


def apply_common_zoom_updates(text, *, root, case, stage, source_restart,
                              source_state, tlim, restart_dt):
    updates = {
        ("comment", "zoom_workflow_root"): str(root),
        ("comment", "zoom_case"): case,
        ("comment", "zoom_stage"): stage,
        ("comment", "zoom_source_restart"): str(source_restart),
        ("comment", "zoom_source_time"): format_real(source_state["time"]),
        ("comment", "zoom_source_cycle"): str(source_state["cycle"]),
        ("time", "nlim"): "-1",
        ("time", "tlim"): format_real(tlim),
    }
    for (block, key), value in updates.items():
        text = set_param(text, block, key, value)
    return set_restart_output_policy(text, restart_dt)


def stage2_horizon_text(text, *, root, case, source_restart, source_state,
                        orbit_duration, restart_dt):
    tlim = source_state["time"] + orbit_duration
    text = apply_common_zoom_updates(
        text, root=root, case=case, stage="stage2_horizon_zero_spin",
        source_restart=source_restart, source_state=source_state, tlim=tlim,
        restart_dt=restart_dt)
    updates = {
        ("job", "restart_file"): str(source_restart),
        ("coord", "excise_1_rad"): "4.0",
        ("coord", "excise_2_rad"): "4.0",
        ("coord", "excise_to_horizon"): "false",
        ("coord", "excise_shrink_to_horizon"): "true",
        ("coord", "excise_shrink_timescale"): format_real(orbit_duration),
        ("problem", "a1"): "0.0",
        ("problem", "a2"): "0.0",
        ("problem", "th_a1"): "0.0",
        ("problem", "th_a2"): "0.0",
        ("problem", "ph_a1"): "0.0",
        ("problem", "ph_a2"): "0.0",
        ("problem", "spin_ramp"): "false",
        ("problem", "spin_ramp_timescale"): format_real(orbit_duration),
        ("problem", "spin_ramp_start_time"): format_real(source_state["time"]),
    }
    for (block, key), value in updates.items():
        text = set_param(text, block, key, value)
    return text, tlim


def stage3_spin_text(text, *, root, case, spin_name, spin_mag, theta_deg,
                     alignment, source_restart_root, source_state, expected_start_time,
                     orbit_duration, spin_orbits, restart_dt):
    tlim = expected_start_time + spin_orbits * orbit_duration
    text = apply_common_zoom_updates(
        text, root=root, case=case, stage=f"stage3_{spin_name}_{alignment}",
        source_restart=source_restart_root, source_state=source_state, tlim=tlim,
        restart_dt=restart_dt)
    spin_ramp = spin_mag > 0.0
    updates = {
        ("comment", "zoom_spin_case"): spin_name,
        ("comment", "zoom_spin_alignment"): alignment,
        ("comment", "zoom_expected_stage2_end_time"): format_real(expected_start_time),
        ("job", "restart_file"): str(source_restart_root),
        ("coord", "excise_to_horizon"): "true",
        ("coord", "excise_shrink_to_horizon"): "false",
        ("problem", "a1"): format_real(spin_mag),
        ("problem", "a2"): format_real(spin_mag),
        ("problem", "th_a1"): format_real(theta_deg),
        ("problem", "th_a2"): format_real(theta_deg),
        ("problem", "ph_a1"): "0.0",
        ("problem", "ph_a2"): "0.0",
        ("problem", "spin_ramp"): "true" if spin_ramp else "false",
        ("problem", "spin_ramp_timescale"): format_real(orbit_duration),
        # Launch scripts override this from the actual selected restart time.
        ("problem", "spin_ramp_start_time"): format_real(expected_start_time),
    }
    for (block, key), value in updates.items():
        text = set_param(text, block, key, value)
    return text, tlim, spin_ramp


def apply_overrides(text, overrides):
    for override in overrides:
        try:
            lhs, value = override.split("=", 1)
            block, key = lhs.split("/", 1)
        except ValueError:
            raise SystemExit(
                f"bad --set '{override}', expected block/key=value"
            ) from None
        text = set_param(text, block.strip(), key.strip(), value.strip())
    return text


def setup_zoom_survey(args):
    if args.base_dir.exists() and not args.force:
        raise SystemExit(f"{args.base_dir} already exists; choose a fresh directory")
    if args.base_dir.exists():
        shutil.rmtree(args.base_dir)
    if not args.case_source:
        raise SystemExit("--workflow zoom-survey requires at least one --case-source")
    if not args.exe.exists():
        raise SystemExit(f"missing executable {args.exe}")

    root = args.base_dir.expanduser().resolve()
    scripts_dir = root / "scripts"
    inputs_dir = root / "input"
    cases_dir = root / "cases"
    bin_dir = root / "bin"
    logs_dir = root / "logs"
    for directory in (scripts_dir, inputs_dir, cases_dir, bin_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exe = bin_dir / "athena"
    shutil.copy2(args.exe, exe)
    exe.chmod(0o755)
    helper = scripts_dir / "restart_header_time.py"
    write_restart_time_helper(helper)

    case_sources = [parse_case_source(spec) for spec in args.case_source]
    stage2_runs = []
    stage3_runs = []
    readme_rows = []

    for source in case_sources:
        case = source["case"]
        template = source["template"]
        rank0_restart = source["restart"]
        if not template.exists():
            raise SystemExit(f"missing template parfile {template}")
        if not rank0_restart.exists():
            raise SystemExit(f"missing restart {rank0_restart}")
        check_restart_set(rank0_restart, args.ranks)
        state = read_restart_state(rank0_restart)
        template_text = template.read_text()

        case_root = cases_dir / case
        stage2_run = case_root / "stage2_horizon_zero_spin" / "run"
        stage2_rst = stage2_run / "rst"
        stage2_rank0 = copy_restart_set(rank0_restart, stage2_rst, args.ranks)
        stage2_input = inputs_dir / f"{case}_stage2_horizon_zero_spin.par"
        text, stage2_tlim = stage2_horizon_text(
            template_text, root=root, case=case, source_restart=stage2_rank0,
            source_state=state, orbit_duration=args.orbit_duration,
            restart_dt=args.restart_dt)
        text = apply_overrides(text, args.set)
        stage2_input.write_text(text)
        stage2_launch = stage2_run / "launch.sh"
        write_zoom_launch(
            stage2_launch, label=f"{case} stage2_horizon_zero_spin",
            run_dir=stage2_run, exe=exe, input_file=stage2_input,
            runtime=args.runtime, initial_restart=stage2_rank0, helper=helper)
        stage2_runs.append({
            "name": f"{case}_stage2",
            "run_dir": stage2_run,
            "launch": stage2_launch,
        })
        readme_rows.append(
            f"- {case}: template `{template}`, restart `{rank0_restart}`, "
            f"time={format_real(state['time'])}, cycle={state['cycle']}"
        )

        for spin_name, spin_mag, theta_deg, alignment in SPIN_CASES:
            spin_run = case_root / "stage3_spin_survey" / spin_name / "run"
            spin_run.mkdir(parents=True, exist_ok=True)
            spin_input = inputs_dir / f"{case}_stage3_{spin_name}.par"
            spin_text, _, spin_ramp = stage3_spin_text(
                template_text, root=root, case=case, spin_name=spin_name,
                spin_mag=spin_mag, theta_deg=theta_deg, alignment=alignment,
                source_restart_root=stage2_rst, source_state=state,
                expected_start_time=stage2_tlim,
                orbit_duration=args.orbit_duration, spin_orbits=args.spin_orbits,
                restart_dt=args.restart_dt)
            spin_text = apply_overrides(spin_text, args.set)
            spin_input.write_text(spin_text)
            spin_launch = spin_run / "launch.sh"
            write_zoom_launch(
                spin_launch, label=f"{case} {spin_name}",
                run_dir=spin_run, exe=exe, input_file=spin_input,
                runtime=args.runtime, initial_restart_root=stage2_rst,
                spin_ramp=spin_ramp, helper=helper,
                require_source_advanced=True)
            stage3_runs.append({
                "name": f"{case}_{spin_name}",
                "run_dir": spin_run,
                "launch": spin_launch,
            })

    stage2_pbs = scripts_dir / "submit_stage2_horizon_debug_scaling.pbs"
    write_bundle_pbs(
        stage2_pbs, root=root, runs=stage2_runs, account=args.account,
        queue=args.stage2_queue, job_name=args.stage2_job_name,
        walltime=args.stage2_walltime, runtime=args.runtime, max_iters=1,
        ranks=args.ranks, ranks_per_node=args.ranks_per_node,
        nodes_per_run=args.nodes_per_run, cpu_bind=args.cpu_bind,
        auto_resubmit=False)

    stage3_pbs = scripts_dir / "submit_stage3_spin_survey_prod.pbs"
    write_bundle_pbs(
        stage3_pbs, root=root, runs=stage3_runs, account=args.account,
        queue=args.stage3_queue, job_name=args.stage3_job_name,
        walltime=args.stage3_walltime, runtime=args.runtime,
        max_iters=args.stage3_max_iters, ranks=args.ranks,
        ranks_per_node=args.ranks_per_node, nodes_per_run=args.nodes_per_run,
        cpu_bind=args.cpu_bind, auto_resubmit=args.auto_resubmit)

    readme = root / "README.md"
    readme.write_text(f"""# dynbbh staged zoom workflow

Generated by `{Path(__file__).resolve()}`.

## Sources

{os.linesep.join(readme_rows)}

## Stage order

1. `qsub {stage2_pbs}` runs SANE, MAD, and BONDI together on debug-scaling with
   {args.nodes_per_run} nodes each ({len(stage2_runs) * args.nodes_per_run} total).
   These runs keep both black holes at zero spin and shrink the smooth-excision
   radius from 4M to the local horizon over {format_real(args.orbit_duration)}M.
2. After the stage-2 runs have produced clean horizon restarts, inspect them and then
   `qsub {stage3_pbs}` to run the 12 spin-survey cases on {len(stage3_runs) * args.nodes_per_run}
   prod nodes.  Each spin run first looks for its own latest restart, then falls back
   to the latest stage-2 restart for the same SANE/MAD/BONDI case.

No job was submitted by this setup step.
""")

    print(f"root={root}")
    print(f"exe={exe}")
    print(f"stage2_pbs={stage2_pbs}")
    print(f"stage3_pbs={stage3_pbs}")
    print(f"readme={readme}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=["single", "zoom-survey"], default="single")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--stage", choices=[
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
    parser.add_argument(
        "--case-source", action="append", default=[],
        help="zoom-survey source as CASE=template.par,rank0_restart.rst[,label]",
    )
    parser.add_argument("--orbit-duration", type=float, default=ORBIT_DURATION)
    parser.add_argument("--spin-orbits", type=float, default=1.0)
    parser.add_argument("--restart-dt", type=float, default=50.0)
    parser.add_argument("--nodes-per-run", type=int, default=DEFAULT_NODES_PER_RUN)
    parser.add_argument("--stage2-queue", default="debug-scaling")
    parser.add_argument("--stage2-walltime", default="01:00:00")
    parser.add_argument("--stage2-job-name", default="zoom_horizon")
    parser.add_argument("--stage3-queue", default="prod")
    parser.add_argument("--stage3-walltime", default="24:00:00")
    parser.add_argument("--stage3-job-name", default="zoom_spin")
    parser.add_argument("--stage3-max-iters", type=int, default=26)
    parser.add_argument("--auto-resubmit", action="store_true")
    args = parser.parse_args()

    if args.workflow == "zoom-survey":
        setup_zoom_survey(args)
        return

    if args.case is None or args.stage is None:
        raise SystemExit("--workflow single requires --case and --stage")
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
