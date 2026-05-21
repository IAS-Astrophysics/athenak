#!/usr/bin/env python3
"""Run and parse 2nd-order CTS smoother omega comparisons."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build-critical-kerr-mpi" / "src" / "athena"
INPUT = ROOT / "inputs" / "z4c" / "gw_collapse" / "z4c_critical_kerr_formation.athinput"
OUTDIR = ROOT / "analysis" / "critical_kerr_convergence" / "smoother_comparison"
LOGDIR = OUTDIR / "logs"

ITER_RE = re.compile(r"MG iteration\s+(\d+): defect = ([0-9.eE+-]+)")
INITIAL_RE = re.compile(r"MG initial defect = ([0-9.eE+-]+)")
FINAL_RE = re.compile(r"IDSolve final CTS defect = [0-9.eE+-]+ \(total = ([0-9.eE+-]+)\)")
STATS_RE = re.compile(
    r"CTS smoother stats.*accepted=([0-9.eE+-]+), limited=([0-9.eE+-]+), "
    r"backtracked=([0-9.eE+-]+), fallback=([0-9.eE+-]+), "
    r"singular=([0-9.eE+-]+), nonfinite=([0-9.eE+-]+), "
    r"rejected=([0-9.eE+-]+), psi_floor=([0-9.eE+-]+), "
    r"max_update=([0-9.eE+-]+)"
)


def fmt_omega(omega: float) -> str:
    text = f"{omega:.6g}".replace(".", "p").replace("-", "m")
    return text


def base_overrides(
    n: int,
    meshblock: int,
    smoother: str,
    omega: float,
    niter: int,
    basename: str,
    spatial_order: int,
    nghost: int,
    mg_bc: str,
    smoother_update: str,
    mg_coarse_fd_stencil: int,
) -> list[str]:
    return [
        f"job/basename=analysis/critical_kerr_convergence/smoother_comparison/{basename}",
        f"mesh/nghost={nghost}",
        f"mesh/nx1={n}",
        f"mesh/nx2={n}",
        f"mesh/nx3={n}",
        "mesh/x1min=-16.0",
        "mesh/x1max=16.0",
        "mesh/x2min=-16.0",
        "mesh/x2max=16.0",
        "mesh/x3min=-16.0",
        "mesh/x3max=16.0",
        f"meshblock/nx1={meshblock}",
        f"meshblock/nx2={meshblock}",
        f"meshblock/nx3={meshblock}",
        f"z4c/spatial_order={spatial_order}",
        "id_solve/stop_after_solve=true",
        f"id_solve/smoother={smoother}",
        f"id_solve/omega={omega:.16g}",
        f"id_solve/niteration={niter}",
        f"id_solve/mg_nghost={nghost}",
        f"id_solve/mg_bc={mg_bc}",
        f"id_solve/smoother_update={smoother_update}",
        f"id_solve/mg_coarse_fd_stencil={mg_coarse_fd_stencil}",
        "id_solve/npresmooth=2",
        "id_solve/npostsmooth=2",
        "id_solve/reject_worse=false",
        "id_solve/keep_best_solution=false",
        "id_solve/stop_on_defect_increase=false",
        "id_solve/show_defect=true",
        "id_solve/show_smoother_stats=true",
        "id_solve/ngs_line_search_steps=8",
        "id_solve/ngs_line_search_min=1.0e-4",
        "id_solve/smoother_max_update_fraction=0.25",
        "problem/initial_lapse=1.0",
        "problem/amplitude=1.0e-3",
        "problem/support_radius=4.0",
    ]


def parse_log(path: Path) -> dict[str, object]:
    initial = math.nan
    final = math.nan
    by_iter: dict[int, float] = {}
    stats = {
        "accepted": 0.0,
        "limited": 0.0,
        "backtracked": 0.0,
        "fallback": 0.0,
        "singular": 0.0,
        "nonfinite": 0.0,
        "rejected": 0.0,
        "psi_floor": 0.0,
        "max_update": 0.0,
    }
    for line in path.read_text(errors="replace").splitlines():
        if math.isnan(initial):
            match = INITIAL_RE.search(line)
            if match:
                initial = float(match.group(1))
        match = ITER_RE.search(line)
        if match:
            iteration = int(match.group(1))
            by_iter.setdefault(iteration, float(match.group(2)))
        match = FINAL_RE.search(line)
        if match and math.isnan(final):
            final = float(match.group(1))
        match = STATS_RE.search(line)
        if match:
            keys = [
                "accepted",
                "limited",
                "backtracked",
                "fallback",
                "singular",
                "nonfinite",
                "rejected",
                "psi_floor",
                "max_update",
            ]
            values = [float(value) for value in match.groups()]
            for key, value in zip(keys[:-1], values[:-1]):
                stats[key] += value
            stats["max_update"] = max(stats["max_update"], values[-1])

    history = [{"iteration": i, "defect": by_iter[i]} for i in sorted(by_iter)]
    values = [row["defect"] for row in history]
    all_values = ([initial] if math.isfinite(initial) else []) + values
    if math.isnan(final) and values:
        final = values[-1]
    max_defect = max(all_values) if all_values else math.nan
    min_defect = min(all_values) if all_values else math.nan
    finite = all(math.isfinite(v) for v in all_values + ([final] if math.isfinite(final) else []))
    final_over_initial = final / initial if math.isfinite(final) and initial > 0.0 else math.nan
    min_over_initial = min_defect / initial if math.isfinite(min_defect) and initial > 0.0 else math.nan
    max_over_initial = max_defect / initial if math.isfinite(max_defect) and initial > 0.0 else math.nan
    convergent = finite and math.isfinite(final_over_initial) and final_over_initial < 1.0
    bounded = finite and math.isfinite(max_over_initial) and max_over_initial < 10.0
    return {
        "initial": initial,
        "final": final,
        "min": min_defect,
        "max": max_defect,
        "final_over_initial": final_over_initial,
        "min_over_initial": min_over_initial,
        "max_over_initial": max_over_initial,
        "convergent": convergent,
        "bounded": bounded,
        "history": history,
        "stats": stats,
    }


def run_case(
    n: int,
    meshblock: int,
    smoother: str,
    omega: float,
    niter: int,
    tag: str,
    spatial_order: int,
    nghost: int,
    mg_bc: str,
    smoother_update: str,
    mg_coarse_fd_stencil: int,
) -> dict[str, object]:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    basename = (
        f"{tag}_{smoother}_omega{fmt_omega(omega)}_n{n}_"
        f"ord{spatial_order}_ng{nghost}_{mg_bc}_{smoother_update}_"
        f"cfd{mg_coarse_fd_stencil}_iter{niter}"
    )
    log_path = LOGDIR / f"{basename}.log"
    cmd = [
        "mpirun",
        "-np",
        "4",
        str(EXE),
        "-i",
        str(INPUT),
        *base_overrides(n, meshblock, smoother, omega, niter, basename,
                        spatial_order, nghost, mg_bc, smoother_update,
                        mg_coarse_fd_stencil),
    ]
    print(f"RUN {basename}", flush=True)
    start = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - start
    parsed = parse_log(log_path)
    row = {
        "tag": tag,
        "smoother": smoother,
        "omega": omega,
        "N": n,
        "meshblock": meshblock,
        "niteration": niter,
        "spatial_order": spatial_order,
        "nghost": nghost,
        "mg_bc": mg_bc,
        "smoother_update": smoother_update,
        "mg_coarse_fd_stencil": mg_coarse_fd_stencil,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "log": str(log_path.relative_to(ROOT)),
        **{k: v for k, v in parsed.items() if k != "history"},
    }
    row.update({f"stats_{key}": value for key, value in parsed["stats"].items()})
    row["stable"] = proc.returncode == 0 and bool(row["convergent"]) and bool(row["bounded"])
    print(
        f"  rc={proc.returncode} final/initial={row['final_over_initial']:.6g} "
        f"max/initial={row['max_over_initial']:.6g} stable={row['stable']}",
        flush=True,
    )
    return row


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fields = [
        "tag",
        "smoother",
        "omega",
        "N",
        "meshblock",
        "spatial_order",
        "nghost",
        "mg_bc",
        "smoother_update",
        "mg_coarse_fd_stencil",
        "niteration",
        "returncode",
        "wall_seconds",
        "stable",
        "convergent",
        "bounded",
        "initial",
        "final",
        "min",
        "max",
        "final_over_initial",
        "min_over_initial",
        "max_over_initial",
        "stats_accepted",
        "stats_limited",
        "stats_backtracked",
        "stats_fallback",
        "stats_singular",
        "stats_nonfinite",
        "stats_rejected",
        "stats_psi_floor",
        "stats_max_update",
        "log",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_history(case_rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "smoother",
        "omega",
        "N",
        "meshblock",
        "spatial_order",
        "nghost",
        "mg_bc",
        "smoother_update",
        "mg_coarse_fd_stencil",
        "niteration",
        "iteration",
        "defect",
        "defect_over_initial",
        "log",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in case_rows:
            parsed = parse_log(ROOT / str(row["log"]))
            initial = float(parsed["initial"])
            for entry in parsed["history"]:
                defect = float(entry["defect"])
                writer.writerow(
                    {
                        "smoother": row["smoother"],
                        "omega": row["omega"],
                        "N": row["N"],
                        "meshblock": row["meshblock"],
                        "spatial_order": row["spatial_order"],
                        "nghost": row["nghost"],
                        "mg_bc": row["mg_bc"],
                        "smoother_update": row.get("smoother_update", ""),
                        "mg_coarse_fd_stencil": row.get("mg_coarse_fd_stencil", ""),
                        "niteration": row["niteration"],
                        "iteration": entry["iteration"],
                        "defect": defect,
                        "defect_over_initial": defect / initial if initial > 0.0 else math.nan,
                        "log": row["log"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--meshblock", type=int, required=True)
    parser.add_argument("--niter", type=int, required=True)
    parser.add_argument("--spatial-order", type=int, default=2)
    parser.add_argument("--nghost", type=int, default=2)
    parser.add_argument("--mg-bc", choices=["zerograd", "zerofixed"], default="zerograd")
    parser.add_argument("--smoother-update", choices=["frozen_view"], default="frozen_view")
    parser.add_argument("--mg-coarse-fd-stencil", type=int, default=2)
    parser.add_argument("--smoother", choices=["diagonal", "newton_gs"], action="append", required=True)
    parser.add_argument("--omega", type=float, action="append", required=True)
    parser.add_argument("--summary", default="smoother_omega_sweep_summary.csv")
    parser.add_argument("--history", default="")
    parser.add_argument("--stop-after-unstable", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for smoother in args.smoother:
        for omega in args.omega:
            row = run_case(
                args.n, args.meshblock, smoother, omega, args.niter, args.tag,
                args.spatial_order, args.nghost, args.mg_bc,
                args.smoother_update, args.mg_coarse_fd_stencil,
            )
            rows.append(row)
            if args.stop_after_unstable and not bool(row["stable"]):
                print(f"STOP {smoother}: first unstable omega={omega:.6g}", flush=True)
                break

    summary_path = OUTDIR / args.summary
    existing: list[dict[str, object]] = []
    if summary_path.exists():
        with summary_path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    write_summary(existing + rows, summary_path)
    print(f"WROTE {summary_path.relative_to(ROOT)}", flush=True)
    if args.history:
        history_path = OUTDIR / args.history
        write_history(rows, history_path)
        print(f"WROTE {history_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
