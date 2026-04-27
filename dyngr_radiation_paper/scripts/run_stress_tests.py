#!/usr/bin/env python3
"""Run compact stress tests for passive ADM dyn_radiation."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


FINITE_TOKEN = re.compile(r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", re.IGNORECASE)
DT_TOKEN = re.compile(
    r"\bdt=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass(frozen=True)
class Case:
    name: str
    command: list[str]
    expect_success: bool = True
    expect_text: str | None = None
    min_dt: float | None = None
    check_outputs: bool = True
    tags: set[str] = field(default_factory=set)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def text_has_nonfinite(text: str) -> bool:
    return FINITE_TOKEN.search(text) is not None


def scan_text_outputs(run_dir: Path) -> list[str]:
    bad: list[str] = []
    suffixes = {".tab", ".dat", ".hst", ".trk"}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if text_has_nonfinite(text):
            bad.append(str(path))
    return bad


def parse_min_dt(stdout: str) -> float | None:
    values = []
    for match in DT_TOKEN.finditer(stdout):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return min(values) if values else None


def write_log(log_dir: Path, case: Case, result: subprocess.CompletedProcess[str]) -> None:
    safe_name = case.name.replace("/", "_")
    (log_dir / f"{safe_name}.out").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{safe_name}.err").write_text(result.stderr, encoding="utf-8")


def run_case(case: Case, root: Path, run_dir: Path, keep_going: bool) -> bool:
    print(f"[RUN] {case.name}", flush=True)
    result = subprocess.run(case.command, cwd=root, text=True, capture_output=True)
    write_log(run_dir / "logs", case, result)

    failures: list[str] = []
    if case.expect_success and result.returncode != 0:
        failures.append(f"exit code {result.returncode}")
    if not case.expect_success and result.returncode == 0:
        failures.append("expected failure but command succeeded")
    if case.expect_text is not None:
        combined = result.stdout + result.stderr
        if case.expect_text not in combined:
            failures.append(f"missing expected text: {case.expect_text!r}")
    if case.expect_success and text_has_nonfinite(result.stdout + result.stderr):
        failures.append("stdout/stderr contains NaN or Inf token")
    if case.expect_success and case.check_outputs:
        bad_outputs = scan_text_outputs(run_dir)
        if bad_outputs:
            failures.append("non-finite token in " + ", ".join(bad_outputs))
    if case.expect_success and case.min_dt is not None:
        observed = parse_min_dt(result.stdout)
        if observed is None:
            failures.append("could not parse timestep from stdout")
        elif observed < case.min_dt:
            failures.append(f"minimum dt {observed:.6e} below threshold {case.min_dt:.6e}")

    if failures:
        print(f"[FAIL] {case.name}: {'; '.join(failures)}", flush=True)
        if not keep_going:
            print(f"       logs: {relative(run_dir / 'logs', root)}", flush=True)
            sys.exit(1)
        return False

    print(f"[PASS] {case.name}", flush=True)
    return True


def input_path(name: str) -> str:
    return str(repo_root() / name)


def athena_case(
    name: str,
    exe: Path,
    input_file: str,
    run_dir: Path,
    *overrides: str,
    tags: set[str] | None = None,
    **kwargs: object,
) -> Case:
    command = [
        str(exe),
        "-i",
        input_path(input_file),
        "-d",
        str(run_dir),
        "job/basename=" + name,
        *overrides,
    ]
    return Case(name=name, command=command, tags=tags or set(), **kwargs)


def mpi_case(
    name: str,
    exe: Path,
    input_file: str,
    run_dir: Path,
    *overrides: str,
    tags: set[str] | None = None,
    **kwargs: object,
) -> Case:
    command = [
        "mpirun",
        "-n",
        "2",
        str(exe),
        "-i",
        input_path(input_file),
        "-d",
        str(run_dir),
        "job/basename=" + name,
        *overrides,
    ]
    return Case(name=name, command=command, tags=(tags or set()) | {"mpi"}, **kwargs)


def build_cases(root: Path, run_dir: Path) -> list[Case]:
    athena = root / "build" / "src" / "athena"
    dynbbh = root / "build_dynbbh_rad" / "src" / "athena"

    restart_base = "stress_restart_src"
    restart_file = run_dir / "rst" / f"{restart_base}.00001.rst"
    dynbbh_input = "dyngr_radiation_paper/inputs/dynbbh_beam_particles.athinput"
    legacy_dynbbh = run_dir / "dynbbh_legacy_radiation.athinput"
    legacy_dynbbh.write_text(
        (root / dynbbh_input).read_text(encoding="utf-8").replace(
            "<dyn_radiation>", "<radiation>", 1
        ),
        encoding="utf-8",
    )

    cases = [
        athena_case("stress_tetrad_cks", athena, "inputs/tests/dynrad_tetrad_cks.athinput",
                    run_dir, "time/nlim=0"),
        athena_case("stress_tetrad_adm", athena, "inputs/tests/dynrad_tetrad_adm.athinput",
                    run_dir, "time/nlim=0"),
        athena_case("stress_beam_cks", athena, "inputs/tests/dynrad_beam_cks.athinput",
                    run_dir),
        athena_case("stress_beam_adm_flat", athena,
                    "inputs/tests/dynrad_beam_adm_flat.athinput", run_dir),
        athena_case("stress_bh_beam_adm", athena,
                    "inputs/tests/dynrad_bh_beam_adm.athinput", run_dir,
                    min_dt=1.0e-4),
        athena_case("stress_z4c_wave_adm", athena,
                    "inputs/tests/dynrad_z4c_wave_adm.athinput", run_dir),
        athena_case("stress_lwave", athena, "inputs/tests/dynrad_lwave.athinput",
                    run_dir),
        athena_case("stress_lwave_smr", athena,
                    "inputs/tests/dynrad_lwave_smr.athinput", run_dir),
        mpi_case("stress_mpi_beam_cks", athena,
                 "inputs/tests/dynrad_beam_cks.athinput", run_dir),
        mpi_case("stress_mpi_beam_adm_flat", athena,
                 "inputs/tests/dynrad_beam_adm_flat.athinput", run_dir),
        mpi_case("stress_mpi_lwave_smr", athena,
                 "inputs/tests/dynrad_lwave_smr.athinput", run_dir),
        mpi_case("stress_mpi_z4c_wave_adm", athena,
                 "inputs/tests/dynrad_z4c_wave_adm.athinput", run_dir,
                 "meshblock/nx1=4", "meshblock/nx2=8", "meshblock/nx3=8"),
        mpi_case("stress_mpi_bh_beam_adm", athena,
                 "inputs/tests/dynrad_bh_beam_adm.athinput", run_dir,
                 "time/nlim=3", min_dt=1.0e-4),
        athena_case(restart_base, athena, "inputs/tests/dynrad_beam_cks.athinput",
                    run_dir, "time/tlim=0.1", "output1/file_type=rst", "output1/dt=0.1",
                    "output1/variable=rad_coord", tags={"restart"}),
        Case(
            name="stress_restart_resume",
            command=[
                str(athena),
                "-r",
                str(restart_file),
                "-d",
                str(run_dir),
                "job/basename=stress_restart_resume",
                "time/tlim=0.12",
                "time/nlim=2",
            ],
            tags={"restart"},
        ),
        athena_case("stress_dynbbh", dynbbh, dynbbh_input, run_dir, tags={"dynbbh"}),
        athena_case("stress_dynbbh_coupling", dynbbh, dynbbh_input,
                    run_dir, "dyn_radiation/rad_source=true", "time/nlim=2",
                    tags={"dynbbh"}),
        mpi_case("stress_mpi_dynbbh", dynbbh,
                 dynbbh_input, run_dir, "time/nlim=1", tags={"dynbbh"}),
        athena_case("stress_z4c_cks_rejected", athena,
                    "inputs/tests/dynrad_z4c_wave_adm.athinput", run_dir,
                    "dyn_radiation/geometry=cks", "time/nlim=0",
                    expect_success=False,
                    expect_text="requires geometry='adm'",
                    check_outputs=False),
        Case(
            name="stress_dynbbh_prad_rejected",
            command=[
                str(dynbbh),
                "-i",
                str(legacy_dynbbh),
                "-d",
                str(run_dir),
                "job/basename=stress_dynbbh_prad_rejected",
                "time/nlim=0",
            ],
            expect_success=False,
            expect_text="Legacy <radiation> cannot be combined with ADM/Z4c",
            check_outputs=False,
            tags={"dynbbh"},
        ),
    ]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="output directory, default /tmp/dynrad_stress_<timestamp>")
    parser.add_argument("--skip-mpi", action="store_true", help="skip two-rank MPI cases")
    parser.add_argument("--skip-dynbbh", action="store_true", help="skip dynbbh cases")
    parser.add_argument("--skip-restart", action="store_true", help="skip restart cases")
    parser.add_argument("--keep-going", action="store_true",
                        help="run remaining cases after a failure")
    args = parser.parse_args()

    root = repo_root()
    run_dir = args.run_dir or Path(f"/tmp/dynrad_stress_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    missing = [
        path for path in (root / "build" / "src" / "athena",
                          root / "build_dynbbh_rad" / "src" / "athena")
        if not path.exists()
    ]
    if missing:
        for path in missing:
            print(f"missing executable: {path}", file=sys.stderr)
        return 2
    if not shutil.which("mpirun") and not args.skip_mpi:
        print("mpirun not found; rerun with --skip-mpi or load MPI", file=sys.stderr)
        return 2

    cases = build_cases(root, run_dir)
    if args.skip_mpi:
        cases = [case for case in cases if "mpi" not in case.tags]
    if args.skip_dynbbh:
        cases = [case for case in cases if "dynbbh" not in case.tags]
    if args.skip_restart:
        cases = [case for case in cases if "restart" not in case.tags]

    print(f"stress run directory: {run_dir}", flush=True)
    passed = 0
    for case in cases:
        if run_case(case, root, run_dir, args.keep_going):
            passed += 1

    failed = len(cases) - passed
    print(f"stress summary: {passed} passed, {failed} failed", flush=True)
    print(f"logs: {run_dir / 'logs'}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
