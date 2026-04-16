"""
Hydro shearing-wave STS acceptance test.

Extends the existing Hydro shearing-box path with viscosity on STS and verifies that the
first-cycle timestep exceeds the explicit viscous limit.
"""

import re
from pathlib import Path

import pytest

import test_suite.testutils as testutils

INPUT_FILE = Path("inputs/hydro_shwave.athinput")


def insert_block_lines(text: str, block: str, lines: list[str]) -> str:
    marker = f"<{block}>"
    if marker not in text:
        raise RuntimeError(f"Could not find block {block} in temporary input template")
    return text.replace(marker, marker + "\n" + "\n".join(lines), 1)


def write_temp_input(temp_name: str, template: Path, block_lines: dict[str, list[str]]) -> str:
    text = template.read_text()
    for block, lines in block_lines.items():
        text = insert_block_lines(text, block, lines)
    path = Path(temp_name)
    path.write_text(text)
    return str(path)


def parse_first_cycle_dt(output: str) -> float:
    match = re.search(r"\bdt\s*=\s*([0-9.eE+-]+)", output)
    if match is None:
        raise RuntimeError("Could not find the first-cycle dt in AthenaK output")
    return float(match.group(1))


def test_hydro_shwave_sts_mpicpu():
    temp_file = None
    try:
        temp_file = write_temp_input(
            "hydro_shwave_sts_tmp.athinput",
            INPUT_FILE,
            {
                "time": ["sts_integrator = rkl2", "sts_max_dt_ratio = -1.0"],
                "hydro": ["viscosity = 0.1", "viscosity_integrator = sts"],
            },
        )
        output = testutils.mpi_run_capture(
            temp_file,
            [
                "job/basename=shwave_sts",
                "mesh/nx1=64",
                "meshblock/nx1=32",
                "mesh/nx2=64",
                "meshblock/nx2=32",
                "mesh/nx3=4",
                "meshblock/nx3=4",
                "time/nlim=1",
            ],
            threads=4,
        )
        first_dt = parse_first_cycle_dt(output)
        dx1 = 0.5 / 64.0
        explicit_viscous_dt = 0.3 * 0.5 * dx1 * dx1 / 0.1
        if first_dt <= explicit_viscous_dt:
            pytest.fail(
                f"Hydro shwave STS did not exceed the explicit viscous limit: "
                f"{first_dt:g} <= {explicit_viscous_dt:g}"
            )
    finally:
        if temp_file is not None:
            Path(temp_file).unlink(missing_ok=True)
        testutils.cleanup()
