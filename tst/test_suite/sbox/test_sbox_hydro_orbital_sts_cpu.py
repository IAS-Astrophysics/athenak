"""
Hydro orbital-advection STS smoke test.

Uses the tracked orbital-advection example as a mesh/shearing-box template, swaps in the
built-in Hydro viscosity diffusion pgen, and verifies that the Hydro orbital-remap path
now runs inside the STS sweep.
"""

from pathlib import Path

import test_suite.testutils as testutils

INPUT_FILE = Path("../../../inputs/shearing_box/hydro_orb_adv.athinput")


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


def test_hydro_orbital_sts_cpu():
    temp_file = None
    try:
        temp_file = write_temp_input(
            "hydro_orbital_sts_tmp.athinput",
            INPUT_FILE,
            {
                "time": ["sts_integrator = rkl2", "sts_max_dt_ratio = -1.0"],
                "hydro": ["viscosity = 0.1", "viscosity_integrator = sts"],
                "problem": [
                    "pgen_name = diffusion",
                    "diffusion_test = hydro_viscosity",
                ],
            },
        )
        result = testutils.run(
            temp_file,
            [
                "job/basename=orbital_sts",
                "mesh/nx1=64",
                "meshblock/nx1=64",
                "mesh/nx2=64",
                "meshblock/nx2=64",
                "mesh/nx3=4",
                "meshblock/nx3=4",
                "time/nlim=1",
            ],
        )
        assert result, "Hydro orbital-advection STS smoke failed."
    finally:
        if temp_file is not None:
            Path(temp_file).unlink(missing_ok=True)
        testutils.cleanup()
