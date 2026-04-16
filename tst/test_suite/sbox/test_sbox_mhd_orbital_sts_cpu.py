"""
MHD orbital-advection STS smoke test.

Uses the tracked orbital-advection example as a mesh/shearing-box template, swaps in the
built-in resistive diffusion pgen, and verifies that the MHD orbital-remap path now runs
inside the STS sweep.
"""

from pathlib import Path

import test_suite.testutils as testutils

INPUT_FILE = Path("../../../inputs/shearing_box/mhd_orb_adv.athinput")


def insert_block_lines(text: str, block: str, lines: list[str]) -> str:
    marker = f"<{block}>"
    if marker not in text:
        raise RuntimeError(f"Could not find block {block} in temporary input template")
    return text.replace(marker, marker + "\n" + "\n".join(lines), 1)


def write_temp_input(temp_name: str, template: Path, block_lines: dict[str, list[str]]) -> str:
    text = template.read_text()
    text = text.replace("eos          = ideal", "eos          = isothermal", 1)
    for block, lines in block_lines.items():
        text = insert_block_lines(text, block, lines)
    path = Path(temp_name)
    path.write_text(text)
    return str(path)


def test_mhd_orbital_sts_cpu():
    temp_file = None
    try:
        temp_file = write_temp_input(
            "mhd_orbital_sts_tmp.athinput",
            INPUT_FILE,
            {
                "time": ["sts_integrator = rkl2", "sts_max_dt_ratio = -1.0"],
                "mhd": [
                    "ohmic_resistivity = 0.1",
                    "ohmic_resistivity_integrator = sts",
                    "iso_sound_speed = 1.0",
                ],
                "problem": [
                    "pgen_name = diffusion",
                    "diffusion_test = mhd_resistivity",
                ],
            },
        )
        result = testutils.run(
            temp_file,
            [
                "job/basename=mhd_orbital_sts",
                "mesh/nx1=64",
                "meshblock/nx1=64",
                "mesh/nx2=64",
                "meshblock/nx2=64",
                "mesh/nx3=4",
                "meshblock/nx3=4",
                "time/nlim=1",
            ],
        )
        assert result, "MHD orbital-advection STS smoke failed."
    finally:
        if temp_file is not None:
            Path(temp_file).unlink(missing_ok=True)
        testutils.cleanup()
