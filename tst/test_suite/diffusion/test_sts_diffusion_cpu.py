"""
STS diffusion regression coverage for CPU builds.

- exact Hydro viscosity and MHD resistivity convergence, plus Hydro conduction exactness
  with an explicit-reference control
- mixed explicit/STS MHD timestep-budget coverage
- multilevel Hydro STS smoke
- runtime-fence regression checks
"""

import re
from pathlib import Path

import pytest

import athena_read
import test_suite.testutils as testutils

ROOT_TEST_INPUTS = Path("../../../inputs/tests")
TST_INPUTS = Path("../../../tst/inputs")
L1_RMS_INDEX = 4

EXACT_CASES = {
    "hydro_viscosity": {
        "resolutions": [64, 128],
        "thresholds": (3.5e-11, 0.60),
        "extra_flags": [],
    },
    "hydro_conduction": {
        "resolutions": [128, 256],
        "thresholds": (3.0e-12, 0.30),
        "extra_flags": ["time/sts_max_dt_ratio=4"],
    },
    "mhd_resistivity": {
        "resolutions": [64, 128],
        "thresholds": (1.2e-11, 0.32),
        "extra_flags": ["time/sts_max_dt_ratio=8"],
    },
}

EXPLICIT_CONDUCTION_CASE = {
    "resolutions": [128, 256],
    "thresholds": (4.0e-12, 0.30),
}


def repo_input(name: str) -> str:
    return str(ROOT_TEST_INPUTS / name)


def tst_input(name: str) -> Path:
    return TST_INPUTS / name


def exact_flags(
    basename: str,
    resolution: int,
    integrator_flag: str,
    process_selector: str,
    extra_flags: list[str] | None = None,
) -> list[str]:
    return [
        f"job/basename={basename}",
        f"mesh/nx1={resolution}",
        "mesh/nx2=1",
        "mesh/nx3=1",
        f"meshblock/nx1={resolution // 4}",
        "meshblock/nx2=1",
        "meshblock/nx3=1",
        integrator_flag,
        process_selector,
    ] + ([] if extra_flags is None else extra_flags)


def assert_exact_convergence(
    input_file: str,
    basename: str,
    integrator_flag: str,
    process_selector: str,
    resolutions: list[int],
    thresholds: tuple[float, float],
    extra_flags: list[str] | None = None,
) -> None:
    try:
        for resolution in resolutions:
            result = testutils.run(
                input_file,
                exact_flags(
                    basename,
                    resolution,
                    integrator_flag,
                    process_selector,
                    extra_flags=extra_flags,
                ),
            )
            assert result, f"Exact diffusion run failed for {input_file} at resolution {resolution}."

        max_error, max_ratio = thresholds
        data = athena_read.error_dat(f"{basename}-errs.dat")
        low_res_error = data[0][L1_RMS_INDEX]
        high_res_error = data[1][L1_RMS_INDEX]
        if high_res_error > max_error:
            pytest.fail(
                f"{basename} high-resolution RMS error too large: "
                f"{high_res_error:g} > {max_error:g}"
            )
        if (high_res_error / low_res_error) > max_ratio:
            pytest.fail(
                f"{basename} convergence too slow: "
                f"{(high_res_error/low_res_error):g} > {max_ratio:g}"
            )
    finally:
        testutils.cleanup()


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


def test_hydro_viscosity_exact_sts_cpu():
    case = EXACT_CASES["hydro_viscosity"]
    assert_exact_convergence(
        repo_input("viscosity.athinput"),
        "sts_visc_exact",
        "time/sts_integrator=rkl2",
        "hydro/viscosity_integrator=sts",
        case["resolutions"],
        case["thresholds"],
        extra_flags=case["extra_flags"],
    )


def test_hydro_conduction_exact_sts_cpu():
    case = EXACT_CASES["hydro_conduction"]
    assert_exact_convergence(
        repo_input("sts_conduction.athinput"),
        "sts_cond_exact",
        "time/sts_integrator=rkl2",
        "hydro/conductivity_integrator=sts",
        case["resolutions"],
        case["thresholds"],
        extra_flags=case["extra_flags"],
    )


def test_hydro_conduction_explicit_control_cpu():
    assert_exact_convergence(
        repo_input("sts_conduction.athinput"),
        "sts_cond_explicit",
        "time/sts_integrator=none",
        "hydro/conductivity_integrator=explicit",
        EXPLICIT_CONDUCTION_CASE["resolutions"],
        EXPLICIT_CONDUCTION_CASE["thresholds"],
    )


def test_mhd_resistivity_exact_sts_cpu():
    case = EXACT_CASES["mhd_resistivity"]
    assert_exact_convergence(
        repo_input("sts_resistivity.athinput"),
        "sts_resist_exact",
        "time/sts_integrator=rkl2",
        "mhd/ohmic_resistivity_integrator=sts",
        case["resolutions"],
        case["thresholds"],
        extra_flags=case["extra_flags"],
    )


def test_mhd_mixed_modes_cpu():
    dx1 = 10.0 / 128.0
    expected_dt = 0.4 * 0.5 * dx1 * dx1 / 0.5
    explicit_resistive_dt = 0.4 * 0.5 * dx1 * dx1 / 1.0
    try:
        output = testutils.run_capture(
            repo_input("sts_mhd_mixed_modes.athinput"),
            [
                "job/basename=sts_mixed_modes",
                "time/sts_integrator=rkl2",
                "mhd/viscosity_integrator=explicit",
                "mhd/ohmic_resistivity_integrator=sts",
                "time/nlim=1",
            ],
        )
        first_dt = parse_first_cycle_dt(output)
        if abs(first_dt - expected_dt) > 1.0e-6:
            pytest.fail(
                f"Mixed-mode MHD first-cycle dt mismatch: {first_dt:g} != {expected_dt:g}"
            )
        if first_dt <= explicit_resistive_dt:
            pytest.fail(
                f"Mixed-mode MHD dt is still resistive-limited: {first_dt:g} <= "
                f"{explicit_resistive_dt:g}"
            )
    finally:
        testutils.cleanup()


def test_hydro_viscosity_sts_smr_cpu():
    try:
        output = testutils.run_capture(
            repo_input("sts_viscosity_smr.athinput"),
            ["job/basename=sts_visc_smr"],
        )
        if "physical levels of refinement" not in output:
            pytest.fail("Static-refinement STS smoke did not report refinement levels")
        match = re.search(r"physical levels of refinement\s*=\s*(\d+)", output)
        if match is None or int(match.group(1)) <= 0:
            pytest.fail("Static-refinement STS smoke did not initialize a refined level")
    finally:
        testutils.cleanup()


def test_runtime_fence_global_none_cpu():
    try:
        output = testutils.run_expect_failure(
            repo_input("viscosity.athinput"),
            [
                "job/basename=sts_fail_none",
                "time/sts_integrator=none",
                "hydro/viscosity_integrator=sts",
                "time/nlim=1",
            ],
        )
        assert "<time>/sts_integrator = none" in output
    finally:
        testutils.cleanup()


def test_runtime_fence_no_active_sts_process_cpu():
    try:
        output = testutils.run_expect_failure(
            repo_input("viscosity.athinput"),
            [
                "job/basename=sts_fail_no_process",
                "time/sts_integrator=rkl2",
                "hydro/viscosity_integrator=explicit",
                "time/nlim=1",
            ],
        )
        assert "requires at least one active parabolic process" in output
    finally:
        testutils.cleanup()


def test_runtime_fence_ion_neutral_cpu():
    temp_file = None
    try:
        temp_file = write_temp_input(
            "sts_ion_neutral_tmp.athinput",
            tst_input("cshock.athinput"),
            {
                "time": ["sts_integrator = rkl2"],
                "mhd": [
                    "ohmic_resistivity = 1.0e-3",
                    "ohmic_resistivity_integrator = sts",
                ],
            },
        )
        output = testutils.run_expect_failure(
            temp_file,
            ["job/basename=sts_ion_neutral", "time/nlim=1"],
        )
        assert "STS is not yet supported for <ion-neutral> runs" in output
    finally:
        if temp_file is not None:
            Path(temp_file).unlink(missing_ok=True)
        testutils.cleanup()
