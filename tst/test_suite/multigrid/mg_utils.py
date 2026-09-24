"""
Shared utilities for multigrid test suite.

Provides helpers to run AthenaK with stdout capture and parse multigrid-specific
output (defect norms, binary gravity errors, Jeans wave errors).
"""

import os
import re
import math
import logging
from subprocess import Popen, PIPE
from typing import List, Dict, Optional
from test_suite.testutils import cleanup

import pytest

LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "test_log.txt")
)


def run_athenak(inputfile, flags=None, mpi=False, threads=1):
    """Run the AthenaK binary and return its success flag together with stdout.

    The multigrid tests parse solver diagnostics (defect norms, Jeans growth
    rates, per-iteration output) that AthenaK prints to stdout. The shared
    ``testutils.run``/``testutils.mpi_run`` helpers only return a success bool,
    so this local runner captures and returns the output as well.

    Args:
        inputfile (str): Path to the AthenaK input file.
        flags (list): Additional command-line flags for the AthenaK binary.
        mpi (bool): Whether to launch the run under mpirun.
        threads (int): Number of MPI ranks (only used when mpi=True).

    Returns:
        list: [success (bool), stdout (str)].
    """
    if flags is None:
        flags = []
    if mpi:
        command = ["mpirun", "-np", str(threads), "./athena", "-i", inputfile] + flags
    else:
        command = ["./athena", "-i", inputfile] + flags
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
    output, _errors = process.communicate()
    if process.returncode != 0:
        logging.error(f"Command failed with return code {process.returncode}")
    return [process.returncode == 0, output]


def parse_mg_defects(stdout: str) -> List[float]:
    """Parse MG defect values from stdout.

    Matches lines like:
      MG initial defect = 1.234e-01
      MG iteration 0: defect = 5.678e-05
      MG iteration N: defect = ...
    Returns flat list of defect values in order (initial first, then iterations).
    """
    pattern = re.compile(
        r"(?:MG\s+initial\s+defect|MG\s+iteration\s+\d+:\s+defect)\s*=\s*([0-9.eE+\-]+)")
    defects = []
    for match in pattern.finditer(stdout):
        defects.append(float(match.group(1)))
    return defects


def parse_mg_defects_per_solve(stdout: str) -> List[List[float]]:
    """Parse MG defect values grouped by solve invocation.

    Each group starts with "MG initial defect" and contains subsequent
    "MG iteration" values.  Returns a list of lists, one per solve.
    """
    pat_init = re.compile(r"MG\s+initial\s+defect\s*=\s*([0-9.eE+\-]+)")
    pat_iter = re.compile(r"MG\s+iteration\s+\d+:\s+defect\s*=\s*([0-9.eE+\-]+)")
    solves: List[List[float]] = []
    for line in stdout.splitlines():
        m = pat_init.search(line)
        if m:
            solves.append([float(m.group(1))])
            continue
        m = pat_iter.search(line)
        if m and solves:
            solves[-1].append(float(m.group(1)))
    return solves


def parse_binary_gravity_errors(stdout: str) -> Dict[str, float]:
    """Parse BinaryGravityErrors output from stdout.

    Matches lines like:
      Potential    L2       : 1.234e-02
      Acceleration L2       : 5.678e-03
      Max Potential Error    : 9.012e-02
      Max Acceleration Error : 3.456e-02
    Returns dict with keys: pot_l2, acc_l2, pot_max, acc_max.
    """
    result = {}
    patterns = {
        "pot_l2": re.compile(r"Potential\s+L2\s*:\s*([0-9.eE+\-]+)"),
        "acc_l2": re.compile(r"Acceleration\s+L2\s*:\s*([0-9.eE+\-]+)"),
        "pot_max": re.compile(r"Max\s+Potential\s+Error\s*:\s*([0-9.eE+\-]+)"),
        "acc_max": re.compile(r"Max\s+Acceleration\s+Error\s*:\s*([0-9.eE+\-]+)"),
    }
    for key, pat in patterns.items():
        m = pat.search(stdout)
        if m:
            result[key] = float(m.group(1))
    return result


def parse_jeans_omega(stdout: str) -> Optional[Dict[str, float]]:
    """Parse measured and analytical omega from JeansWaveErrors output.

    Matches lines like:
      Jeans wave omega measured  : 1.234e+01
      Jeans wave omega analytical: 1.234e+01
    Returns dict with keys 'measured' and 'analytical', or None if not found.
    """
    m_meas = re.search(
        r"Jeans\s+wave\s+omega\s+measured\s*:\s*([0-9.eE+\-]+)", stdout)
    m_anal = re.search(
        r"Jeans\s+wave\s+omega\s+analytical\s*:\s*([0-9.eE+\-]+)", stdout)
    if m_meas and m_anal:
        return {"measured": float(m_meas.group(1)),
                "analytical": float(m_anal.group(1))}
    return None


def assert_defect_convergence(stdout: str, min_orders: float = 8.0,
                              label: str = ""):
    """Assert that the MG defect drops by at least min_orders of magnitude.

    Parses all defect values from stdout and checks that
    final_defect / initial_defect < 10^{-min_orders}.
    """
    defects = parse_mg_defects(stdout)
    if len(defects) < 2:
        pytest.fail(f"{label}Expected at least 2 defect values, got {len(defects)}")

    initial = defects[0]
    final = defects[-1]
    if initial <= 0:
        pytest.fail(f"{label}Initial defect is non-positive: {initial}")

    ratio = final / initial
    threshold = 10.0 ** (-min_orders)
    if ratio > threshold:
        orders = -math.log10(ratio) if ratio > 0 else float("inf")
        pytest.fail(
            f"{label}Defect reduction insufficient: {orders:.1f} orders "
            f"(need {min_orders}). Initial={initial:.3e}, Final={final:.3e}"
        )


def assert_binary_gravity_accuracy(stdout: str,
                                   pot_l2_max: float = 1.0,
                                   acc_l2_max: float = 1.0,
                                   label: str = ""):
    """Assert that binary gravity L2 errors are below thresholds."""
    errs = parse_binary_gravity_errors(stdout)
    if not errs:
        pytest.fail(f"{label}No binary gravity errors found in output")

    if errs.get("pot_l2", float("inf")) > pot_l2_max:
        pytest.fail(
            f"{label}Potential L2 error {errs['pot_l2']:.3e} exceeds "
            f"threshold {pot_l2_max:.3e}"
        )
    if errs.get("acc_l2", float("inf")) > acc_l2_max:
        pytest.fail(
            f"{label}Acceleration L2 error {errs['acc_l2']:.3e} exceeds "
            f"threshold {acc_l2_max:.3e}"
        )


def assert_jeans_growth_rate(input_file: str, flags_fn, res_list: List[int],
                             max_rel_error: float, max_ratio: float,
                             mpi: bool = False, nranks: int = 4,
                             label: str = ""):
    """Run Jeans wave test at multiple resolutions and check growth rate accuracy.

    At each resolution, parses the measured and analytical omega from the C++
    output, computes the relative error, and then checks:
      1. Threshold: relative error at highest resolution < max_rel_error
      2. Convergence: ratio of errors (high_res / low_res) < max_ratio

    Args:
        input_file: Path to the Jeans wave input file.
        flags_fn: Callable(res) -> list of flag strings for a given resolution.
        res_list: List of resolutions to test (e.g. [32, 64]).
        max_rel_error: Maximum allowed relative omega error at highest resolution.
        max_ratio: Maximum allowed ratio of errors (high_res / low_res).
        mpi: Whether to use MPI.
        nranks: Number of MPI ranks.
        label: Label for error messages.
    """
    rel_errors = []
    for res in res_list:
        flags = flags_fn(res)
        results = run_athenak(input_file, flags, mpi=mpi, threads=nranks)
        if not results[0]:
            pytest.fail(f"{label}Run failed at resolution {res}")
        omega_data = parse_jeans_omega(results[1])
        if omega_data is None:
            pytest.fail(f"{label}No omega data found at resolution {res}")
        rel_err = (abs(omega_data["measured"] - omega_data["analytical"])
                   / omega_data["analytical"])
        rel_errors.append(rel_err)
        cleanup()

    if rel_errors[-1] > max_rel_error:
        pytest.fail(
            f"{label}Omega relative error {rel_errors[-1]:.4e} at "
            f"res={res_list[-1]} exceeds threshold {max_rel_error:.4e}"
        )

    if len(rel_errors) >= 2:
        ratio = rel_errors[-1] / rel_errors[-2]
        if ratio > max_ratio:
            pytest.fail(
                f"{label}Omega error ratio {ratio:.3f} exceeds threshold "
                f"{max_ratio:.3f}. Errors: {[f'{e:.4e}' for e in rel_errors]}"
            )


def parse_amr_block_counts(stdout: str) -> Optional[Dict[str, int]]:
    """Parse AMR block creation/deletion counts from stdout.

    Matches lines like:
      1008 MeshBlocks created, 0 deleted by AMR
    Returns dict with keys 'created' and 'deleted', or None if not found.
    """
    m = re.search(
        r"(\d+)\s+MeshBlocks\s+created,\s+(\d+)\s+deleted\s+by\s+AMR", stdout)
    if m:
        return {"created": int(m.group(1)), "deleted": int(m.group(2))}
    return None


def parse_final_defect(stdout: str) -> Optional[float]:
    """Parse the last 'Final defect norm' value from stdout.

    Matches: MGGravityDriver::Solve: Final defect norm = 1.234e-06
    Returns the last occurrence (there may be multiple from successive timesteps).
    """
    matches = re.findall(
        r"Final\s+defect\s+norm\s*=\s*([0-9.eE+\-]+)", stdout)
    if matches:
        return float(matches[-1])
    return None


def assert_solver_convergence(stdout: str, threshold: float,
                              max_iterations: int = 40,
                              max_avg_ratio: float = 0.5,
                              label: str = ""):
    """Assert that SolveIterative reached the target threshold.

    Parses per-solve defect trajectories and checks across all solves:
      1. Every final defect <= threshold
      2. Max V-cycles in any single solve <= max_iterations
      3. Worst-case average convergence ratio <= max_avg_ratio

    Returns (max_vcycles, worst_geo_mean, solves).
    """
    solves = parse_mg_defects_per_solve(stdout)
    if not solves:
        pytest.fail(f"{label}No MG defect output found")

    max_vcycles = 0
    worst_geo_mean = 0.0
    for si, defects in enumerate(solves):
        if len(defects) < 2:
            continue
        n_vc = len(defects) - 1
        max_vcycles = max(max_vcycles, n_vc)

        final = defects[-1]
        if final > threshold:
            pytest.fail(
                f"{label}Solve {si}: final defect {final:.3e} > "
                f"{threshold:.3e} after {n_vc} V-cycles")

        ratios = [defects[i+1] / defects[i]
                  for i in range(len(defects) - 1) if defects[i] > 0]
        if ratios:
            geo_mean = math.exp(
                sum(math.log(r) for r in ratios) / len(ratios))
            worst_geo_mean = max(worst_geo_mean, geo_mean)

    if max_vcycles > max_iterations:
        pytest.fail(
            f"{label}Solver took too many iterations: {max_vcycles} > "
            f"{max_iterations}")

    if worst_geo_mean > max_avg_ratio:
        pytest.fail(
            f"{label}Worst average convergence ratio {worst_geo_mean:.4f} "
            f"exceeds {max_avg_ratio}")

    return max_vcycles, worst_geo_mean, solves


def assert_defect_consistency(defects: List[float], max_spread: float,
                              label: str = ""):
    """Assert that all defect values are within max_spread orders of magnitude.

    Checks that log10(max/min) < max_spread across the provided defect values.
    """
    if not defects:
        pytest.fail(f"{label}No defect values to compare")
    dmin, dmax = min(defects), max(defects)
    if dmin <= 0:
        pytest.fail(f"{label}Non-positive defect: {dmin}")
    spread = math.log10(dmax / dmin)
    if spread > max_spread:
        pytest.fail(
            f"{label}Defect spread {spread:.2f} orders exceeds threshold "
            f"{max_spread}. Values: {[f'{d:.3e}' for d in defects]}"
        )
