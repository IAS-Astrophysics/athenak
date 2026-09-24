#!/usr/bin/env python

"""
Script to run automatic test suite.

Usage: From this directory, call this script with python:
      python run_test_suite.py ARGS [ARGS]

Notes:
  - Mandatory arguments specify whether to run tests on CPU, CPU+MPI, or GPU
  - Additional optional arguments for cmake can be supplied (e.g. to build on GPUs)
  - Requires Python 2.7+. (compliant with Python 3)
  - This file does not need to be modified when adding new scripts.
  - To add a new script, create a new .py file in a /test_suite/ subdirectory.
  - Scripts that run tests on CPU must have '_cpu' in name
  - Scripts that run tests on CPU with MPI must have '_mpicpu' in name
  - Scripts that run tests on GPU must have '_gpu' in name
  - For more information, check online automatic testing wiki page.
"""

import os
import sys
import pytest
import argparse
import test_suite.testutils as testutils

sys.path.append("../tst/test_suite")

# Remove the log file at the beginning of the script
LOG_FILE_PATH = os.path.abspath("../tst/test_log.txt")
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)
else:
    print(f"Log file {LOG_FILE_PATH} does not exist, nothing to remove.")


def cmake_flags(args, flags):
    """Process command line arguments and return cmake flags."""
    if args:
        for arg in args:
            flags += arg.split(" ")
    return flags


def test(args):
    """Run pytest with given arguments."""
    exit_code = pytest.main(args)
    if exit_code != 0:
        sys.exit(exit_code)


def run_tests_with_custom_problem(
    selected_tests, keyword, build_flags, custom_test, custom_problem
):
    """Run default tests, then rebuild and run a test that needs a custom pgen."""
    test_driver_dir = os.getcwd()
    test_root = os.path.abspath("test_suite")
    custom_test = os.path.abspath(custom_test)
    run_custom = selected_tests in (test_root, custom_test)

    if selected_tests != custom_test:
        os.chdir(test_driver_dir)
        testutils.clean_make(flags=build_flags)
        pytest_args = [selected_tests, "-k", keyword]
        if run_custom:
            pytest_args.extend(["--ignore", custom_test])
        test(pytest_args)

    if run_custom:
        os.chdir(test_driver_dir)
        custom_flags = build_flags + ["-D", f"PROBLEM={custom_problem}"]
        testutils.clean_make(flags=custom_flags)
        test([custom_test, "-k", keyword])


# Set up argument parser
parser = argparse.ArgumentParser(description="Run AthenaK test suite.")
parser.add_argument(
    "--style", action="store_true", help="check source code conforms to style guide."
)
parser.add_argument(
    "--cpu", nargs="*", help="Run test on CPU. Can add optional cmake arguments."
)
parser.add_argument(
    "--mpicpu",
    nargs="*",
    help="Run test on CPU with MPI. Can add additional cmake arguments.",
)
parser.add_argument(
    "--gpu", nargs="*", help="Run test on GPU. Can add optional cmake arguments."
)
parser.add_argument("--test", type=str, help="Run a specific test by name.")


args = parser.parse_args()
status = True
for arg in vars(args):
    status *= getattr(args, arg) is None
if status:
    print("No target device (CPU/GPU) specified.")
    print(parser.format_help())
    sys.exit(1)

# Run tests based on arguments
if args.style:
    test(["test_suite/style"])

original_dir = os.getcwd()
tests = "test_suite/"

if args.test is not None:
    tests = args.test
    for suffix in ["cpu", "mpicpu", "gpu"]:
        if "_" + suffix in tests:
            if getattr(args, suffix) is None:
                setattr(args, suffix, [])
        else:
            setattr(args, suffix, None)

    if "_cpu" not in tests and "_mpicpu" not in tests and "_gpu" not in tests:
        print(
            "Invalid test name. Please ensure it contains '_cpu', '_mpicpu', or '_gpu'."
        )
        sys.exit(1)

tests = os.path.abspath(tests)

if args.cpu is not None:
    run_tests_with_custom_problem(
        tests,
        "_cpu",
        cmake_flags(args.cpu, []),
        "test_suite/dyngrmhd/test_dyngrmhd_dynbbh_metric_cpu.py",
        "dyn_grmhd/dynbbh",
    )

if args.mpicpu is not None:
    testutils.clean_make(flags=cmake_flags(args.mpicpu, ["-D", "Athena_ENABLE_MPI=ON"]))
    test([tests, "-k", "_mpicpu"])  # run all scripts with _mpicpu in name

if args.gpu is not None:
    run_tests_with_custom_problem(
        tests,
        "_gpu",
        cmake_flags(args.gpu, ["-D", "Kokkos_ENABLE_CUDA=On"]),
        "test_suite/dyngrmhd/test_dyngrmhd_dynbbh_metric_gpu.py",
        "dyn_grmhd/dynbbh",
    )

os.chdir(original_dir)
testutils.clean()
