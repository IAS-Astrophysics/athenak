
# Modules
import sys
sys.path.append('../tst/tests_suite')
import tests_suite.testutils as testutils
import pytest
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Run AthenaK test suites.")
parser.add_argument(
    "--suite", 
    choices=["style", "cpu", "mpicpu"], 
    nargs="+", 
    help="Specify which test suites to run."
)
args = parser.parse_args()

if not args.suite:
    print("No test suite specified. Use --suite to select one or more suites.")
    sys.exit(1)

# Run tests based on arguments
if "style" in args.suite:
    pytest.main(["tests_suite/style"])

if "cpu" in args.suite:
    testutils.clean_make()
    pytest.main(["tests_suite/hydro", "-k", "_cpu"])
    pytest.main(["tests_suite/mhd", "-k", "_cpu"])

if "mpicpu" in args.suite:
    testutils.clean_make(flags=["-D","Athena_ENABLE_MPI=ON"])
    pytest.main(["tests_suite/hydro", "-k", "_mpicpu"])
    pytest.main(["tests_suite/mhd", "-k", "_mpicpu"])

testutils.clean()