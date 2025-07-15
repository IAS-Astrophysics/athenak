
# Modules
import sys
sys.path.append('../tst/tests_suite')
import tests_suite.testutils as testutils
import pytest
import argparse

def cmake_flags(args,flags):
    """Process command line arguments and return cmake flags."""
    if args:
        for arg in args:
            flags += (arg.split(" "))
    return flags

# Set up argument parser
parser = argparse.ArgumentParser(description="Run AthenaK test suites.")
parser.add_argument(
    "--style", 
    help="Test code style."
)
parser.add_argument(
    "--cpu", 
    nargs="*", 
    help="Add cmake arguments if needed."
)
parser.add_argument(
    "--mpicpu", 
    nargs="*", 
    help="Add cmake arguments if needed."
)
args = parser.parse_args()

if args.style==None and args.cpu==None and args.mpicpu==None:
    print("No test suite specified.")
    print(parser.format_help())
    sys.exit(1)

# Run tests based on arguments
if  args.style:
    pytest.main(["tests_suite/style"])

if args.cpu != None:
    testutils.clean_make(flags=cmake_flags(args.cpu, []))
    pytest.main(["tests_suite/hydro", "-k", "_cpu"])
    pytest.main(["tests_suite/mhd", "-k", "_cpu"])

if args.mpicpu != None: 
    testutils.clean_make(flags=cmake_flags(args.mpicpu, ["-D", "Athena_ENABLE_MPI=ON"]))
    pytest.main(["tests_suite/hydro", "-k", "_mpicpu"])
    pytest.main(["tests_suite/mhd", "-k", "_mpicpu"])

testutils.clean()