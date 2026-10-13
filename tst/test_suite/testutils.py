"""
Various utility functions used for automatic testing, including
  - functions for building code on target device (CPU/GPU)
  - functions for running code on target device
  - functions to clean up run directories at end of testing
"""

# Modules
import os
from subprocess import Popen, PIPE
from typing import List
import time
import pytest
import logging
import sys

sys.path.insert(0, "../vis/python")
import athena_read  # noqa: E402

athena_read.check_nan_flag = True  # Enable NaN checking in athena_read

# Constants and configurations
ATHENAK_PATH = ".."
ATHENAK_BUILD = "build/src"

# Configure logging
LOG_FILE_PATH = os.path.abspath(os.path.join(ATHENAK_PATH, "tst", "test_log.txt"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(),  # Optional: Keep console logging
    ],
)


def run_command(command: List[str], text: bool = False) -> bool:
    """
    Executes a shell command and captures its output and errors.

    Args:
        command (list): The command to execute as a list of strings.
        text (bool): Whether to treat output and errors as text (default: False).

    Returns:
        bool: True if the command executed successfully, False otherwise.
    """

    logging.info(f"Executing command: {' '.join(command)}")
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
    # Log the output only to the file
    with open(LOG_FILE_PATH, "a") as log_file:
        output, errors = process.communicate()
        log_file.write(output)
        log_file.write(errors)

    if process.returncode != 0:
        logging.error(f"Command failed with return code {process.returncode}")
    return process.returncode == 0


def cmake(flags: List[str] = [], **kwargs) -> bool:
    """
    Runs the CMake command to configure the build system.

    Args:
        flags (list): Additional flags to pass to the CMake command.
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the CMake command succeeded, False otherwise.

    Raises:
        RuntimeError: If the CMake command fails.
    """
    original_dir = os.getcwd()
    try:
        os.chdir(ATHENAK_PATH)
        logging.info(f"Configuring CMake in {os.getcwd()}")

        command = ["cmake"] + flags + ["-B", "tst/build"]
        if not run_command(command, **kwargs):
            raise RuntimeError("CMake configuration failed")
    finally:
        os.chdir(original_dir)
    return True


def make(threads: int = os.cpu_count(), **kwargs) -> bool:
    """
    Runs the Make command to compile the project.

    Args:
        threads (int): number of threads to use for compilation (default: num of cores).
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the Make command succeeded, False otherwise.

    Raises:
        RuntimeError: If the Make command fails.
    """
    os.chdir(ATHENAK_BUILD)
    command = ["make", "-j", f"{threads}"]
    start_time = time.time()
    status = run_command(command, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logging.info(f"make completed in {elapsed_time:.2f} seconds")
    if not status:
        raise RuntimeError("Make command failed")
    return True


def run(inputfile: str, flags=[], **kwargs) -> bool:
    """
    Executes a test case using the AthenaK binary.

    Args:
        inputfile (str): The path to the test case inputfile file.
        flags (list): Additional flags to pass to the AthenaK binary.
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the test case executed successfully, False otherwise.

    Raises:
        AssertionError: If the test case execution fails.
    """
    command = ["./athena", "-i", inputfile] + flags
    if not run_command(command, **kwargs):
        logging.error(f"Failed to execute {inputfile} with flags {flags}")
        raise RuntimeError(f"Failed to execute {inputfile} with flags {flags}")
    return True


def mpi_run(
    inputfile: str, flags=[], threads: int = min(16, os.cpu_count()), **kwargs
) -> bool:
    """
    Executes a test case using the AthenaK binary with MPI support.

    Args:
        inputfile (str): The path to the test case input file.
        flags (list): Additional flags to pass to the AthenaK binary.
        threads (int): Number of threads to use for MPI execution (default: num of cores).
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the test case executed successfully, False otherwise.

    Raises:
        AssertionError: If the test case execution fails.
    """
    command = ["mpirun", "-np", str(threads), "./athena", "-i", inputfile] + flags
    if not run_command(command, **kwargs):
        logging.error(
           f"Failed to execute {inputfile} with flags {flags} using MPI "
           f"and {threads}-threads"
        )
        raise RuntimeError(
            f"Failed to execute {inputfile} with flags {flags} using MPI "
            f"and {threads}-threads"
        )
    return True


def cleanup(text=False) -> None:
    """
    Cleans up the test environment by removing generated files.
    """
    if text:
        logging.info("Cleaning up test environment")
    Popen(["rm -rf tab/"], shell=True, stdout=PIPE).communicate()
    Popen(["rm " + "*.dat"], shell=True, stdout=PIPE).communicate()
    if text:
        logging.info("Cleanup completed")


def clean() -> None:
    """
    Cleans the build directory.
    """
    logging.info("Cleaning build directory")
    Popen(["rm -rf build/"], shell=True, stdout=PIPE).communicate()


def clean_make(threads: int = os.cpu_count(), **kwargs) -> None:
    """
    Cleans the build directory and rebuilds the project.
    Removes all files in the build directory and then runs CMake and Make.
    """
    clean()
    cmake(**kwargs)
    make(threads=threads)
    logging.info("Build directory cleaned and project rebuilt")
    run_command(["ln", "-s", "../../inputs", "inputs"])


def read_dictionary_from_file(file_path):
    """
    Reads a dictionary from a file where each line is in the format "key: value".

    Args:
        file_path: The path to the file.

    Returns:
        A dictionary, or None if an error occurred.
    """
    try:
        my_dict = {}
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Skip empty lines
                    try:
                        keys, values = line.split(": ", 1)  # Split at the first ": "
                        values = values.strip("()")
                        error, ratio = values.split(",")
                        keys = keys.strip("()")
                        keys = keys.split(",")
                        keys = [key.strip(" '") for key in keys]
                        my_dict[tuple(keys)] = (float(error), float(ratio))
                    except ValueError:
                        print(f"Warning: Skipping invalid line: {line}")
        return my_dict
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None  # Or raise the exception, depending on desired behavior
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def test_error_convergence(
    input_file,
    test_name,
    arguments,
    errors,
    _wave,
    _res,
    iv,
    rv,
    fv,
    soe,
    left_wave="0",
    right_wave="0",
    mpi=False,
):
    RUN = mpi_run if mpi else run
    for wv in _wave:
        try:
            for res in _res:
                results = RUN(
                    input_file, arguments(iv, rv, fv, wv, res, soe, test_name)
                )
                assert results, f"Run failed for {soe}+{iv}+{res}+{fv}+{rv}+{wv}."
            maxerror, maxerrorratio = errors[(soe, iv, rv, wv)]
            data = athena_read.error_dat(f"{test_name}-errs.dat")
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_nLR = data[0][L1_RMS_INDEX]
            l1_rms_nHR = data[1][L1_RMS_INDEX]
            errorratio = l1_rms_nHR / l1_rms_nLR
            if l1_rms_nHR > maxerror and not (rv == "ppmx" and iv == "rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                pytest.fail(
                    f"{wv} wave error too large for {soe}+{iv}+{rv}+{fv},"
                    f"error: {l1_rms_nHR:g} threshold: {maxerror:g}"
                )
            if errorratio > maxerrorratio and not (rv == "ppmx" and iv == "rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                pytest.fail(
                    f"{wv} not converging for {soe}+{iv}+{rv}+{fv},"
                    f"error ratio: {errorratio:g} threshold: {maxerrorratio:g}"
                )
            # store errors for selected L/R-going waves
            if wv == left_wave:
                l1_rms_l = l1_rms_nHR
            if wv == right_wave:
                l1_rms_r = l1_rms_nHR
        finally:
            cleanup()
    return l1_rms_l, l1_rms_r
