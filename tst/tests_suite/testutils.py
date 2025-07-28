import os
import sys
from subprocess import Popen, PIPE
from typing import List
import time
sys.path.append('../vis/python')
import pytest
import athena_read

# Constants and configurations
ATHENAK_PATH = ".."
ATHENAK_BUILD = "build/src"

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=text)
    output, errors = process.communicate()
    if text:
        logging.debug(f"Output: {output}")
        logging.debug(f"Errors: {errors}")
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
        os.makedirs(os.path.join(ATHENAK_PATH, 'build'), exist_ok=True)
        os.chdir(ATHENAK_PATH)
        logging.info(f"Configuring CMake in {os.getcwd()}")
        
        command = ['cmake'] + flags + ["-B", "tst/build"]
        if not run_command(command, **kwargs):
            raise RuntimeError("CMake configuration failed")
    finally:
        os.chdir(original_dir)
    return True

def make(threads: int = os.cpu_count(), **kwargs) -> bool:
    """
    Runs the Make command to compile the project.

    Args:
        threads (int): The number of threads to use for compilation (default: number of CPU cores).
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the Make command succeeded, False otherwise.

    Raises:
        RuntimeError: If the Make command fails.
    """
    command = ['make', '-C', ATHENAK_BUILD, '-j', f'{threads}']
    start_time = time.time()
    status =  run_command(command, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    logging.info(f"make completed in {elapsed_time:.2f} seconds")
    if not status:
        raise RuntimeError("Make command failed")
    return True

def run(inputfile: str, flags=[], **kwargs)-> bool:
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
    command = [f"{ATHENAK_BUILD}/athena", '-i', inputfile] + flags
    if not run_command(command, **kwargs):
        logging.error(f"Failed to execute {inputfile} with flags {flags}")
        raise RuntimeError(f"Failed to execute {inputfile} with flags {flags}")
    return True

def mpi_run(inputfile: str, flags=[], threads: int = 8, **kwargs) -> bool:
    """
    Executes a test case using the AthenaK binary with MPI support.

    Args:
        inputfile (str): The path to the test case input file.
        flags (list): Additional flags to pass to the AthenaK binary.
        threads (int): The number of threads to use for MPI execution (default: number of CPU cores).
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the test case executed successfully, False otherwise.

    Raises:
        AssertionError: If the test case execution fails.
    """
    command = ['mpirun', '-np', str(threads), f"{ATHENAK_BUILD}/athena", '-i', inputfile] + flags
    if not run_command(command, **kwargs):
        logging.error(f"Failed to execute {inputfile} with flags {flags} using MPI")
        raise RuntimeError(f"Failed to execute {inputfile} with flags {flags} using MPI")
    return True

def athenak_run(inputfile, flags, use_cmake=False, use_make=False, **kwargs):
    """
    Executes a full test workflow, including optional CMake and Make steps.

    Args:
        inputfile (str): The path to the test case input file.
        flags (list): Additional flags to pass to the AthenaK binary.
        use_cmake (bool): Whether to run the CMake command before testing (default: False).
        use_make (bool): Whether to run the Make command before testing (default: False).
        **kwargs: Additional keyword arguments for `run_command`.

    Returns:
        bool: True if the test workflow succeeded, False otherwise.

    Raises:
        AssertionError: If any step in the workflow fails.
    """
    if use_cmake:
        cmake()
    if use_make:
        make()
    return run(inputfile, flags=flags, **kwargs)

def cleanup(text=False)-> None:
    """
    Cleans up the test environment by removing generated files.

    This function removes the output files generated during the test run.
    It is typically called after a test case has completed to ensure a clean state for subsequent tests.
    """
    if text:
        logging.info("Cleaning up test environment")
    Popen(["rm -rf tab/"], shell=True, stdout=PIPE).communicate()
    Popen(["rm " + "*.dat"], shell=True, stdout=PIPE).communicate()
    if text:
        logging.info("Cleanup completed")


def clean() -> None:
    """
    Cleans the build directoryt.

    This function is typically used to ensure that the build directory is clean
    """
    logging.info("Cleaning build directory")
    Popen(["rm -rf build/"], shell=True, stdout=PIPE).communicate()

def clean_make(threads: int = os.cpu_count(),**kwargs) -> None:
    """
    Cleans the build directory and rebuilds the project.

    This function is typically used to ensure that the build directory is clean before starting a new build.
    It removes all files in the build directory and then runs CMake and Make to rebuild the project.
    """
    clean()
    cmake(**kwargs) 
    make(threads=threads)  
    logging.info("Build directory cleaned and project rebuilt")

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
                        my_dict[tuple(keys)] = (float(error),float(ratio))
                    except ValueError:
                        print(f"Warning: Skipping invalid line: {line}")
        return my_dict
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None  # Or raise the exception, depending on desired behavior
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def test_error_convergence(input_file,
                        test_name,
                        arguments,
                        _wave,
                        _res,
                        iv,
                        rv,
                        fv,
                        left_wave='0',
                        right_wave='0',
                        rel="NR",
                        soe="hydro",
                        eos="ideal",
                        refinement="none",
                        dim=1,
                        mpi=False):
    run = mpi_run if mpi else athenak_run
    for wv in _wave:                    
        try:
            for res in _res:
                results = run(input_file, arguments(iv,rv,fv,wv,res,test_name))
                assert results, f"Test failed for iv={iv}, res={res}, fv={fv}, rv={rv}, wv={wv}./AthenaK run did not complete successfully."
            errors = read_dictionary_from_file("tests_suite/linwave1d_errors.txt")
            assert errors!=None, "Couldn't open errors dictionary"
            maxerror, errorratio = errors[(rel, soe, eos, iv, rv, wv, refinement, f"{dim}D")]
            #maxerror, errorratio = errors[('NR', 'hydro', 'ideal', 'rk3', 'wenoz', '0', 'none', '1D')]
            data = athena_read.error_dat(f'{test_name}-errs.dat')
            L1_RMS_INDEX = 4  # Index for L1 RMS error in data
            l1_rms_nLR = data[0][L1_RMS_INDEX]
            l1_rms_nHR = data[1][L1_RMS_INDEX]
            if l1_rms_nHR > maxerror and not(rv=="ppmx" and iv=="rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                pytest.fail(f"{wv} wave error too large for {iv}+{rv}+{fv} configuration, "
                    f"error: {l1_rms_nHR:g} threshold: {maxerror:g}")
            if l1_rms_nHR / l1_rms_nLR > errorratio and not(rv=="ppmx" and iv=="rk2"):
                # PPMX with RK2 is known to have larger errors, so we skip the check
                # Note that the convergence rate is defined as the ratio of errors at different resolutions
                pytest.fail(f"{wv} wave not converging for {iv}+{rv}+{fv}, "
                        f"conv: {l1_rms_nHR / l1_rms_nLR:g} threshold: {errorratio:g}")
            if wv == left_wave:  # Left wave
                l1_rms_l = l1_rms_nHR
            if wv == right_wave:  # Right wave
                l1_rms_r = l1_rms_nHR
        finally:
            cleanup()
    return l1_rms_l,l1_rms_r