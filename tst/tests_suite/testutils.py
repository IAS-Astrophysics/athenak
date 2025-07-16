import os
from subprocess import Popen, PIPE
from typing import List
import time

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

def mpi_run(inputfile: str, flags=[], threads: int = os.cpu_count(), **kwargs) -> bool:
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