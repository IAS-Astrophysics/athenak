"""
Checks source code style, based on Google style guide with some custom modifications.
"""

# Modules
import sys
import os
import pytest
import subprocess
from subprocess import Popen, PIPE

def test_style():
    """
    Checks AthenaK source code to ensure it adheres to the coding standards.
    If any style violations are found, the test will fail.
    """
    original_dir = os.getcwd()
    os.chdir('test_suite/style/')
    try:
        command = ['bash', 'check_athena_cpp_style.sh']
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        output, errors = process.communicate()
        status = process.returncode == 0
        if not status:
            pytest.fail("Code style check failed. Please fix the style issues." \
            "\nErrors:\n" + errors.decode())
        status = False if "Done" not in output.decode() else status
        if not status:
            pytest.fail("Code style check did not complete successfully. " \
            "Please ensure the script runs correctly.")
    finally:
        os.chdir(original_dir)
