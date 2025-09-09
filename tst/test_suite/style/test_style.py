"""
Checks source code style, based on Google style guide with some custom modifications.
"""

# Modules
import os
import pytest
from subprocess import Popen, PIPE


def test_style():
    """
    Checks AthenaK source code to ensure it adheres to the coding standards.
    If any style violations are found, the test will fail.
    """
    original_dir = os.getcwd()
    os.chdir("test_suite/style/")
    try:
        command = ["bash", "check_athena_cpp_style.sh"]
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        output, errors = process.communicate()
        status = process.returncode == 0
        if not status:
            pytest.fail(
                "Code style check failed. Please fix the style issues."
                "\nErrors:\n" + errors.decode()
            )
        status = False if "Done" not in output.decode() else status
        if not status:
            pytest.fail(
                "Code style check did not complete successfully. "
                "Please ensure the script runs correctly."
            )
    finally:
        os.chdir(original_dir)


def test_lint_python():
    """
    Checks Python source code to ensure it adheres to the coding standards using pylint.
    If any linting issues are found, the test will fail.
    """
    original_dir = os.getcwd()
    os.chdir("..")
    try:
        print("Running Python linting...")
        print("Current directory:", os.getcwd())
        command = ["python", "-m", "flake8"]
        process = Popen(command, stdout=PIPE, stderr=PIPE)
        output, errors = process.communicate()
        status = process.returncode == 0
        if not status:
            pytest.fail(
                "Python linting failed. Please fix the linting issues."
                "\nErrors:\n" + errors.decode() + "\nOutput:\n" + output.decode()
            )
    finally:
        os.chdir(original_dir)
