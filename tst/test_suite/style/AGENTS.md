# Directory Guide

## Role
Pytest wrapper for code-style enforcement used by CI.

## Important Files
- `test_style.py`: pytest entrypoint that shells out to the style-check assets.
- `check_athena_cpp_style.sh`, `cpplint.py`, `clang_format_v1.cfg`, `clang_format_v2.cfg`, `count_athena.sh`: copied-in style resources kept next to the test for CI convenience.

## Read This Next
- CI lint failures usually start here, then backtrack to the specific script or source file that violated the checks.
