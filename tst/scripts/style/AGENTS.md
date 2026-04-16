# Directory Guide

## Role
Legacy style-check assets referenced by the older test harness.

## Important Files
- `check_athena_cpp_style.sh`, `cpplint.py`: style entrypoints.
- `clang_format_v1.cfg`, `clang_format_v2.cfg`: formatting configurations.
- `count_athena.sh`: counting helper used by style checks.

## Read This Next
- CI currently exercises the pytest wrapper in `tst/test_suite/style/`.
