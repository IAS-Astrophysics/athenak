# Directory Guide

## Role
Older regression framework organized around Python modules that expose `run()` and `analyze()` functions.

## Important Files
- `__init__.py`: package marker used by `run_tests.py`.
- `utils/`: shared build/run/logging helpers.
- `diffusion/`: standalone STS diffusion benchmark driver that produces CSV data and TeX
  table fragments for the colleague-facing benchmark note.
- Physics subdirectories (`gr/`, `hydro/`, `mhd/`, `radiation/`, `z4c/`): legacy regression scripts grouped by area.
- `style/`: legacy style-check shell and config files.

## Read This Next
- Use this tree only if you are working on `tst/run_tests.py` or maintaining the legacy harness.
- For the actively used pytest path, move to `../test_suite/AGENTS.md`.
- For the explicit-vs-STS accuracy/cost benchmark package, go directly to `diffusion/`.
