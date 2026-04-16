# Directory Guide

## Role
Owns GitHub-hosted project automation. This tree is small; the main value is knowing that CI configuration lives here, not in CMake or Python tooling.

## Important Subdirectories
- `workflows/`: GitHub Actions definitions. Read `workflows/AGENTS.md` for the job matrix and runner assumptions.

## Read This Next
- For CI edits, start in `workflows/main.yml`.
- For local reproduction of failing jobs, jump to `tst/AGENTS.md`.
