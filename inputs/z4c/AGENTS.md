# Directory Guide

## Role
Numerical-relativity input decks centered on the Z4c spacetime solver.

## Important Subdirectories
- `awa/`: gauge-wave and stability-style tests.
- `onepuncture/`: single puncture black-hole examples.
- `spectre_bbh/`: SpECTRE-exported binary black hole input.
- `twopuncture/`: binary puncture and tracker-focused setups.

## Important Files
- `z4c_boosted_puncture.athinput`: top-level boosted puncture example used by regression tests.

## Read This Next
- For implementation details, read `src/z4c/AGENTS.md`.
- For matter-coupled dynamical spacetime runs, also inspect `inputs/dyngr/AGENTS.md`.
