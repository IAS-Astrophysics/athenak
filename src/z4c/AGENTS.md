# Directory Guide

## Role
Numerical-relativity spacetime evolution based on the Z4c formalism, plus related diagnostics and waveform extraction.

## Important Files
- `z4c.hpp`: high-level map of evolved variables, constraints, options, and owned data products.
- `z4c.cpp`, `z4c_calcrhs.cpp`, `z4c_update.cpp`, `z4c_newdt.cpp`: core evolution path.
- `z4c_tasks.cpp`, `z4c_adm.cpp`, `z4c_amr.*`: task integration, ADM conversion, and AMR-specific support.
- `z4c_gauge.cpp`, `z4c_Sbc.cpp`: gauge handling and Sommerfeld-style boundaries.
- `compact_object_tracker.*`, `horizon_dump.*`: puncture tracking and horizon diagnostics.
- `tmunu.*`, `z4c_calculate_weyl_scalars.cpp`, `z4c_wave_extr.cpp`: matter coupling and waveform/constraint diagnostics.

## Important Subdirectories
- `cce/`: Cauchy-characteristic extraction helpers.

## Read This Next
- For matter-coupled runs, also read `src/dyn_grmhd/AGENTS.md`.
- For task assembly, read `src/tasklist/AGENTS.md`.
