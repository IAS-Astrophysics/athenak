# Directory Guide

## Role
Newtonian MHD example inputs covering canonical Athena-style verification problems.

## Important Files
- `bw.athinput`, `blast_mhd*.athinput`, `orszag_tang.athinput`: standard shock and turbulence-style MHD regressions.
- `field_loop*.athinput`, `current_sheet.athinput`: magnetic advection and topology-preservation checks.
- `kh2d-lecoanet-mhd.athinput`, `rt2d-mhd.athinput`, `resistivity.athinput`: instability and diffusion-focused cases.

## Read This Next
- For face-centered field handling and CT, read `src/mhd/AGENTS.md`.
- For resistive terms, also inspect `src/diffusion/AGENTS.md`.
