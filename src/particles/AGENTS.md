# Directory Guide

## Role
Particle storage, pushers, and inter-meshblock migration.

## Important Files
- `particles.hpp`: particle-owned state, pusher choice, and task IDs.
- `particles.cpp`: construction and task assembly.
- `particles_pushers.cpp`: concrete push/update algorithms.
- `particles_tasks.cpp`: communication and migration task implementations.

## Read This Next
- For boundary migration details, also read `src/bvals/AGENTS.md`.
