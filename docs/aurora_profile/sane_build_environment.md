# SANE continuation environment audit — 2026-10-13

Read-only inspection of `restart_simulation` and its production provenance identified
`/lus/flare/projects/MHDTidal/hzhu/SANE_continued/20261001/bin/athena_compat_v3`,
SHA256 `902aa894d1a2a34b8a882061492b2c5d9d3b54a8e15a4299df664eea18065f3b`.
The live controller and generated CMake flags were checked on Aurora, not merely
inferred from an old submission script. Production files were not changed.

| Setting | Validated SANE configuration | Previous cyclic-zoom benchmark |
|---|---|---|
| Compiler | oneAPI 2026.1.0 (20260617), GCC 14.3.0 environment | oneAPI 2025.3.1 |
| Optimization | Release, `-O3 -DNDEBUG`, C++17 | Release, `-O2 -DNDEBUG` |
| GPU | PVC AOT, `spir64_gen`, backend device `12.60.7` | generic Intel SYCL |
| Host execution | OpenMP, `-fiopenmp`, 8 threads/rank | Serial, 1 thread/rank |
| Kokkos | 4.7.02, commit `6739bc623081648af9e752b616d9671527922cbf`, two compatibility patches | 4.6.1 |
| MPI at compile | `5.0.0.aurora_test.87e2045` | `.87e2045`, older site installation |
| MPI at runtime | `5.0.0.aurora_test.51a9474` | `.87e2045` |
| GPU IPC | `pidfd`, cache 1024, GPU support enabled | not explicitly pinned |

Other pinned modules: libfabric 2.3.1, cray-pals/cray-libpals 1.8.0, CMake 3.31.11.
Kokkos uses SYCL relocatable device code, aggressive vectorization, debug and
bounds checks off, DualView modification checking on. Link flag:
`-fsycl-max-parallel-link-jobs=8`. No explicit fast-math flag.

Runtime also sets `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`,
`OMP_PLACES=threads`, `OMP_PROC_BIND=spread`, `FI_MR_CACHE_MONITOR=disabled`,
`FI_LOG_LEVEL=ERROR`, and the site wrapper's `EnableImplicitScaling=0`
(required there for RDMA). The 12-rank layout uses one tile/rank and eight disjoint CPU
threads/rank, covering all six GPUs. The six-rank comparison selects one root GPU/rank with eight CPU threads each;
implicit tile scaling remains disabled, so the twelve-rank layout is the
explicit test of all twelve tiles. Each rank has its own log.

The production compiler launcher handles only the host-only meshblock pack
translation unit; all kernel translation units retain SYCL compilation. Its
exact source is copied into `scripts/tests/sane_sycl_compile.py`. The two Kokkos
compatibility edits are retained in `sane-environment/kokkos-compat.patch`.

The new build uses an isolated `source-sane` tree with the matching patched
Kokkos dependency and `build-sane` directory. Local Mac builds keep their existing
dependency. Build concurrency is four, with a 24 GB aggregate RSS guard, rather
than SANE's eight, to respect this task's memory constraint. Runtime tests retain
the combined allocation/RSS budgets below 30 GB and one-node debug-only scope.

Jobs through 8924777 used the previous environment and cannot establish
performance under the SANE environment. Job 8924777 was cancelled after its
six-rank case stopped advancing; its partial logs are retained as diagnostic
material. Rebuild and new benchmark validation are required before quoting new
performance. Existing one-GPU team-size results are specific to the old build. The new
benchmark repeats AUTO/32/64/128 before selecting the team size, then compares
six whole-GPU ranks and twelve tile ranks under this same executable/runtime.

For a reproducible fresh source copy on Aurora:

```bash
bash source/scripts/tests/aurora_sane_prepare.sh source source-sane-new
bash source-sane-new/scripts/tests/aurora_sane_build.sh "$PWD/source-sane-new" "$PWD/build-sane-new"
```

The prepare helper checks the pinned Kokkos commit and refuses an existing
output directory. Use the matching build path in the one-node PBS script.

Rebuild completed successfully: 702.73 seconds, aggregate peak RSS 4.378 GB,
SHA256 `9cbf45fa4ecef2049af1c6dfa2d2587acaa4c7e95bb8b5d2ca4472032e20ca33`.
The saved `configuration-comparison.json` confirms all 17 selected settings
match. A runtime `ldd` check resolves MPICH `.51a9474`, oneAPI 2026.1 SYCL and
OpenMP libraries. PVC compilation emitted register-spill warnings in some flux
kernels; these are performance evidence, not build errors.

Matched-environment benchmark job **8924820** subsequently finished successfully
(PBS Exit_status=0, wall time 8m51s). Warm step timings over cycles 10–50 on the
original 1520-block benchmark were 0.8824 s (one rank, AUTO), 0.3071 s (one rank,
team 64), 0.3574 s (six ranks, team 64), and 0.19185 s (twelve tile ranks, team 64).
Team 64 was the fastest of 0/32/64/128 under this toolchain. Twelve tile ranks
were faster than six root-GPU ranks for this small mesh. These are short warm
intervals, not end-to-end production or convergence measurements. The full
nine-level accelerated roundtrip returned to its finest stage, and the reduced-
residence stage profile reached 330 M. See `sane-benchmark-8924820.json`.
