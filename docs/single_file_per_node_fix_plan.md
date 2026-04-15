# `single_file_per_node` Fix Plan

## Goals

- [ ] Make `single_file_per_node` scale cleanly for very large restart/output workloads.
- [ ] Remove restart-path bottlenecks before tuning lower-priority output paths.
- [ ] Preserve restart compatibility with existing `shared`, `per_rank`, and current `per_node` files unless an explicit format bump is introduced.
- [ ] Add enough tests and instrumentation to verify both correctness and performance regressions.

## Phase 1: Redesign Per-Node Restart Metadata

- [ ] Define a restart metadata layout that avoids writing the full global manifest into every node shard.
- [ ] Choose one of these implementations and document it before coding:
  - [ ] Preferred: one shared manifest file plus per-node payload files.
  - [ ] Fallback: a single canonical manifest embedded once, with node shards containing payload only.
- [ ] Update restart writer logic in [src/outputs/restart.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/outputs/restart.cpp) so per-node output writes global metadata exactly once.
- [ ] Update restart discovery/open logic in [src/main.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/main.cpp) to locate the new manifest/payload layout robustly.
- [ ] Update restart metadata storage in [src/mesh/mesh.hpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/mesh/mesh.hpp) and related setup code so the reader has all state it needs without re-deriving paths ad hoc.
- [ ] Decide and implement backward-compatibility behavior for old per-node restart files:
  - [ ] Read legacy per-node files unchanged.
  - [ ] Read new-format files with the optimized path.
  - [ ] Emit a clear format/version marker so the reader can distinguish them cheaply.

## Phase 2: Fix Per-Node Restart Read Scalability

- [ ] Rework `LoadPartitionedRestartData()` in [src/pgen/pgen.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/pgen/pgen.cpp) so shard files are not reopened once per physics component.
- [ ] Precompute per-rank intra-node block prefixes once, replacing the current O(ranks) scan in `chunk_base()`.
- [ ] Group restart block requests by shard and sort them by file offset.
- [ ] Coalesce adjacent or nearly adjacent reads into larger transfers where practical.
- [ ] Keep shard handles open for the full load pass instead of repeated open/close cycles.
- [ ] Decide on the final read execution model:
  - [ ] Option A: every rank reads its own data, but with coalesced large reads.
  - [ ] Option B: one reader per current node loads shard data and redistributes over `node_comm`.
  - [ ] Option C: use MPI-IO collectively for per-node shard reads instead of forced serial I/O.
- [ ] Benchmark the chosen model against the current implementation on at least one multi-node restart workload.

## Phase 3: Remove Large-Count MPI Overflow Risks

- [ ] Audit all restart metadata read/write/broadcast paths for `IOWrapperSizeT` to `int` truncation.
- [ ] Add chunked helpers for large byte transfers in [src/outputs/io_wrapper.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/outputs/io_wrapper.cpp) and [src/outputs/io_wrapper.hpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/outputs/io_wrapper.hpp), or add higher-level chunking at each call site.
- [ ] Apply chunking to manifest reads in [src/mesh/build_tree.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/mesh/build_tree.cpp).
- [ ] Apply chunking to any large metadata writes in [src/outputs/restart.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/outputs/restart.cpp).
- [ ] Review `ParameterInput::LoadFromFile()` and any other header read path for similar count-limit assumptions.
- [ ] Add explicit error handling when a path still exceeds supported transfer sizes.

## Phase 4: Harden Restart Path Parsing and Layout Detection

- [ ] Replace raw substring detection in [src/main.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/main.cpp) with path-component-based detection for `rank_*` and `node_*`.
- [ ] Make relative paths, absolute paths, and trailing-slash variations behave the same way.
- [ ] Add tests for:
  - [ ] `rst/node_00000003/foo.rst`
  - [ ] `node_00000003/foo.rst`
  - [ ] `./rst/node_00000003/foo.rst`
  - [ ] shared restart files with no shard directory

## Phase 5: Reduce Spherical-Slice Per-Node Communication Cost

- [ ] Redesign the per-node `spherical_slice` accumulation in [src/outputs/spherical_slice.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/outputs/spherical_slice.cpp) so it does not reduce a dense `(nvars, ntheta, nphi)` array when only sparse owned angles are written.
- [ ] Replace the dense node-wide mask reduction with a sparse angle ownership exchange.
- [ ] Ensure the writer still emits a format that [vis/python/read_sphslice.py](/Users/dbf75/Work/Research/AthenaK/athenak-DF/vis/python/read_sphslice.py) can reconstruct, or update the reader in the same change.
- [ ] Add a correctness check that the union of node shard angle indices covers the full surface exactly once.
- [ ] Benchmark communication volume and memory footprint before and after the change.

## Phase 6: Remove Full-Payload Host Staging in Binary Writers

- [ ] Refactor [src/outputs/binary.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/outputs/binary.cpp) to stream MeshBlock payloads directly to the final offsets instead of staging `write_mbs * data_size` bytes in one large host buffer.
- [ ] Apply the same streaming approach to [src/outputs/coarsened_binary.cpp](/Users/dbf75/Work/Research/AthenaK/athenak-DF/src/outputs/coarsened_binary.cpp).
- [ ] Preserve the current on-disk block order and offset math.
- [ ] Keep the existing large-write chunking behavior for MPI count safety.
- [ ] Re-measure peak host memory during binary/coarsened-binary output.

## Cross-Cutting Validation

- [ ] Add or extend regression tests for `shared`, `per_rank`, and `per_node` restart round-trips.
- [ ] Add at least one compatibility test that reads a legacy per-node restart produced before these changes.
- [ ] Add a stress test that exercises very large manifest sizes without overflowing MPI count-limited paths.
- [ ] Add lightweight timing/logging around restart metadata load, payload load, and shard open counts so performance regressions are visible.
- [ ] Document the final format and operational tradeoffs in `docs/`.

## Recommended Execution Order

- [ ] Implement Phase 1 and Phase 4 together so the new restart layout and path handling land coherently.
- [ ] Implement Phase 2 immediately after Phase 1 so the new format gets the optimized reader from the start.
- [ ] Implement Phase 3 before declaring restart work complete.
- [ ] Implement Phases 5 and 6 after restart work is stable, since they are important but not as correctness-critical.
