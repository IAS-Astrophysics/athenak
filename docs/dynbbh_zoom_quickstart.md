# Circumbinary disk zoom: collaborator quick-start

Use the latest `project/ttorus` and rebuild AthenaK with
`PROBLEM=dyn_grmhd/dynbbh` and your usual Kokkos backend. Run these commands
from the repository root; replace the example paths. Use your site's launcher
(e.g. `srun`) before each Athena command inside a compute allocation.

## 1. Prepare

Start with a validated disk input. Keep `<dyn_radiation>` with `geometry=adm`
for radiation, or omit that block for GRMHD only. Disable `unresolved_sink`;
remove static refinement regions and non-user AMR criteria.

```sh
cp inputs/dyn_grmhd/dynbbh_zoom_schedule.json zoom.json
# Edit zoom.json: stage end times, refinement levels/radii, excision radii,
# horizon_fraction, shrink_start_time, shrink_timescale and block capacity.
python3 scripts/setup_dynbbh_zoom.py \
  --input /absolute/path/to/disk.athinput \
  --schedule zoom.json \
  --output-dir /absolute/path/to/cbd-zoom
```

Choose a new output directory. The example times are placeholders, not proven
equilibration times. A trajectory table must cover the entire run.

## 2. Run stages in order

**outer -> inner -> resolve -> shrink -> settle**

```sh
export ATHENA=/absolute/path/to/build/src/athena
export ZOOM=/absolute/path/to/cbd-zoom

# Fresh coarse relaxation:
"$ATHENA" -i "$ZOOM/outer/stage.athinput" -d "$ZOOM/outer"

# After outer reaches its tlim, replace NNNNN with its final checkpoint number:
"$ATHENA" -r "$ZOOM/outer/rst/outer.NNNNN.rst" \
  -i "$ZOOM/inner/stage.athinput" -d "$ZOOM/inner"
```

Repeat the restart command for `resolve`, `shrink`, then `settle`, using each
preceding stage's final checkpoint. `RUN.txt` lists every stage command.
Before `shrink`, verify that the actual mesh resolves both moving holes.
The example ramp ends at `0.8*rH`; `rH` is an isolated-Kerr estimate, not a
measured binary horizon. Allow the newly uncovered inner flow to settle.

## 3. Resume an interrupted stage

Use that stage's latest complete checkpoint and the same output directory,
**without `-i`**:

```sh
"$ATHENA" -r "$ZOOM/shrink/rst/shrink.NNNNN.rst" -d "$ZOOM/shrink"
```

Do not advance after a walltime stop until the stage reaches its tlim. Do not
reset the shrink clock or change mesh sizes, radiation settings or angular grid.
For per-rank restarts pass the rank-00000000 file, retain its sibling rank files
and keep the MPI rank count unchanged.

See [the full guide](dynbbh_zoom.md) for resolution, memory, excision and
validation details. Serial/OpenMP workflow tests passed; fresh CUDA validation
was blocked by Della SSH authentication.
