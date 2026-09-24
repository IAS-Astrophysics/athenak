# Aurora Staging Scan

This documents the initial non-invasive scan for planning a later transfer of
selected Aurora simulation outputs.

The scan writes manifests and summaries only. It does not copy, move, delete,
compress, or modify simulation data under `/home/hzhu/scratch2/hzhu/acc/...`.

## Searched Locations

For each matching `run*` directory, `scripts/staging/scan_aurora_runs.py` uses
constrained `find` calls:

- `run*/bin/`, depth 1, for `*.athdf*`.
- `run*/rst/`, depth 2, for restart files, including `rank_*` subdirectories.
- `run*/bin/` and the run directory, depth 1, for history files.
- The run directory, depth 2, for likely input/configuration files such as
  `parfile.par`, `*.athinput`, `*.par`, and `*.in`.
- The full run tree for `*.npz` and `*.png`, because those analysis products and
  diagnostic figures are explicitly requested for staging.

The scan avoids broad recursive AthenaDF discovery. AthenaDF files are only
inventoried from `run*/bin/`.

## Orbital Time

The scanner parses `parfile.par` and inspects `src/pgen/dynbbh.cpp`. In the
current source, analytic circular binary runs set:

```text
bbh.om = pow(sep, -1.5)
```

For runs without `problem/use_traj_table = true`, the scanner uses:

```text
orbital_period = 2*pi*sep^1.5
orbit = simulation_time / orbital_period
```

The `sep`, inferred period, method, and confidence are written to
`run_summary.csv`. If `use_traj_table = true`, `sep` is missing, or the source
formula is not recognized, the scanner records the ambiguity and avoids claiming
an exact orbital mapping.

For AthenaDF and restart files, times are inferred from the final numeric token
in the filename multiplied by the corresponding output `dt` from `parfile.par`
when that output block can be identified. If only an index is available, the
selection record marks that limitation.

## Restart Grouping

Restart files are grouped by checkpoint identifier parsed from the filename.
For rank-local restarts such as:

```text
rst/rank_00000000/torus.00207.rst
rst/rank_00000001/torus.00207.rst
```

all files sharing checkpoint `00207` are selected together. The scanner reports
whether rank-local restarts were detected and whether group sizes look
ambiguous.

Restart selection chooses the nearest available checkpoint group to every
`--orbit-stride` target orbit, plus the latest available checkpoint group. The
selection metadata records target orbit, selected orbit/time, mismatch, and
group id.

## AthenaDF Classification

All `*.athdf*` candidates found under `run*/bin/` are classified.

- `slice`: output name or tokens contain `slice`; all are selected.
- `mhd_w_bcc`: selected nearest to every stride target orbit, plus latest.
- `torque_excluded`: output-name tokens contain `torque`; not selected.
- `am_excluded`: conservative output-name-aware angular momentum exclusion.
  The scanner only treats `am`, `angular`, or `angular_momentum` output-name
  tokens as angular-momentum outputs. It does not blindly reject every path that
  contains the letters `am`.
- `other_athdf`: discovered and summarized, but not selected by default.

Slice files are classified before torque/angular-momentum exclusions, so slice
outputs are not dropped because of unrelated torque or angular-momentum logic.

## NPZ And PNG Files

All `.npz` files under each run directory are selected and categorized as `npz`.
All `*.png` files under each run directory are selected and categorized as
`png`. Counts and byte totals appear in both `run_summary.csv` and
`scan_report.md`.

## Running A Limited Scan

Start with one or two runs:

```bash
python scripts/staging/scan_aurora_runs.py \
  --roots \
    /home/hzhu/scratch2/hzhu/acc/cbd/tilted_large \
    /home/hzhu/scratch2/hzhu/acc/bondi/adi \
    /home/hzhu/scratch2/hzhu/acc/bondi/cooling \
  --run-glob 'run*' \
  --output-dir staging_manifests \
  --orbit-stride 100 \
  --max-runs 2 \
  --dry-run
```

Regenerate the report from existing manifests:

```bash
python scripts/staging/report_aurora_scan.py staging_manifests
```

Use `--refresh-cache` when you want to rescan the filesystem instead of reusing
per-run raw inventory caches in `staging_manifests/raw_cache/`.

## Outputs

The scanner writes:

- `staging_manifests/raw_inventory.jsonl`
- `staging_manifests/classified_inventory.jsonl`
- `staging_manifests/selected_transfer_manifest.jsonl`
- `staging_manifests/selected_transfer_manifest.txt`
- `staging_manifests/run_summary.csv`
- `staging_manifests/scan_report.md`
- `staging_manifests/manifest_metadata.json`

`selected_transfer_manifest.txt` is intended for later `rsync --files-from`
use. For the requested trees, paths are written relative to:

```text
/home/hzhu/scratch2/hzhu/acc
```

Review `manifest_metadata.json` before using the text manifest for transfer.

## Known Limitations

- AthenaDF physical time is inferred from filename index and output `dt`; the
  scanner does not open AthenaDF files to read header time.
- Runs using trajectory tables need manual review of the orbit mapping.
- Full-tree `.npz` and `.png` discovery is required by the staging rules and may
  still be expensive on very large run trees.
- The first implementation writes line-oriented manifests and per-run raw
  caches. If a full scan produces manifests too large for comfortable local
  processing, the next refinement should switch the aggregation layer to SQLite.

## Suggested Transfer Next Step

After reviewing `scan_report.md` and pruning any oversized categories or runs,
perform a small transfer rehearsal with `rsync --dry-run --files-from` using
`selected_transfer_manifest.txt` and the source root recorded in
`manifest_metadata.json`. Do not start the actual transfer until the selected
size and category breakdown fit the destination budget.
