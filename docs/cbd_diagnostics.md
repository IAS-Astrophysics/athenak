# Circumbinary-disk diagnostics

`scripts/cbd_diagnostics.py` is the maintained command-line post-processor for
native AthenaK `.bin` volume dumps and `.hst` surface histories.  It discovers
all physics and mesh metadata from each binary header; it does not inspect C++
source or assume a local run path.

The script requires NumPy.  Run `python scripts/cbd_diagnostics.py --help` and
the subcommand help for the complete interface.

## Volume profiles and budgets

Write unsliced, ghost-zone-free native binary outputs for `mhd_w` (or
`mhd_w_bcc`) and `mhd_u_d` at the same cycles.  Optional matching outputs for
`angular_momentum` and `torque` add fluid/EM angular-momentum and torque
budgets.  For example:

```sh
python scripts/cbd_diagnostics.py volume \
  --primitive 'run/bin/torus.mhd_w.*.bin' \
  --conserved 'run/bin/torus.mhd_u_d.*.bin' \
  --angular 'run/bin/torus.angular_momentum.*.bin' \
  --torque 'run/bin/torus.torque.*.bin' \
  --rmin 20 --rmax 300 --dr 5 --output run/cbd_diagnostics
```

Quoted globs may select several times or run directories.  Dumps are matched
by the input metadata's `job/basename`, cycle, and run directory.  The summary
CSV is a time series suitable for direct run comparison, and one radial-profile
CSV is written per dump.  Default MPI binary output is already a global file;
when `single_file_per_rank=true`, pass the matching `rank_*` files and the tool
will combine their disjoint leaf MeshBlocks.  Duplicate blocks, different AMR
hierarchies, slices, and ghost-zone outputs fail clearly.

For DynGRMHD, `mhd_u_d` is the Valencia conserved density
`D = sqrt(gamma) rho W`.  Thus `sum(D d^3x)` is rest mass with the proper
spatial measure already represented, and no extra `sqrt(gamma)` is applied.
The `angular_momentum` and `torque` fields are likewise densitized and are
integrated with coordinate cell volume exactly once over the same radial and
density selection as the disk mass.  Native outputs contain only leaf blocks,
so AMR cells are neither resampled nor double counted.

The profile reports primitive density averaged with coordinate volume, rest
mass per radial bin, and—in cylindrical mode—rest mass divided by coordinate
annulus area as `surface_density_coordinate_area`.  The latter is a common
global CBD plotting convention, not a local proper-area scalar.  A genuinely
local proper-area surface density would require additional metric fields and
is deliberately not fabricated.  This field is emitted only for bins whose
complete annulus is enclosed by the Cartesian mesh; `full_annulus_coverage`
marks those bins and partial annuli remain `NaN`.  Primitive
`velx/vely/velz` are AthenaK's
`W v^i`; the tool does not reinterpret them as coordinate transport velocity.

`--rho-min` can remove a known atmosphere floor from disk totals.  The default
includes every finite cell in the selected radial range.  Binary metadata also
records `q`, dimensionless `a1/a2`, and trajectory-table use.  No deprecated
`adjust_mass*` convention is used.

## Radial and horizon flux histories

The history workflow extracts both the raw time series and tidy 17-component
surface records:

```sh
python scripts/cbd_diagnostics.py history \
  'run/torus.user.hst' --output run/cbd_diagnostics
```

`surface_fluxes.csv` includes radial `mdot`, energy, gas/EM linear and angular
momentum, magnetic flux, and proper area.  Moving `h1` and `h2` groups are
handled identically and provide the horizon accretion time series.  These
surface values are already evaluated with the current generalized GR surface
measure by AthenaK, so the script does not apply another area factor.  Repeated
history headers from restarts are supported.  A history without a complete
current-format surface group fails clearly unless `--raw-only-ok` is selected.
