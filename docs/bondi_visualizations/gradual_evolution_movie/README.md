# Gradual-cavity cyclic-zoom evolution

This sequence uses only saved fluid states from Aurora jobs 8925240 and 8925561,
with a smooth quintic attenuation from the unmodified Michel profile at r=a=32 M
to the density floor at the origin. Total binary mass is one, with equal masses,
and the Bondi radius is approximately 161.23 M. The magnetic field is zero.

The first job stopped cleanly at its 26-minute solver cutoff, at t=11812.91 M.
The continuation resumed its final checkpoint and reached t=35000 M, cycle 48130,
with PBS exit status zero after 50 minutes 7 seconds. Combined allocation use was
76 minutes 11 seconds on one node with twelve GPU tiles. All 24 rank reports
indicate successful exits without guard stops. Conservative sums of rank peak
RSS and tracked allocation peaks are below 20 GB for both jobs, including some
double counting; the requested ceiling was 30 GB.

The full physical residence-time schedule completed three outward/inward
cycles, returning to zone zero at 11471.34, 22942.68 and 34414.02 M, then started
a fourth outward sweep. The global elapsed time is 30.77 binary orbital periods.
The accumulated zone-zero dwell is only 200.04 M (about 0.176 binary periods),
including its initial dwell and the dwell after each return. Global elapsed
orbits therefore do not establish near-horizon relaxation or convergence.

The movie shows density and temperature on the **active** equatorial AMR mesh
in the corotating frame. Cyan crosses indicate the hole positions. Coarser
central pixels during outward sweeps are expected: frozen finer states in zoom
storage are not reconstructed in these regular fluid outputs. No intermediate
simulation states are interpolated. The time label is authoritative; checkpoint
snapshots introduce occasional shorter time intervals between frames.

The chain assembler checks every field for finite values, positive density and
pressure, strictly increasing unique times, and identical data for any duplicate
time. The saved manifest records source filenames and SHA256 hashes. The movie
renderer repeats the finite/positivity checks while drawing each frame. The
initial/final static slices and zoom history accompany the movie.

Reproduction (paths relative to repository root, using downloaded run folders):

```sh
python3 scripts/assemble_bondi_zoom_chain.py \
  --runs build-mac-zoom-opt/aurora-sane/gradual-chain/evolution-lowres-8925240 \
         build-mac-zoom-opt/aurora-sane/gradual-chain/evolution-lowres-8925561 \
  --out build-mac-zoom-opt/aurora-sane/gradual-assembled
python3 scripts/make_bondi_zoom_movie.py \
  --run build-mac-zoom-opt/aurora-sane/gradual-assembled \
  --out docs/bondi_visualizations/gradual_evolution_movie --fps 12 \
  --caption 'Gradual cavity seed; active AMR mesh only. Frozen finer states are not shown; local evolution exposure differs.' \
  --ffmpeg /path/to/ffmpeg
python3 scripts/plot_bondi_zoom_evolution.py \
  --run build-mac-zoom-opt/aurora-sane/gradual-assembled \
  --out docs/bondi_visualizations/gradual_evolution_movie --residence-scale 1
```
