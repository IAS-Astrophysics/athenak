# Production Runs on TACC Vista — Tilted Magnetospheric Accretion (`tilted_magsph`)

Self-contained instructions to build AthenaK on TACC Vista, run a short timing
pre-test, and then launch the production runs. Written so an agent (Cursor /
Claude) with **no prior context** can execute it end to end. Target machine:
<https://tacc.utexas.edu/systems/vista/> (Grace-Hopper `gh` partition, 1x NVIDIA
H200 / 96 GiB HBM3 per node, NDR InfiniBand, `ibrun` MPI launcher).

> TL;DR: build the `magsph-testing` branch, run `inputs/mhd/tilted_magsph.athinput`
> on **64 `gh` nodes** (1 GPU/node — exactly the queue cap, no exception needed).
> Two production runs are defined below (Run A: spin+dipole tilted 45 deg; Run B:
> dipole tilted 45 deg, spin aligned). Each takes **~7.2 days** to 30 T0 in **~8
> segments** of a 24 h queue via auto-resubmit, and ~3 TiB of storage. Diagnostic
> outputs add <0.6% to runtime.

---

## 0. What you are running

- **Problem:** 3D MHD tilted magnetospheric accretion (`disk-magnetosphere` pgen).
- **Grid (`cand20`):** root 240^3 over [-48,48]^3, meshblock 40x40x20, static AMR to
  physical level 7. **7432 meshblocks, 2.378e8 cells, finest dx = 0.003125 (32 cells/R*).**
- **Base input:** `athenak/inputs/mhd/tilted_magsph.athinput` (on `magsph-testing`).
- **End time:** `tlim = 188.5` code units = 30 T0 (T0 = 2*pi/Omega0 = 6.2832).
- **Outputs:** bin (full cube, 0.1 T0), hst (history), rst (restart, 1 T0),
  3x geosph, sphshell, azavg. All already configured in the input file.

The two production runs differ **only** in the stellar tilt angles in `<problem>`
(angles in radians; 45 deg = 0.785398). The star spins in both (`origid = 2.8`):

| Run | Name | thetab (dipole) | thetaw (spin) | Notes |
|-----|------|-----------------|---------------|-------|
| **A** | spin45_dip45 | 0.785398 (45 deg) | 0.785398 (45 deg) | spin AND dipole tilted; this is the file default |
| **B** | spin0_dip45  | 0.785398 (45 deg) | 0.0 (aligned)      | dipole tilted, stellar spin aligned with disk |

---

## 1. Get the code (relevant branch on GitHub)

```bash
git clone git@github.com:IAS-Astrophysics/athenak.git
cd athenak
git checkout magsph-testing
git submodule update --init --recursive   # pulls the pinned Kokkos
ls inputs/mhd/tilted_magsph.athinput       # confirm the production input is present
```

---

## 2. Build on Vista (`gh` / Grace-Hopper)

Vista is ARM (Grace) + Hopper GPU. Build with the NVHPC toolchain, CUDA, and HPC-X
MPI, targeting `HOPPER90`. Reference build that produced the scaling results:
Kokkos 4.7.02, NVHPC 26.1, CUDA 13, HPC-X 2.25.1, `nvc++` host compiler via
`nvcc_wrapper`, async CUDA malloc OFF.

```bash
module purge
module load nvidia        # NVHPC compilers (nvc++) + CUDA  (use `module spider` to confirm versions)
module load cuda
module load openmpi       # HPC-X based MPI on Vista (provides ibrun)

cd athenak
cmake -S . -B build_vista \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DKokkos_ARCH_NEOVERSEV2=ON \
  -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
  -DCMAKE_CXX_COMPILER=$(pwd)/kokkos/bin/nvcc_wrapper \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROBLEM=disk-magnetosphere
cmake --build build_vista -j 16 --target athena
```

Yields `build_vista/src/athena`. **Sanity-check the grid:**

```bash
./build_vista/src/athena -i inputs/mhd/tilted_magsph.athinput -m
# Expect: "Total number of MeshBlocks = 7432" and physical level 7.
```

---

## 3. GPU count, memory, and load balance

- **Use 64 `gh` nodes = 64 GPUs (1 rank/node).** This is exactly Vista's standard
  `gh` queue cap — **no allocation exception required.**
- Total VRAM ~425 GiB. At 64 GPUs that is ~116 blocks/GPU, ~6.6 GiB/GPU — memory is
  well-saturated and far under the 96 GiB/GPU limit.
- 7432 = 2^3 x 929 (929 prime), so 64 does **not** divide it evenly. The AMR load
  balancer splits it 8 nodes x 117 + 56 nodes x 116 -> **99.9% efficiency**
  (negligible). No action needed.

---

## 4. STEP A — 200-cycle timing pre-test (do this first, once)

Pins down the real on-Vista throughput at 64 GPUs before committing to the full runs.

1. Make a no-output, 200-cycle copy (tilt angles do not matter for timing):

```bash
cp inputs/mhd/tilted_magsph.athinput timing.athinput
# In timing.athinput:
#   - set  nlim = 200   and  tlim = 1e10   (terminate on cycle limit)
#   - delete/comment the <output1>..<output8> blocks (no file output)
```

2. `timing.slurm` (64 nodes, ibrun):

```bash
#!/bin/bash
#SBATCH -J magsph_timing
#SBATCH -p gh
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -t 00:30:00
#SBATCH -o timing.%j.out
#SBATCH -e timing.%j.err
#SBATCH -A <YOUR_ALLOCATION>

module purge
module load nvidia cuda openmpi
ibrun ./build_vista/src/athena -i timing.athinput -d ./timing_run
```

```bash
mkdir -p timing_run && sbatch timing.slurm
```

3. From the end of `timing.*.out` read `zone-cycles/cpu_second` (ZCPS) and the steady
   per-cycle wall time (difference between consecutive late `elapsed=... cycle=N`
   lines — cleaner than the run-averaged ZCPS for a short run).

4. Refine the full-run estimate:

```
N_cycles = 30 T0 / 4.11e-6 T0 ≈ 7.30e6
walltime_integration(hr) = N_zones * N_cycles / (3600 * ZCPS)
                         = 2.378e8 * 7.30e6 / (3600 * ZCPS)
# or equivalently = 7.30e6 * (steady s/cycle) / 3600
```
   Expected ZCPS(64) ~ 2.81e9 (extrapolated from the benchmarked 54-GPU point
   2.37e9). If measured value differs, scale the days/segments below proportionally.

---

## 5. STEP B — production runs

The base `inputs/mhd/tilted_magsph.athinput` is production-ready (`tlim = 188.5`,
`nlim = -1`, all outputs on) and defaults to **Run A** (spin+dipole tilted 45 deg).
Set up each run in its own directory with its own input copy and basename so the
two runs never collide.

### Prepare the two inputs

```bash
# Run A: spin + dipole both tilted 45 deg (file default)
cp inputs/mhd/tilted_magsph.athinput tilted_spin45_dip45.athinput
sed -i 's/^basename .*/basename  = tilted_spin45_dip45/' tilted_spin45_dip45.athinput
# (thetaw = thetab = 0.785398 already)

# Run B: dipole tilted 45 deg, stellar spin aligned
cp inputs/mhd/tilted_magsph.athinput tilted_spin0_dip45.athinput
sed -i 's/^basename .*/basename  = tilted_spin0_dip45/' tilted_spin0_dip45.athinput
sed -i 's/^thetaw  =.*/thetaw  = 0.0        # aligned stellar spin/' tilted_spin0_dip45.athinput
# (thetab stays 0.785398)
```

Verify each: `grep -E "basename|thetaw|thetab" tilted_spin*.athinput`.

### Generic production SLURM script (`run_production.sh`)

Use one script per run via the `RUN` variable (set it to `spin45_dip45` or
`spin0_dip45`). Restart files are written every 1 T0, so a killed segment loses at
most ~1 T0. The script restarts from the newest `.rst` and resubmits until `tlim`.

```bash
#!/bin/bash
#SBATCH -J tilted_magsph
#SBATCH -p gh
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -A <YOUR_ALLOCATION>
#SBATCH --open-mode=append

module purge
module load nvidia cuda openmpi

RUN=spin45_dip45                       # <-- set to spin45_dip45 OR spin0_dip45
REPO=$PWD
INPUT=$REPO/tilted_${RUN}.athinput
WORKDIR=$REPO/run_${RUN}
STOP=23:50:00                          # stop+flush a restart before the 24h wall
mkdir -p $WORKDIR/rst

RST="$(ls -t $WORKDIR/rst/*.rst 2>/dev/null | head -1)"
if [[ -n "$RST" ]]; then
  echo "Restarting from $RST"
  ibrun $REPO/build_vista/src/athena -r "$RST" -d $WORKDIR -t $STOP
else
  echo "Starting new run $RUN"
  ibrun $REPO/build_vista/src/athena -i $INPUT -d $WORKDIR -t $STOP
fi

# Resubmit unless the physical time limit (tlim = 30 T0) was reached.
if tail -n 20 ${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out | grep -q "Terminating on time limit"; then
  echo "Reached tlim (30 T0) for $RUN — done."
else
  echo "Wall/segment limit — resubmitting $RUN."
  sbatch --job-name=$SLURM_JOB_NAME run_production.sh
fi
```

Launch each run (edit `RUN=` between submissions, or keep two copies of the script):

```bash
# Run A
sed -i 's/^RUN=.*/RUN=spin45_dip45/' run_production.sh
sbatch --job-name=tilted_spin45_dip45 run_production.sh

# Run B
sed -i 's/^RUN=.*/RUN=spin0_dip45/' run_production.sh
sbatch --job-name=tilted_spin0_dip45 run_production.sh
```

Notes:
- `-t 23:50:00` makes AthenaK stop and flush a restart before the 24 h wall; the
  script then chains segments automatically until `tlim`.
- If you see MPI HCOLL multicast warnings (`Failed to setup rcache`,
  `MCAST: Error initializing vmc context`) and degraded speed, add before `ibrun`:
  `export OMPI_MCA_coll_hcoll_enable=0` and `export HCOLL_ENABLE_MCAST=0`.
- Check the H200 single-job queue policy on Vista; if only one `gh` job runs at a
  time for your allocation, launch Run B after Run A finishes (or stagger them).

---

## 6. Expected performance, cost, and storage (64 GPUs, 30 T0, per run)

| Quantity | Value |
|----------|-------|
| Throughput (zcps) | ~2.81e9 (extrapolated; confirm with Step A) |
| Integration walltime | ~172 hr ≈ **7.2 days** |
| Output overhead | ~0.9 hr (**<0.6%** of runtime) |
| Total walltime | ~173 hr ≈ **7.2 days** |
| 24 h segments | **~8** (≈7 resubmissions) |
| VRAM / GPU | ~6.6 GiB (of 96) |
| Load-balance efficiency | ~99.9% |
| Storage / run | **~3.0 TiB** (bin 301 x 7.1 GiB = 2.1 TiB; rst 31 x 29.5 GiB = 0.9 TiB) |

Two runs total ~14.4 days of walltime and ~6 TiB of storage.

---

## 7. Monitoring / diagnostics
- Progress: `tail -f run_<RUN>/../tilted_<RUN>.<id>.out` — `elapsed= cycle= time= dt=` per cycle.
- Restarts accumulate in `run_<RUN>/rst/`; data dumps in `run_<RUN>/bin/` etc.
- End-of-segment summary prints `zone-cycles/cpu_second`.
- Mesh layout: re-run with `-m` (Section 2) any time to dump `mesh_structure.dat`.

## 8. Checklist (per run)
- [ ] `magsph-testing` checked out, submodules updated
- [ ] `build_vista/src/athena` built; `-m` shows 7432 blocks / level 7
- [ ] Allocation set (`-A`); 64 nodes (within `gh` cap, no exception)
- [ ] Step A pre-test run once; ZCPS read; walltime/segments refined
- [ ] Input prepared with correct `basename`, `thetaw`, `thetab`
- [ ] Production script submitted; restart + resubmission verified after segment 1
