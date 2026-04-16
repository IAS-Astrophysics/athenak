# AthenaK STS Implementation Guide

## Status

This document now describes the landed AthenaK STS design after the original seven
planned implementation steps plus the first two post-v1 shearing-box/
orbital-advection extensions. It also records the remaining unsupported runtime scopes
and the follow-on extension work that still sits outside the current branch.

The branch now includes:

- shared `parabolic` enums, the metadata-only parabolic-process registry, and the host-side
  `RKL2` math helpers under `src/diffusion/`
- per-process `*_integrator` parsing and registration for viscosity, conduction, and
  Ohmic resistivity
- split mesh timestep bookkeeping with `dt_parabolic_sts`
- driver-owned STS controller scaffolding plus the parabolic stage-task shells
- Hydro-owned STS state, Hydro parabolic task assembly, and real `RKL2` execution for
  Hydro diffusion on both the standard path and the Hydro shearing-box/
  orbital-advection path
- MHD-owned STS state, MHD parabolic task assembly, and real `RKL2` execution for
  single-fluid MHD diffusion on both the standard path and the MHD shearing-box/
  orbital-advection path
- a maintained STS regression suite under `tst/test_suite/diffusion/`, truthful tracked
  fixtures under `inputs/tests/`, and exact-solution modes in `src/pgen/tests/diffusion.cpp`
- owner-aware runtime fencing that now rejects only the remaining unsupported scope:
  all `ion-neutral` STS runs

The target is AthenaK in this repository, not the older Athena++ tree. Athena++ is used
only as a reference for the RKL2 algorithm, stage-count rules, and operator-split sweep
structure that still fit AthenaK's current architecture.

## Design Goals

- Preserve AthenaK's existing ownership boundaries: the driver owns timestep orchestration,
  `Mesh` owns global timestep reduction, and Hydro/MHD own their own state updates.
- Support per-process opt-in to STS so users can run some parabolic operators with STS and
  leave others on the current explicit path in the same problem.
- Keep the first implementation small and AthenaK-like by reusing the current task lists,
  state arrays, boundary exchange wrappers, and diffusion kernels wherever practical.
- Make the extension path explicit so that a future parabolic process can join STS by
  registering itself and adding module-local hooks instead of redesigning the driver.
- Start with `RKL2` only. The guide should be good enough to implement without making any
  further architectural choices.

## Non-Goals

- Do not port Athena++'s monolithic `SuperTimeStepTaskList` structure into AthenaK.
- Do not promise Hall, ambipolar, or passive-scalar diffusion support in v1, because
  AthenaK does not currently implement those processes as maintained diffusion modules.
- Do not redesign the main explicit RK integrators, the AMR tree, or unrelated physics
  modules.
- Do not add a separate STS-only boundary-variable subset system. AthenaK should reuse its
  current boundary objects first, even if that over-communicates slightly relative to
  Athena++.

## AthenaK Baseline That The Design Must Respect

### Current time-integration control flow

AthenaK currently advances one cycle through `src/driver/driver.cpp`, with task lists
created in `src/mesh/meshblock_pack.cpp`:

```text
before_timeintegrator
for stage = 1..nexp_stages:
  before_stagen
  stagen
  after_stagen
after_timeintegrator
```

The task lists are generic containers in `src/tasklist/task_list.hpp`. Hydro and MHD
assemble their own task graphs in `src/hydro/hydro_tasks.cpp` and
`src/mhd/mhd_tasks.cpp`.
There is no existing Athena++-style global STS task list to extend.

### Current diffusion insertion points

Current diffusion is injected directly into the owning module's stage work:

- Hydro adds viscosity and conduction inside
  `Hydro::Fluxes()` in `src/hydro/hydro_tasks.cpp`
  after the hydro Riemann fluxes are computed.
- MHD adds viscosity, Ohmic energy flux, and conduction inside
  `MHD::Fluxes()` in `src/mhd/mhd_tasks.cpp`.
- MHD adds the Ohmic electric field inside
  `MHD::CornerE()` in `src/mhd/mhd_corner_e.cpp`, before constrained transport in
  `MHD::CT()` in `src/mhd/mhd_ct.cpp`.
- The current diffusion classes are `src/diffusion/viscosity.hpp`,
  `src/diffusion/conduction.hpp`, and `src/diffusion/resistivity.hpp`.

That means AthenaK already has the right low-level kernels. STS should change when those
kernels are applied and how their timestep limits are interpreted, not replace the kernels
themselves.

### Current timestep plumbing

The current timestep path is:

- `Hydro::NewTimeStep()` / `MHD::NewTimeStep()` compute the module-local hyperbolic limit.
- Conduction updates its own `dtnew` each cycle.
- Viscosity and resistivity expose their explicit parabolic limit through `dtnew`.
- `Mesh::NewTimeStep()` reduces everything into one global `dt` by taking the minimum over
  the module limit and every active diffusion object's `dtnew`.

Before STS, AthenaK had only one budget: if a diffusion process existed, it directly
constrained the main cycle `dt`. The current branch now carries the split budgets and, for
valid STS runs, uses:

- the cycle timestep that still constrains the explicit RK driver
- the explicit parabolic limit used only to size the STS stage count for STS-managed
  processes

The split belongs in `src/mesh/mesh.cpp`, because that is where AthenaK already combines
per-module timestep estimates.

## Athena++ Lessons To Carry Forward

The old Athena++ reference tree `athena++df` is useful for only four things:

1. STS is operator split around the main integrator.
2. `RKL2` is the right first method.
3. Stage count must be derived from the parabolic timestep ratio and forced odd.
4. `sts_max_dt_ratio` is a useful user-facing clamp on how aggressively STS stretches the
   explicit parabolic timestep.

Everything else should be ported selectively.

### Keep

- The `RKL2` stage-count rule from Athena++ `main.cpp`
- The Meyer-Balsara-Aslam `RKL2` coefficients from Athena++ `sts_task_list.cpp`
- The pre-sweep and post-sweep split around the existing main integrator
- The idea that STS-managed processes are removed from the normal explicit diffusion path

### Adapt

- Athena++ hides the half-step inside `0.5 * dt` factors. AthenaK should make the sweep
  duration explicit as `dt_sweep = 0.5 * dt_cycle` and use that directly everywhere.
- Athena++ uses one monolithic STS task list. AthenaK should keep module-owned tasks and
  add a separate parabolic stage loop in the driver.
- Athena++ needed special hydro/field coupling because field diffusion lived outside hydro.
  AthenaK's MHD module already owns both total energy and magnetic state, so resistive
  energy coupling stays within MHD.
- Athena++ had STS-specific boundary subsets. AthenaK should reuse existing
  `MeshBoundaryValuesCC/FC` paths first.

### Do Not Carry Over

- Hall diffusion references from Athena++ old docs: AthenaK does not have Hall support in
  its diffusion classes today.
- Passive-scalar diffusion references from Athena++: AthenaK currently advects passive
  scalars inside Hydro/MHD state vectors but does not implement scalar diffusion as a
  maintained subsystem.
- Misleading Athena++ example decks. The only old stock deck that actually enables STS by
  default is `inputs/mhd/athinput.linear_wave3d`, and that fact matters only as an
  algorithm reference, not as an AthenaK runtime example.

## Target Runtime Interface

### New `<time>` parameters

Add the following runtime parameters:

| Block | Parameter | Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| `time` | `sts_integrator` | `none`, `rkl2` | `none` | Global STS method selector |
| `time` | `sts_max_dt_ratio` | real | `-1.0` | Optional cap on `dt_cycle / dt_parabolic_min` |

### New per-process parameters

Add process-local selectors beside the existing coefficients:

| Block | Parameter | Values | Default |
| --- | --- | --- | --- |
| `hydro` | `viscosity_integrator` | `explicit`, `sts` | `explicit` |
| `hydro` | `conductivity_integrator` | `explicit`, `sts` | `explicit` |
| `mhd` | `viscosity_integrator` | `explicit`, `sts` | `explicit` |
| `mhd` | `conductivity_integrator` | `explicit`, `sts` | `explicit` |
| `mhd` | `ohmic_resistivity_integrator` | `explicit`, `sts` | `explicit` |

Future parabolic processes should follow the same naming rule: put
`<process>_integrator` next to the coefficient that activates the process.

### Startup validation rules

Implement these checks as fatal input errors:

- `*_integrator = sts` is invalid when `time/sts_integrator = none`.
- `time/sts_integrator = rkl2` is invalid if no active process selects `sts`.
- A process-local selector is invalid when the corresponding process is inactive after its
  normal activation rules are applied.
- Any selector value other than `explicit` or `sts` is invalid.
- `time/sts_max_dt_ratio` must be positive when set; keep `-1.0` as the only disabled
  value.

These checks should live near the current parameter parsing in the diffusion classes and
driver setup, not in problem generators.

## Target Timestep Design

### Keep the existing per-process explicit limits

Do not replace the current `dtnew` logic in the diffusion classes. AthenaK already has
the right explicit parabolic estimates:

- `Viscosity::Viscosity()` in `src/diffusion/viscosity.cpp`
- `Conduction::NewTimeStep()` in `src/diffusion/conduction.cpp`
- `Resistivity::Resistivity()` in `src/diffusion/resistivity.cpp`

The STS design should reinterpret those limits, not recompute them with a new formula.

### Split the global reduction in `Mesh::NewTimeStep()`

Modify `Mesh::NewTimeStep()` in `src/mesh/mesh.cpp` to compute two minima:

- `dt_cycle_candidate`: the timestep that still constrains the main explicit driver
- `dt_parabolic_min`: the smallest explicit parabolic limit among STS-managed processes

Use the same CFL-scaled quantity that AthenaK uses today when comparing limits. In other
words, if a diffusion process currently constrains the cycle through `cfl_no * dtnew`,
then `dt_parabolic_min` should also store `cfl_no * dtnew`.

The reduction rule should become:

```text
dt_cycle_candidate =
  min(hydro_or_mhd_hyperbolic_limits,
      source_limits,
      radiation/z4c/particle limits,
      every parabolic process still on explicit)

dt_parabolic_min =
  min(every active parabolic process on sts)

if sts is active and sts_max_dt_ratio > 0:
  dt_cycle_candidate = min(dt_cycle_candidate,
                           sts_max_dt_ratio * dt_parabolic_min)
```

Then:

- `Mesh::dt` stays the final cycle timestep.
- `Mesh` gains a new reduced field `dt_parabolic_sts` to hold `dt_parabolic_min`.
- The driver reads `pmesh->dt` and `pmesh->dt_parabolic_sts` at the start of each cycle.

Do not add a second, broad refactor of `Hydro::dtnew` or `MHD::dtnew`. The split belongs
at the global reduction site.

## RKL2 Controller Design

Add a small STS utility layer under `src/diffusion/`:

- `sts_rkl2.hpp`
- `sts_rkl2.cpp`

It should provide:

- `ComputeRKL2StageCount(Real dt_sweep, Real dt_parabolic_min)`
- `ComputeRKL2Coefficients(int stage, int nstages)`
- a compact `RKL2Coefficients` struct with `muj`, `nuj`, `muj_tilde`, `gammaj_tilde`

### Stage-count rule

For each pre/post sweep, use:

```text
dt_sweep = 0.5 * dt_cycle
nstages = floor(0.5 * (-1 + sqrt(9 + 16 * dt_sweep / dt_parabolic_min))) + 1
if nstages is even: ++nstages
```

This is the Athena++ `RKL2` rule, written in terms of the actual sweep duration instead
of hiding the half-step inside the update weights.

### Coefficients

Use the Athena++ `RKL2` coefficients:

```text
b_j = (j^2 + j - 2) / (2 j (j + 1))

mu_j          = ((2j - 1) / j) * (b_j / b_{j-1})
nu_j          = -((j - 1) / j) * (b_j / b_{j-2})
mu_tilde_j    = stage == 1 ? 4 b_j / (s^2 + s - 2)
                            : mu_j * 4 / (s^2 + s - 2)
gamma_tilde_j = stage == 1 ? 0
                            : -(1 - b_{j-1}) * mu_tilde_j
```

For startup stability, seed the small-stage `b_j` values exactly as Athena++ does:

- for stage `1` and `2`, set `b_j = b_{j-1} = b_{j-2} = 1/3`
- for stage `3`, set `b_{j-1} = b_{j-2} = 1/3`
- for stage `4`, set `b_{j-2} = 1/3`

### Ownership

Store per-cycle STS state on `Driver`, not on the diffusion objects. Add a compact
controller such as:

```text
enabled
sweep = pre | post
dt_sweep
dt_parabolic_min
nstages
current coefficients
```

This matches AthenaK's existing pattern where the driver owns explicit RK coefficients and
stage progression, while `Mesh` owns globally reduced timestep numbers.

## Parabolic Process Registry

Add a metadata-only registry in `src/diffusion/parabolic_process.hpp` and store it on
`MeshBlockPack`.

The registry entry should contain:

- process name
- owning subsystem (`hydro`, `mhd`, or future module tag)
- current integration mode (`explicit` or `sts`)
- a way to read the current explicit parabolic limit
- flags describing whether the process updates cell-centered conserved state,
  face-centered fields through CT, or both

Hydro and MHD should register their diffusion processes during construction, immediately
after creating `pvisc`, `pcond`, or `presist`.

The registry has one job in v1: give the driver and `Mesh::NewTimeStep()` a uniform view
of which parabolic processes are active and which ones are currently managed by STS.

It does **not** own task callbacks. Task ownership remains in Hydro/MHD, because that is
how AthenaK is already organized.

## Driver And Task-List Changes

### New task lists

Extend `MeshBlockPack::tl_map` with three new lists:

- `before_parabolic_stagen`
- `parabolic_stagen`
- `after_parabolic_stagen`

Use the existing naming convention with `stagen` so the new loop reads like AthenaK's
current explicit loop.

Step 4 lands these task-list names and the driver-side sweep loop. Steps 5 and 6 populate
them with Hydro-owned and MHD-owned parabolic tasks on the supported standard single-fluid
paths. `ion-neutral` runs still do not populate the parabolic task lists and remain
explicitly fenced from STS.

### New driver sequence

Update `Driver::Execute()` in `src/driver/driver.cpp` to run:

```text
if STS is active:
  run parabolic pre-sweep over dt_sweep = 0.5 * dt

run before_timeintegrator
run explicit stage loop
run after_timeintegrator

if STS is active:
  run parabolic post-sweep over dt_sweep = 0.5 * dt
```

The pre-sweep must happen before `before_timeintegrator`, not inside it. That keeps the
current `before_timeintegrator` contract intact. In particular, `MHD::SaveMHDState()`
will then snapshot the state that the main explicit integrator actually starts from.

Implement the parabolic sweep as:

```text
for sts_stage = 1..nstages:
  set driver STS coefficients for this stage
  ExecuteTaskList("before_parabolic_stagen", sts_stage)
  ExecuteTaskList("parabolic_stagen", sts_stage)
  ExecuteTaskList("after_parabolic_stagen", sts_stage)
```

Do not move `after_timeintegrator` tasks into the STS loop in v1. AthenaK currently does
not populate that list with diffusion-specific work, so no migration is needed. If future
modules start relying on `after_timeintegrator` for end-of-cycle state, they will need a
separate audit once STS exists.

## Hydro Integration Plan

### Current landed Hydro scope

Hydro STS is now enabled on both the standard path and the Hydro shearing-box/
orbital-advection path.

The landed Hydro extension keeps the explicit RK task order unchanged and instead gives
the parabolic loop its own sweep-aware wrappers:

- orbital advection is treated as a once-per-sweep remap that runs only on the final STS
  stage, with `remap_dt = dt_sweep`
- shearing-box x1 boundary exchange is treated as stage-local boundary maintenance and
  therefore runs on every STS stage
- the pre-sweep uses `shear_time = pmesh->time`, while the post-sweep uses
  `shear_time = pmesh->time + pmesh->dt`; each stage in the same sweep reuses that fixed
  time instead of inventing per-stage physical times

### Split the current diffusion tail

Add a Hydro-local helper that appends only the selected parabolic operators to the current
face flux array.

```text
AddSelectedDiffusionFluxes(selection)
```

with `selection` chosen from:

- `explicit_only`
- `sts_only`

Then:

- `Hydro::Fluxes()` keeps its current ideal Riemann-solver path, but now appends only
  `explicit_only` diffusion
- the new Hydro parabolic flux task clears `uflx` and then appends only `sts_only`
  diffusion

### Reuse the current live state arrays

During STS sweeps, the live state remains `u0` and `w0`. That allows the new parabolic
loop to reuse the existing boundary and primitive-recovery tasks:

- `SendU`
- `RecvU`
- `ApplyPhysicalBCs`
- `Prolongate`
- `ConToPrim`

Do not build a second live hydro state tree just for STS. The only new Hydro arrays
needed are the `RKL2` history and cached stage-1 operator result:

- `u_sts0`: state at the start of the sweep
- `u_sts1`: previous STS stage state
- `u_sts2`: second previous STS stage state
- `u_sts_rhs`: cached stage-1 operator result for `RKL2`

Reuse `uflx` as the stage-local scratch flux array for STS, because the parabolic sweep
runs outside the normal explicit stage loop.

### New Hydro parabolic tasks

Add Hydro-owned wrappers for the new parabolic task lists:

- clear `uflx`
- compute STS-managed diffusion fluxes into `uflx`
- reuse the current flux-correction wrappers for AMR correction on the standard path
- apply the `RKL2` weighted average/update to `u0`
- reuse the current state exchange, BC, prolongation, and primitive-recovery tasks
- recompute Hydro-local timestep estimates on the final post-sweep stage

The current Hydro STS task order is:

```text
before_parabolic_stagen:
  InitRecvParabolic

parabolic_stagen:
  ClearSTSFlux
  STSFluxes
  SendFlux / RecvFlux
  STSUpdate
  SendU_OA_Parabolic / RecvU_OA_Parabolic
  RestrictU
  SendU / RecvU
  SendU_Shr_Parabolic / RecvU_Shr_Parabolic
  ApplyPhysicalBCs
  Prolongate
  ConToPrim
  STSRefreshTimeStep  (only on post sweep, final stage)

after_parabolic_stagen:
  ClearSendParabolic
  ClearRecvParabolic
```

`InitRecvParabolic` and the parabolic clear wrappers gate orbital-advection MPI
setup/cleanup to the final STS stage of each sweep, because only that stage performs the
orbital remap. The shearing-box wrappers run every stage.

The STS Hydro update should only touch the conserved variables that the enrolled
processes actually modify:

- viscosity: momentum and total energy when the EOS evolves energy
- conduction: total energy only

There is no separate passive-scalar diffusion path to wire up in v1.

The post-sweep timestep refresh is required because the final Hydro STS sweep modifies
`u0` and `w0` after the explicit-stage `Hydro::NewTimeStep()` task has already run.
Without that refresh, the next `Mesh::NewTimeStep()` reduction would see stale Hydro
hyperbolic and conduction timestep estimates.

## MHD Integration

### Current landed MHD scope

MHD STS is now enabled on the single-fluid standard path and on the single-fluid
shearing-box/orbital-advection path.

The landed MHD extension mirrors the Hydro shearing/orbital rollout but has to do so for
both the cell-centered and face-centered state:

- orbital advection is treated as a once-per-sweep remap that runs only on the final STS
  stage, with both CC and FC remap paths using `remap_dt = dt_sweep`
- shearing-box x1 boundary exchange for both U and B is treated as stage-local boundary
  maintenance and therefore runs on every STS stage
- the pre-sweep uses `shear_time = pmesh->time`, while the post-sweep uses
  `shear_time = pmesh->time + pmesh->dt`; each stage in the same sweep reuses that fixed
  time instead of inventing per-stage physical times
- `ion-neutral` runs still do not populate the parabolic task lists and remain explicitly
  fenced from STS

### Split the diffusion work into reusable helpers

The landed MHD implementation uses two MHD-local helpers:

```text
AddSelectedDiffusionFluxes(selection)
AddSelectedDiffusionEMF(selection)
```

Then:

- `MHD::Fluxes()` keeps the ideal MHD flux path and appends only `explicit_only`
  parabolic terms
- `MHD::CornerE()` keeps the ideal EMF path and appends only `explicit_only`
  resistive EMFs
- the new parabolic MHD tasks call the helpers with `sts_only`

### Reuse the current live MHD state

As in Hydro, `u0`, `b0`, `w0`, and `bcc0` remain the live MHD state during STS sweeps.
That allows the parabolic MHD loop to reuse:

- `SendFlux` / `RecvFlux`
- `SendE` / `RecvE`
- `SendU` / `RecvU`
- `SendB` / `RecvB`
- `ApplyPhysicalBCs`
- `Prolongate`
- `ConToPrim`

Reuse `uflx` and `efld` as the stage-local parabolic scratch arrays. Add only the
history arrays required by `RKL2`:

- `u_sts0`, `u_sts1`, `u_sts2`, `u_sts_rhs` for cell-centered updates
- `b_sts0`, `b_sts1`, `b_sts2`, `b_sts_rhs` for face-centered magnetic updates

The MHD STS update now touches only the state that the enrolled processes modify:

- viscosity: momentum and total energy when energy is evolved
- conduction: total energy only
- Ohmic resistivity: face-centered magnetic field, plus total energy when the EOS evolves
  energy

### Current landed MHD parabolic task order

The current MHD STS task order is:

```text
before_parabolic_stagen:
  InitRecvParabolic

parabolic_stagen:
  ClearSTSFlux
  STSFluxes
  SendFlux
  RecvFlux
  ClearSTSEField
  STSEField
  EFieldSrc
  SendE
  RecvE
  STSUpdateU
  SendU_OA_Parabolic
  RecvU_OA_Parabolic
  STSUpdateB
  SendB_OA_Parabolic
  RecvB_OA_Parabolic
  RestrictU
  SendU
  RecvU
  SendU_Shr_Parabolic
  RecvU_Shr_Parabolic
  RestrictB
  SendB
  RecvB
  SendB_Shr_Parabolic
  RecvB_Shr_Parabolic
  ApplyPhysicalBCs
  Prolongate
  ConToPrim
  STSRefreshTimeStep  (only on post sweep, final stage)

after_parabolic_stagen:
  ClearSendParabolic
  ClearRecvParabolic
```

`SaveMHDState()` stays in `before_timeintegrator`, not in the parabolic loop, so it
captures the state that the explicit integrator actually starts from after the pre-sweep.

`InitRecvParabolic` and the parabolic clear wrappers gate orbital-advection MPI
setup/cleanup for both U and B to the final STS stage of each sweep, because only that
stage performs the orbital remap. The shearing-box U/B wrappers run every stage.

### Resistive coupling rule

Unlike Athena++, AthenaK does not need a separate hydro participant when Ohmic diffusion
is active in MHD. The MHD module already owns both total energy and magnetic state, so
the resistive energy flux and resistive EMF stay entirely within MHD-owned tasks.

## Extension Rule For Future Parabolic Processes

Any future parabolic process should be accepted into STS only if it can provide all of the
following:

1. A process-local runtime selector with values `explicit|sts`
2. An explicit parabolic timestep estimate comparable to the current `dtnew` convention
3. A module-local helper that can apply only that process to the current flux or EMF
   scratch arrays
4. A clear declaration of which state type it updates:
   cell-centered conserved variables, face-centered fields, or both
5. Any additional `RKL2` history arrays required for its update
6. Regression coverage that compares the new STS path to an explicit reference

If a future module cannot express its STS work in that shape, it should not be added to
the v1 STS path until the registry contract is deliberately widened.

## Implementation History

Implementation reached the following milestones in order:

1. Shared enums, the parabolic-process registry, and the `RKL2` utility files landed
   under `src/diffusion/`.
2. `Viscosity`, `Conduction`, and `Resistivity` now parse and store their integration
   mode.
3. `Mesh::NewTimeStep()` now computes `dt_cycle_candidate` and `dt_parabolic_sts`
   separately.
4. The driver-owned STS controller and the three new parabolic task lists landed in
   `MeshBlockPack`, and the temporary runtime fence moved from mesh parsing into the
   driver.
5. Hydro STS helpers, history arrays, parabolic tasks, the Hydro-only runtime fence
   narrowing, and `dt_cycle_candidate` activation landed.
6. MHD STS helpers, history arrays, and parabolic tasks landed for the standard
   single-fluid path.
7. The validation suite, truthful tracked fixtures, exact diffusion-test selector, MPI
   smoke coverage, runtime-fence regressions, and final guide cleanup landed under
   `tst/test_suite/diffusion/`, `inputs/tests/`, and `src/pgen/tests/diffusion.cpp`.

## Post-v1 Extension Milestones

1. Hydro shearing-box/orbital-advection STS support landed after the original seven-step
   rollout. The Hydro parabolic loop now uses sweep-aware wrappers so orbital advection
   remaps only on the final STS stage with `dt_sweep`, while shearing-box x1 exchange
   runs on every STS stage with fixed pre/post sweep times. Acceptance coverage lives in
   `tst/test_suite/sbox/test_sbox_hydroshwave_sts_mpicpu.py` and
   `tst/test_suite/sbox/test_sbox_hydro_orbital_sts_cpu.py`.
2. MHD shearing-box/orbital-advection STS support landed next. The MHD parabolic loop now
   uses the same sweep-aware pattern for both CC and FC state: orbital remaps run only on
   the final STS stage with `dt_sweep`, while shearing-box U/B exchange runs every STS
   stage with fixed pre/post sweep times. Acceptance coverage lives in
   `tst/test_suite/sbox/test_sbox_mhdshwave_sts_mpicpu.py` and
   `tst/test_suite/sbox/test_sbox_mhd_orbital_sts_cpu.py`.

## Validation Suite

The maintained STS coverage now lives in these files:

- `tst/test_suite/diffusion/test_sts_diffusion_cpu.py`
- `tst/test_suite/diffusion/test_sts_diffusion_mpicpu.py`
- `tst/test_suite/sbox/test_sbox_hydroshwave_sts_mpicpu.py`
- `tst/test_suite/sbox/test_sbox_hydro_orbital_sts_cpu.py`
- `tst/test_suite/sbox/test_sbox_mhdshwave_sts_mpicpu.py`
- `tst/test_suite/sbox/test_sbox_mhd_orbital_sts_cpu.py`
- `inputs/tests/viscosity.athinput`
- `inputs/tests/sts_conduction.athinput`
- `inputs/tests/sts_resistivity.athinput`
- `inputs/tests/sts_mhd_mixed_modes.athinput`
- `inputs/tests/sts_viscosity_smr.athinput`

The separate explicit-vs-STS benchmark package now lives outside this implementation guide:

- `tst/scripts/diffusion/benchmark_sts_diffusion.py` regenerates the CPU benchmark matrix
  and writes the CSV/TeX data products under `doc/data/sts_diffusion/`
- `vis/python/plot_sts_diffusion_benchmark.py` turns those CSV products into the figures
  under `doc/figures/sts_diffusion/`
- `doc/sts_diffusion_benchmark.tex` is the concise standalone note for sharing the
  accuracy/cost comparison with colleagues
- `src/pgen/tests/diffusion.cpp`

The landed CPU suite covers:

- exact Hydro viscosity STS against the Gaussian viscous-diffusion reference
- exact Hydro conduction using the dedicated conduction fixture, plus an explicit control
  run that proves the analytic setup independently of the STS path
- exact MHD resistive STS against the Gaussian Ohmic-diffusion reference in isothermal
  MHD
- mixed explicit/STS MHD timestep-budget behavior with explicit viscosity and STS
  resistivity
- a multilevel Hydro STS smoke on static refinement
- a Hydro orbital-advection STS smoke built from the tracked
  `inputs/shearing_box/hydro_orb_adv.athinput` template, with the test injecting the
  Hydro viscosity diffusion pgen and STS knobs at runtime
- a MHD orbital-advection STS smoke built from the tracked
  `inputs/shearing_box/mhd_orb_adv.athinput` template, with the test switching to the
  built-in isothermal resistive-diffusion pgen so the smoke runs in the default test
  build without relying on the out-of-tree `field_loop` problem generator
- full-run runtime-fence regressions for:
  `ion-neutral` STS, `sts_integrator = none` with an active STS process, and
  `sts_integrator = rkl2` with no active STS-managed process

The landed MPI suite adds:

- one narrow 4-rank resistive STS smoke using the tracked
  `inputs/tests/sts_resistivity.athinput` fixture
- one Hydro shearing-wave STS acceptance test built from `tst/inputs/hydro_shwave.athinput`
- one MHD shearing-wave STS acceptance test built from `tst/inputs/mhd_shwave.athinput`

The existing non-STS `tst/test_suite/sbox/test_sbox_hydroshwave_mpicpu.py` and
`tst/test_suite/sbox/test_sbox_mhdshwave_mpicpu.py` regressions remain the guards for
the explicit shearing paths.

Two validation details matter for future readers:

- The multilevel STS regression uses static refinement, not adaptive refinement, because
  the current AMR criteria expose density-like refinement variables and are a poor fit for
  the constant-density diffusion fixtures used here.
- The Hydro conduction fixture now has a truthful analytic deck and explicit-reference
  control. The current STS check is still best treated as a bounded exactness regression,
  not as the strongest convergence signal in the suite; if Hydro conduction STS is
  widened or retuned later, revisit those thresholds first.

## First Example Problems

Use only truthful AthenaK examples in the guide and tests.

### Exact and regression fixtures

- `inputs/tests/viscosity.athinput`: exact Hydro viscosity fixture; also the first Hydro
  STS exact regression input
- `inputs/tests/sts_conduction.athinput`: dedicated 1D thermal-diffusion fixture for the
  Hydro conduction exact and explicit-control regressions
- `inputs/tests/sts_resistivity.athinput`: exact MHD Ohmic-diffusion fixture, sized so
  the same tracked deck also serves the 4-rank MPI smoke
- `inputs/tests/sts_mhd_mixed_modes.athinput`: focused mixed explicit/STS MHD smoke with
  explicit viscosity and STS resistivity
- `inputs/tests/sts_viscosity_smr.athinput`: static-refinement Hydro STS smoke that
  exercises prolongation/restriction and multilevel communication

### Broader smoke problems

- `tst/inputs/hydro_shwave.athinput`: shearing-wave template used by the MPI Hydro STS
  acceptance test; pytest injects Hydro viscosity and STS settings at runtime
- `tst/inputs/mhd_shwave.athinput`: shearing-wave template used by the MPI MHD STS
  acceptance test; pytest injects resistivity and STS settings at runtime
- `inputs/shearing_box/hydro_orb_adv.athinput`: orbital-advection template used by the
  Hydro STS smoke; pytest injects the Hydro viscosity diffusion setup and STS settings at
  runtime
- `inputs/shearing_box/mhd_orb_adv.athinput`: orbital-advection template used by the MHD
  STS smoke; pytest injects an isothermal resistive-diffusion setup and STS settings at
  runtime
- `inputs/hydro/viscosity.athinput`: first Hydro STS smoke outside the pytest exact suite
- `inputs/mhd/resistivity.athinput`: first MHD resistive STS smoke outside the pytest
  exact suite

Do not reuse old Athena++ decks verbatim. Their parameter names and module ownership do
not match AthenaK closely enough to serve as user-facing examples.

## Maintainer Checklist

Before declaring AthenaK STS complete for a new process, verify all of the following:

- The process has a runtime selector and startup validation.
- Its explicit parabolic limit participates in `Mesh::NewTimeStep()` correctly.
- The normal explicit path skips the process when it is on STS.
- The parabolic task loop applies only the STS-managed contribution.
- `RKL2` history arrays are allocated only where they are truly needed.
- Boundary exchange, prolongation, and primitive recovery reuse the current AthenaK
  wrappers unless there is a demonstrated correctness issue.
- The new process has at least one regression and one example input.
- This guide is still accurate after implementation.
