# Turbulence Driver Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation Architecture](#implementation-architecture)
4. [AMR Handling](#amr-handling)
5. [Code Integration](#code-integration)
6. [Usage Guide](#usage-guide)
7. [Technical Details](#technical-details)

## Overview

The turbulence driver in AthenaK simulates stochastically forced turbulence by applying random forcing in Fourier space. It supports both compressible and incompressible (solenoidal) forcing modes, evolves forcing patterns via an Ornstein-Uhlenbeck (O-U) process, and maintains physical continuity across Adaptive Mesh Refinement (AMR) events.

### Key Features
- **Stochastic forcing** with controllable correlation time
- **Spectral control** via wavenumber selection
- **Cartesian basis**: Plane wave forcing in Fourier space
- **AMR-aware** with amplitude preservation across mesh refinement
- **Energy injection control** with constant power or constant acceleration modes
- **Spatial windowing** for localized turbulence driving

### File Locations
- **Header**: `src/srcterms/turb_driver.hpp`
- **Implementation**: `src/srcterms/turb_driver.cpp`
- **Problem generator**: `src/pgen/turb.cpp`

## Mathematical Foundations

### 1. Fourier Decomposition

The turbulence forcing field is constructed as a superposition of Fourier modes:

```
F(x,t) = Σₙ [aₙ(t) φₙ^R(x) - bₙ(t) φₙ^I(x)]
```

where:
- `aₙ(t)`, `bₙ(t)` are time-dependent mode amplitudes
- `φₙ^R(x)`, `φₙ^I(x)` are spatial basis functions (real/imaginary parts)
- `n` indexes the selected modes within the forcing band

### 2. Basis Functions

#### Cartesian Basis
For the standard Cartesian basis, the spatial functions are plane waves:

```
φₙ^R(x,y,z) = [cos(kₓx)cos(kᵧy) - sin(kₓx)sin(kᵧy)] cos(kzz)
              - [sin(kₓx)cos(kᵧy) + cos(kₓx)sin(kᵧy)] sin(kzz)

φₙ^I(x,y,z) = [cos(kᵧy)sin(kzz) + sin(kᵧy)cos(kzz)] cos(kₓx)
              + [cos(kᵧy)cos(kzz) - sin(kᵧy)sin(kzz)] sin(kₓx)
```

The wavenumbers are selected from:
```
k = 2π/L × n,  where n ∈ [nlow, nhigh]
```

### 3. Solenoidal Projection

To ensure incompressible (divergence-free) forcing:

```
F_sol = F - k(k·F)/k²
```

For the implementation, this is achieved through projection vectors that create perpendicular unit vectors in the plane orthogonal to k:
- In the x-y plane: `A₁ = kᵧ/√(kₓ² + kᵧ²)`, `B₁ = -kₓ/√(kₓ² + kᵧ²)`
- In the x-z plane: `A₂ = kz/√(kₓ² + kz²)`, `C₂ = -kₓ/√(kₓ² + kz²)`

### 4. Ornstein-Uhlenbeck Process

The mode amplitudes evolve according to:

```
daₙ/dt = -aₙ/τ + √(2/τ) σ ξₙ(t)
```

where:
- `τ` is the correlation time (`tcorr`)
- `σ²` determines the forcing amplitude
- `ξₙ(t)` is white noise

The discrete update (for timestep `dt`) is:

```
aₙ(t+dt) = f·aₙ(t) + g·ξₙ
```

where:
- `f = exp(-dt/τ)` (correlation factor)
- `g = σ√(1-f²)` (noise amplitude)

### 5. Energy Injection Rate

The forcing amplitude is normalized to achieve target energy injection rate `dE/dt`:

```
σ = √(dE/dt × V / Σₙ Pₙ)
```

where:
- `V` is the volume
- `Pₙ` is the power spectrum weight for mode `n`

## Implementation Architecture

### Class Structure

```cpp
class TurbulenceDriver {
  // Core data
  DvceArray5D<Real> force, force_tmp1, force_tmp2;  // [nmb][3][nk][nj][ni]
  DualArray2D<Real> aka, akb;                       // [3][nmodes] amplitudes
  
  // Basis functions
  DvceArray3D<Real> xcos, xsin;  // [nmb][nmodes][ni]
  DvceArray3D<Real> ycos, ysin;  // [nmb][nmodes][nj]
  DvceArray3D<Real> zcos, zsin;  // [nmb][nmodes][nk]
  
  // AMR tracking
  int current_nmb_;
  int last_nmb_created_, last_nmb_deleted_;
  
  // Parameters
  Real tcorr, dedt, kpeak;
  int nlow, nhigh, mode_count;
};
```

### Workflow

1. **Initialization** (`TurbulenceDriver` constructor)
   - Parse input parameters
   - Count and select modes within forcing band
   - Allocate arrays for current mesh configuration
   - Precompute basis functions

2. **Mode Evolution** (`InitializeModes` + `UpdateForcing`)
   - Generate new random amplitudes via O-U process
   - Apply spectral weighting (power law or parabolic)
   - Normalize to target energy injection rate
   - Compute forcing field from modes

3. **Force Application** (`AddForcing`)
   - Add forcing to momentum equations
   - Handle relativistic corrections if needed
   - Apply spatial windowing if configured

4. **AMR Adaptation** (`CheckResize` + `ResizeArrays`)
   - Detect mesh structure changes
   - Reallocate arrays for new mesh block count
   - **Preserve mode amplitudes** (critical!)
   - Recompute basis functions at new grid points

## AMR Handling

### The Challenge

When AMR refines or coarsens the mesh:
1. Number of mesh blocks changes
2. Physical locations of grid points change
3. Array dimensions need resizing
4. **Physics must remain continuous**

### The Solution

The key insight: **AMR is a numerical change, not a physical event**. The turbulence must continue evolving smoothly through the O-U process.

#### Detection Phase (`CheckResize`)
```cpp
if (nmb != current_nmb_ ||
    (pm->adaptive && pm->pmr != nullptr &&
     (pm->pmr->nmb_created != last_nmb_created_ || 
      pm->pmr->nmb_deleted != last_nmb_deleted_))) {
  ResizeArrays(nmb);
}
```

#### Preservation Phase (`ResizeArrays`)
1. **Reallocate arrays** for new mesh block count
2. **DO NOT call Initialize()** - this would reset the turbulence state!
3. **Preserve `aka_` and `akb_`** - these contain the O-U process memory
4. **Recompute basis functions** at new grid locations
5. **Reconstruct forcing** using preserved amplitudes

```cpp
// Critical: preserve mode amplitudes
auto aka_ = aka;  // These contain the turbulence "state"
auto akb_ = akb;  // Must NOT be reset!

// Recompute basis at new locations
par_for("xsin/xcos_resize", ..., KOKKOS_LAMBDA(int m, int n, int i) {
  Real x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
  xsin_(m,n,i) = sin(kx_mode_.d_view(n) * x1v);
  xcos_(m,n,i) = cos(kx_mode_.d_view(n) * x1v);
});

// Reconstruct forcing with preserved amplitudes
for (int n = 0; n < mode_count; n++) {
  par_for(..., KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Use EXISTING aka_, akb_ values
    force_tmp2_(m,dir,k,j,i) += aka_.d_view(dir,n)*forc_real 
                               - akb_.d_view(dir,n)*forc_imag;
  });
}
```

### Physical Continuity

This approach ensures:
- **No phase jumps** - forcing pattern continues smoothly
- **No energy spikes** - power injection remains constant
- **Correlation preserved** - O-U process memory maintained
- **Statistical properties** unchanged across AMR events

## Code Integration

### Task System Integration

The turbulence driver integrates with AthenaK's task-based execution model:

```cpp
// In operator split task list (before time integration)
void IncludeInitializeModesTask(TaskList *tl, TaskID start) {
  auto id_init   = tl->AddTask(&InitializeModes, this, start);
  auto id_resize = tl->AddTask(&CheckResize, this, id_init);
  auto id_update = tl->AddTask(&UpdateForcing, this, id_resize);
}

// In stage task list (during RK stages)
void IncludeAddForcingTask(TaskList *tl, TaskID start) {
  tl->InsertTask(&AddForcing, this, 
                 phydro->id.rkupdt,    // After RK update
                 phydro->id.srctrms);  // Before source terms
}
```

### MeshBlockPack Integration

The driver is owned by each `MeshBlockPack`:

```cpp
class MeshBlockPack {
  TurbulenceDriver *pturb;  // Turbulence driver for this pack
  
  void AddPhysics(ParameterInput *pin) {
    if (pin->GetOrAddBoolean("turb_driving", "turb_flag", false) > 0) {
      pturb = new TurbulenceDriver(this, pin);
    }
  }
};
```

### Coordinate System Support

The driver queries mesh block positions through the coordinate system:

```cpp
Real x1v = CellCenterX(i-is, nx1, mb_size.x1min, mb_size.x1max);
```

This automatically handles:
- Cartesian coordinates
- Cylindrical coordinates (with appropriate basis)
- Spherical coordinates

## Usage Guide

### Input Parameters

```ini
<turb_driving>
# Spectral range
nlow = 8          # Minimum wavenumber (in units of 2π/L)
nhigh = 16        # Maximum wavenumber
kpeak = 12.56     # Peak wavenumber for parabolic spectrum

# Energy injection
dedt = 0.001      # Energy injection rate
tcorr = 0.5       # Correlation time (0 = white noise)

# Spectral shape
spect_form = 1    # 1=parabolic, 2=power law
expo = 2.0        # Power law exponent (if spect_form=2)

# Forcing type
turb_flag = 2     # 1=decaying turbulence, 2=continuously driven
driving_type = 0  # 0=3D isotropic, 1=2D (x-y plane)
sol_fraction = 1.0 # Fraction of solenoidal modes (1.0 = incompressible)

# Random seed
rseed = -1        # -1 = time-based, >0 = fixed seed

# Spatial windowing (optional)
x_turb_scale_height = -1.0  # -1 = no windowing
x_turb_center = 0.0

# Tiled driving (optional)
tile_driving = false    # Enable sub-domain tiling for high-k forcing
tile_factor = 1         # Default tiling factor used for all directions
tile_nx = 1             # Number of tiles along x1 (must divide mesh nx1)
tile_ny = 1             # Number of tiles along x2 (must divide mesh nx2)
tile_nz = 1             # Number of tiles along x3 (must divide mesh nx3)
```

When `tile_driving = true`, the driver builds the random acceleration field on
sub-domains with lengths `L_i/tile_ni` and independently randomizes the Fourier
coefficients in each tile. The tiling counts must be integer factors of the root
grid dimensions so that the pattern can be tiled without discontinuities.

### Example Problem Generator

```cpp
void TurbulenceProblem(ParameterInput *pin, const bool restart) {
  if (!restart) {
    // Initialize uniform density and pressure
    par_for("pgen", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = 1.0;
      u0(m,IEN,k,j,i) = 1.0/(gamma-1.0);  // P = 1.0
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;  
      u0(m,IM3,k,j,i) = 0.0;
    });
  }
  // Turbulence driver handles forcing automatically
}
```

## Technical Details

### Memory Layout

Arrays use Kokkos Views with specific layouts:
- **Force arrays**: `[nmb][3][nk][nj][ni]` - mesh block, component, spatial
- **Amplitudes**: `[ntile][3][nmodes]` - tile, component, mode index
- **Basis functions**: `[nmb][nmodes][ncells]` - mesh block, mode, spatial

### Performance Considerations

1. **Precomputation**: Basis functions computed once per AMR event
2. **Parallelization**: All loops use Kokkos parallel patterns
3. **Memory efficiency**: Arrays sized to active mesh blocks only
4. **Cache optimization**: Inner loops over spatial indices

### Numerical Stability

1. **Amplitude limiting**: Forces capped at reasonable values
2. **Divergence cleaning**: Solenoidal projection maintains ∇·F = 0
3. **Energy conservation**: Normalization ensures consistent injection
4. **CFL safety**: Forces included in timestep calculation

### Debugging Tools

Enable verbose output:
```cpp
if (global_variable::my_rank == 0) {
  std::cout << "Turbulence: nmodes=" << mode_count 
            << " dedt=" << dedt << " tcorr=" << tcorr << std::endl;
}
```

Monitor divergence (for MHD):
```cpp
// Check div(B) before and after forcing
Real divB_max = ComputeMaxDivB();
if (divB_max > threshold) {
  std::cout << "Warning: div(B) = " << divB_max << std::endl;
}
```

## Advanced Topics

### Custom Power Spectra

Implement custom spectral shapes by modifying the power calculation:

```cpp
// In UpdateForcing()
if (spect_form == 3) {  // Custom spectrum
  // Example: Kolmogorov with exponential cutoff
  Real k = sqrt(kx*kx + ky*ky + kz*kz);
  power_spec = pow(k, -5.0/3.0) * exp(-k/k_cutoff);
}
```

### Anisotropic Forcing

Create anisotropic turbulence by mode selection:

```cpp
// Force only horizontal modes
if (driving_type == 2) {
  if (abs(kz) > epsilon) continue;  // Skip vertical modes
}
```

### Time-Dependent Forcing

Implement time-varying injection rates:

```cpp
Real dedt_current = dedt * (1.0 + 0.5*sin(2*PI*time/T_period));
```

## References

1. **Ornstein-Uhlenbeck Process**: Uhlenbeck & Ornstein (1930), Phys. Rev. 36, 823
2. **Turbulence Forcing Methods**: Eswaran & Pope (1988), Computers & Fluids 16, 257
3. **AMR with Turbulence**: Schmidt et al. (2009), A&A 494, 127

## Appendix: Common Issues and Solutions

### Issue: Turbulence dies after AMR
**Cause**: Mode amplitudes being reset  
**Solution**: Ensure `ResizeArrays()` preserves `aka_` and `akb_`

### Issue: Energy injection rate incorrect
**Cause**: Volume calculation or normalization error  
**Solution**: Check mesh size and normalization in `UpdateForcing()`

### Issue: Divergence errors in MHD
**Cause**: Non-solenoidal forcing  
**Solution**: Set `sol_fraction = 1.0` for incompressible forcing

### Issue: No turbulence development
**Cause**: Forces too weak or correlation time too long  
**Solution**: Increase `dedt` or decrease `tcorr`

---

*Document Version: 1.0*  
*Last Updated: 2024*  
*AthenaK Turbulence Driver Documentation*