# Turbulence Driving in AthenaK

This directory contains source term implementations for AthenaK, including the turbulence driving method. The turbulence driver (`turb_driver.hpp/cpp`) implements Cartesian Fourier driving for simulating turbulence in computational domains.

## Cartesian Fourier Driving

### Mathematical Foundation

The turbulence driver applies a stochastic force field constructed from a superposition of plane waves:

```
F(x,t) = ∑_k A_k(t) exp(i k·x)
```

where:
- `k` is the wavevector in Cartesian coordinates
- `A_k(t)` is the time-dependent complex amplitude
- The sum is over all modes satisfying `k_low ≤ |k| ≤ k_high`

### Implementation Details

1. **Mode Selection**: Modes are selected within a spherical shell in k-space:
   - Integer wavevectors: `k = (k_x, k_y, k_z)` where each component is an integer
   - Constraint: `k_low² ≤ k_x² + k_y² + k_z² ≤ k_high²`

2. **Force Computation**: For each mode, the force at position `x` is:
   ```
   F_mode = amplitude × [cos(k·x) + i sin(k·x)]
   ```
   
3. **Amplitude Evolution**: Amplitudes follow an Ornstein-Uhlenbeck process:
   ```
   dA_k/dt = -A_k/t_corr + σ dW/dt
   ```
   where `t_corr` is the correlation time and `dW` is a Wiener process.

4. **Key Features**:
   - Maintains solenoidal (divergence-free) forcing when configured
   - Precomputes sine/cosine values for efficiency
   - Uses global coordinates to ensure continuity across AMR boundaries

## Configuration Parameters

### Common Parameters
- `turb_flag`: 1=decaying turbulence, 2=continuously driven
- `dedt`: Energy injection rate (energy/time/volume)
- `tcorr`: Correlation time for amplitude evolution
- `rseed`: Random seed for reproducibility (-1 for time-based)
- `sol_fraction`: Fraction of solenoidal modes (1.0 = incompressible)

### Spectral Parameters
- `nlow`, `nhigh`: Minimum and maximum |k| for mode selection
- `spect_form`: 1=parabolic spectrum, 2=power law
- `kpeak`: Peak wavenumber for parabolic spectrum
- `expo`: Power law exponent (if spect_form=2)

### Spatial Windowing (optional)
- `x_turb_scale_height`: Gaussian scale height in x-direction (-1 = no windowing)
- `x_turb_center`: Center of Gaussian window in x-direction
- (Similarly for y and z directions)

## Algorithm Flow

1. **Initialization**:
   - Select modes based on configuration
   - Precompute basis functions (sin/cos arrays)
   - Initialize random amplitudes

2. **Update** (every `dt_turb_update`):
   - Evolve amplitudes using Ornstein-Uhlenbeck process
   - Ensure desired power spectrum
   - Remove net momentum to maintain momentum conservation

3. **Force Application** (every timestep):
   - Compute force field at each cell center
   - Add to momentum equation source terms
   - Add corresponding energy source term

## AMR Considerations

The turbulence driver handles Adaptive Mesh Refinement (AMR) by:
- **Preserving mode amplitudes** during mesh refinement (critical!)
- Using global coordinates for basis function evaluation
- Ensuring phase continuity across refinement boundaries
- Computing forces independently on each MeshBlock using global position
- Never resetting the turbulence state during AMR events

The key insight is that AMR is a numerical change, not a physical event. The Ornstein-Uhlenbeck process must continue smoothly through refinement.

## Performance Notes

- Computational cost: O(N_cells × N_modes) per update
- Memory usage scales with number of modes
- Benefits from precomputation and vectorization
- Sine/cosine values are cached for efficiency

## References

- Cartesian driving: Stone et al. (1998), Mac Low (1999)
- Ornstein-Uhlenbeck process: Uhlenbeck & Ornstein (1930), Phys. Rev. 36, 823
- AMR with turbulence: Schmidt et al. (2009), A&A 494, 127
