//========================================================================================
// surface_grid.hpp — Parametric (θ,φ) → r(θ,φ) surface sampler + interpolator (Cartesian)
//----------------------------------------------------------------------------------------
#pragma once

#include "athena.hpp" // For Real, DualArray, etc.
#include <functional> // For std::function
#include <string>     // For std::string

//========================================================================================
// A uniformly sampled (θ,φ) surface with radius r(θ,φ) in Cartesian coords.
// Provides: x^i(θ,φ), finite-difference tangents e_θ,e_φ, quadrature weights,
// and a Lagrange (2*ng)^3 interpolator to pull grid fields onto the surface.
//========================================================================================
class MeshBlockPack;

class SphericalSurfaceGrid {
 public:
  using RFunc = std::function<Real(Real, Real)>;  // r(θ,φ) measured from `center_`

  // Constructor
  SphericalSurfaceGrid(MeshBlockPack* pack,
                       int ntheta, int nphi,
                       RFunc r_of_thph,
                       const std::string& name = "surf",
                       const Real* center = nullptr);

  // Translate the surface. (Rebuilds coordinates + interpolation only.)
  void SetCenter(const Real new_center[3]);

  // Recompute everything if r(θ,φ) changed externally.
  void RebuildFromRadius() { RebuildAll(); }

  // Interpolate a block of variables: val(m, var, k, j, i) → interp_vals(p, var)
  void InterpolateToSurface(int nvars, DvceArray5D<Real> &val);

  // Compute unnormalized surface covector dΣ_i = sqrtg * [ijk] * e_θ^j * e_φ^k * dθ * dφ
  // Note: For curved spacetime, the caller is responsible for ensuring that the passed
  // `sqrtg` is the determinant of the 3D spatial metric, sqrt(γ).
  void BuildSurfaceCovectors(const DualArray1D<Real>& sqrtg,
                             DualArray2D<Real>& dSigma) const;

  // Accessors
  int Npts() const { return npts; }
  const std::string& Label() const { return tag; }
  DualArray1D<Real>& Thetas() { return theta; }
  DualArray1D<Real>& Phis()   { return phi; }
  DualArray2D<Real>& Coords() { return coords; }
  DualArray2D<Real>& TanTheta() { return tan_th; }
  DualArray2D<Real>& TanPhi()   { return tan_ph; }
  DualArray1D<Real>& QuadWeights() { return weights; }
  DualArray2D<Real>& InterpVals()  { return interp_vals; }
  DualArray2D<int>&  InterpIndices() { return interp_indcs; }
  DualArray3D<Real>& InterpWeights() { return interp_wghts; }

 private:
  MeshBlockPack* pmy_pack;
  std::string tag;
  Real center_[3];
  int n_th, n_ph, npts;
  Real dth, dph;
  DualArray1D<Real> theta, phi, radius, weights;
  DualArray2D<Real> coords, tan_th, tan_ph;
  DualArray2D<int>   interp_indcs;
  DualArray3D<Real>  interp_wghts;
  DualArray2D<Real>  interp_vals;

  // --- Host-side setup functions ---
  void RebuildAll();
  void BuildCoordinates();
  void BuildQuadWeights();
  void BuildTangentsFD();
  void SetInterpolationIndices();
  void SetInterpolationWeights();
};