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

  // <--- MODIFIED: Signature changed to accept start and end indices
  // Interpolates variables from source_array[start_index...end_index-1]
  DualArray2D<Real> InterpolateToSurface(const DvceArray5D<Real> &source_array,
                                           int start_index, int end_index);

  // Interpolate the 6 components of the ADM spatial metric to the surface
  void InterpolateMetric();

  // Compute surface covector dΣ_i. Uses interpolated metric if available, otherwise flat.
  void BuildSurfaceCovectors(DualArray2D<Real>& dSigma) const;

  // Accessors
  int Npts() const { return npts; }
  const std::string& Label() const { return tag; }
  DualArray1D<Real>& Thetas() { return theta; }
  DualArray1D<Real>& Phis()   { return phi; }
  DualArray2D<Real>& Coords() { return coords; }
  DualArray2D<Real>& TanTheta() { return tan_th; }
  DualArray2D<Real>& TanPhi()   { return tan_ph; }
  DualArray1D<Real>& QuadWeights() { return weights; }
  DualArray2D<int>&  InterpIndices() { return interp_indcs; }
  DualArray3D<Real>& InterpWeights() { return interp_wghts; }
  DualArray2D<Real>& Metric() { return g_dd_surf_; }

 private:
  MeshBlockPack* pmy_pack;
  std::string tag;
  Real center_[3];
  int n_th, n_ph, npts;
  Real dth, dph;
  bool metric_is_flat_;

  // Core geometry
  DualArray1D<Real> theta, phi, radius, weights;
  DualArray2D<Real> coords, tan_th, tan_ph;

  // Interpolation data
  DualArray2D<int>   interp_indcs;
  DualArray3D<Real>  interp_wghts;

  // Metric data
  DualArray2D<Real> g_dd_surf_; // 6 components of metric g_ij on surface points

  // --- Host-side setup functions ---
  void RebuildAll();
  void InitializeFlatMetric();
  void BuildCoordinates();
  void BuildQuadWeights();
  void BuildTangentsFD();
  void SetInterpolationIndices();
  void SetInterpolationWeights();
};