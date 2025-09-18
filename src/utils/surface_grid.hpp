//========================================================================================
// surface_grid.hpp — Parametric (θ,φ) → r(θ,φ) surface sampler + interpolator (Cartesian)
//----------------------------------------------------------------------------------------
#pragma once

#include "athena.hpp" // For Real, DualArray, etc.
#include <functional> // For std::function
#include <string>     // For std::string

//========================================================================================
// A uniformly sampled (θ,φ) surface with radius r(θ,φ) in Cartesian coords.
// Provides: x^i(θ,φ), tangents e_θ,e_φ, interpolated metric g_ij, induced metric
// gamma_ab, and the proper area element dA.
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

  // Interpolates variables from source_array[start_index...end_index-1]
  DualArray2D<Real> InterpolateToSurface(const DvceArray5D<Real> &source_array,
                                           int start_index, int end_index);

  // Interpolate the 6 components of the ADM spatial metric to the surface.
  // This now automatically triggers the calculation of derived geometric quantities.
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

  // Accessor for the 3D metric (g_ij) interpolated to the surface
  DualArray2D<Real>& Metric() { return g_dd_surf_; }
  // Accessor for the 3 unique components of the 2D induced metric (gamma_ab)
  DualArray2D<Real>& InducedMetric() { return gamma_dd_surf_; }
  // Accessor for the scalar proper area element (dA)
  DualArray1D<Real>& ProperAreaElement() { return proper_dA_; }


 private:
  MeshBlockPack* pmy_pack;
  std::string tag;
  Real center_[3];
  int n_th, n_ph, npts;
  Real dth, dph;
  bool metric_is_flat_;

  // Core geometry
  DualArray1D<Real> theta, phi, radius, weights; // weights is now just dth*dph
  DualArray2D<Real> coords, tan_th, tan_ph;

  // Interpolation data
  DualArray2D<int>   interp_indcs;
  DualArray3D<Real>  interp_wghts;

  // Metric data
  DualArray2D<Real> g_dd_surf_; // 6 components of metric g_ij on surface points
  DualArray2D<Real> gamma_dd_surf_;   // 3 components of induced 2D metric gamma_ab
  DualArray1D<Real> proper_dA_;       // Scalar proper area element dA

  // --- Host-side setup functions ---
  void RebuildAll();
  void InitializeFlatMetric();
  void BuildCoordinates();
  void BuildQuadWeights();
  void BuildTangentsFD();
  void SetInterpolationIndices();
  void SetInterpolationWeights();
  
  // Calculates gamma_ab and proper_dA_ from g_ij and tangents
  void CalculateDerivedGeometry();
};