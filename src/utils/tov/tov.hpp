#ifndef UTILS_TOV_TOV_HPP_
#define UTILS_TOV_TOV_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tov.hpp
//  \brief TOV solver with support for generic equations of state.

#include <math.h>

#include <type_traits>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tov_utils.hpp"

namespace tov {

// Useful container for physical parameters of star
class TOVStar {
 private:
  explicit TOVStar(ParameterInput* pin);

  template<class TOVEOS>
  void RHS(Real r, Real P_pt, Real m_pt, Real alp_pt, Real R_pt, TOVEOS& eos,
           Real& dP, Real& dm, Real& dalp, Real& dR) {
    // In our units, the equations take the form
    // dP/dr = -(e + P)/(1 - 2m/r)(m + 4\pi r^3 P)/r^2
    // dm/dr = 4\pi r^2 e
    // d\alpha dr = \alpha/(1 - 2m/r) (m + 4\pi r^3 P)/r^2
    // dR/dr = R/r (1 - 2m/r)^(-1/2)
    // Handle the case when r ~= 0 because the integrand is indeterminate.
    if (r < 1e-3*dr) {
      dP = 0.0;
      dm = 0.0;
      dalp = 0.0;
      dR = 1.0;
      return;
    }

    Real rho = eos.template GetRhoFromP<LocationTag::Host>(P_pt);
    Real e = eos.template GetEFromRho<LocationTag::Host>(rho);

    Real A = 1.0/(1.0 - 2.0*m_pt/r);
    Real B = (m_pt + 4.0*M_PI*r*r*r*P_pt)/SQR(r);
    dP = -(e + P_pt)*A * B;
    dm = 4.0*M_PI*SQR(r)*e;
    //dalp = alp_pt*A * B;
    dalp = A * B;
    dR = R_pt/r*sqrt(A);
  }

 public:
  template<class TOVEOS>
  static TOVStar ConstructTOV(ParameterInput* pin, TOVEOS& eos);

  KOKKOS_INLINE_FUNCTION
  Real FindSchwarzschildR(Real r_iso, Real mass) const {
    if (r_iso > R_edge_iso) {
      Real psi = 1.0 + mass/(2.*r_iso);
      return r_iso*psi*psi;
    }

    int idx = FindIsotropicIndex(r_iso);
    const auto &R_iso_l = R_iso.d_view;
    const auto &R_l = R.d_view;

    return Interpolate(r_iso, R_iso_l(idx), R_iso_l(idx+1), R_l(idx), R_l(idx+1));
  }

  KOKKOS_INLINE_FUNCTION
  int FindIsotropicIndex(Real r_iso) const {
    // Perform a bisection search to find the closest index to the requested isotropic
    // point.
    const auto &R_iso_l = R_iso.d_view;
    int lb = 0;
    int ub = n_r;
    int idx = lb;
    while (R_iso_l(lb+1) < r_iso) {
      idx = (lb + ub)/2;
      if (R_iso_l(idx) < r_iso) {
        lb = idx;
      } else {
        ub = idx;
      }
    }
    return lb;
  }

  template<class TOVEOS>
  KOKKOS_INLINE_FUNCTION
  void GetPrimitivesAtPoint(const TOVEOS& eos, Real r,
                            Real &rho, Real &p, Real &m, Real &alp) const;

  template<class TOVEOS>
  KOKKOS_INLINE_FUNCTION
  void GetPrimitivesAtIsoPoint(const TOVEOS& eos, Real r,
                               Real &rho, Real &p, Real &m, Real &alp) const;

  template<class TOVEOS>
  KOKKOS_INLINE_FUNCTION
  void GetPandRho(const TOVEOS& eos, Real r, Real &rho, Real &p) const;

  template<class TOVEOS>
  KOKKOS_INLINE_FUNCTION
  void GetPandRhoIso(const TOVEOS& eos, Real r, Real &rho, Real &p) const;

 public:
  Real rhoc;
  Real dfloor;
  Real pfloor;

  int npoints; // Number of points in arrays
  Real dr; // Radial spacing for integration
  DualArray1D<Real> R; // Array of radial coordinates
  DualArray1D<Real> R_iso; // Array of isotropic radial coordinates
  DualArray1D<Real> M; // Integrated mass, M(r)
  DualArray1D<Real> P; // Pressure, P(r)
  DualArray1D<Real> alpha; // Lapse, \alpha(r)

  Real R_edge; // Radius of star
  Real R_edge_iso; // Radius of star in isotropic coordinates
  Real M_edge; // Mass of star
  int n_r; // Point where pressure falls below floor.
};

template<class TOVEOS>
TOVStar TOVStar::ConstructTOV(ParameterInput *pin, TOVEOS& eos) {
  TOVStar tov{pin};

  tov.pfloor = eos.template GetPFromRho<LocationTag::Host>(tov.dfloor);

  Kokkos::realloc(tov.R, tov.npoints);
  Kokkos::realloc(tov.R_iso, tov.npoints);
  Kokkos::realloc(tov.M, tov.npoints);
  Kokkos::realloc(tov.P, tov.npoints);
  Kokkos::realloc(tov.alpha, tov.npoints);

  // Set aliases
  auto &R = tov.R.h_view;
  auto &R_iso = tov.R_iso.h_view;
  auto &M = tov.M.h_view;
  auto &P = tov.P.h_view;
  auto &alp = tov.alpha.h_view;
  int npoints = tov.npoints;
  Real dr = tov.dr;

  // Set initial data
  // FIXME: Assumes ideal gas for now!
  R(0) = 0.0;
  R_iso(0) = 0.0;
  M(0) = 0.0;
  P(0) = eos.template GetPFromRho<LocationTag::Host>(tov.rhoc);
  // FIXME: Assumes ideal gas!
  //P(0) = tov.kappa*pow(tov.rhoc, tov.gamma);
  //alp(0) = 1.0;
  alp(0) = 0.0;

  // Integrate outward using RK4
  tov.n_r = 0;
  for (int i = 0; i < npoints-1; i++) {
    Real r, P_pt, alp_pt, m_pt, R_pt;

    // First stage
    Real dP1, dm1, dalp1, dR1;
    r = i*dr;
    P_pt = P(i);
    alp_pt = alp(i);
    m_pt = M(i);
    R_pt = R_iso(i);
    tov.RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, dP1, dm1, dalp1, dR1);

    // Second stage
    Real dP2, dm2, dalp2, dR2;
    r = (i + 0.5)*dr;
    P_pt = fmax(P(i) + 0.5*dr*dP1,0.0);
    m_pt = M(i) + 0.5*dr*dm1;
    alp_pt = alp(i) + 0.5*dr*dalp1;
    R_pt = R_iso(i) + 0.5*dr*dR1;
    tov.RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, dP2, dm2, dalp2, dR2);

    // Third stage
    Real dP3, dm3, dalp3, dR3;
    P_pt = fmax(P(i) + 0.5*dr*dP2,0.0);
    m_pt = M(i) + 0.5*dr*dm2;
    alp_pt = alp(i) + 0.5*dr*dalp2;
    R_pt = R_iso(i) + 0.5*dr*dR2;
    tov.RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, dP3, dm3, dalp3, dR3);

    // Fourth stage
    Real dP4, dm4, dalp4, dR4;
    r = (i + 1)*dr;
    P_pt = fmax(P(i) + dr*dP3,0.0);
    m_pt = M(i) + dr*dm3;
    alp_pt = alp(i) + dr*dalp3;
    R_pt = R_iso(i) + dr*dR3;
    tov.RHS(r, P_pt, m_pt, alp_pt, R_pt, eos, dP4, dm4, dalp4, dR4);

    // Combine all the stages together
    R(i+1) = (i + 1)*dr;
    P(i+1) = P(i) + dr*(dP1 + 2.0*dP2 + 2.0*dP3 + dP4)/6.0;
    M(i+1) = M(i) + dr*(dm1 + 2.0*dm2 + 2.0*dm3 + dm4)/6.0;
    alp(i+1) = alp(i) + dr*(dalp1 + 2.0*dalp2 + 2.0*dalp3 + dalp4)/6.0;
    R_iso(i+1) = R_iso(i) + dr*(dR1 + 2.0*dR2 + 2.0*dR3 + dR4)/6.0;

    // If the pressure falls below zero, we've hit the edge of the star.
    if (P(i+1) <= 0.0 || P(i+1) <= tov.pfloor) {
      tov.n_r = i+1;
      break;
    }
  }

  if (tov.n_r == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "TOV solver failed to find the edge of the star." << std::endl
              << "Increase number of points, radial step, or rho_cut." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Now we can do a linear interpolation to estimate the actual edge of the star.
  int n_r = tov.n_r;
  tov.R_edge = Interpolate(tov.pfloor, P(n_r-1), P(n_r), R(n_r-1), R(n_r));
  tov.M_edge = Interpolate(tov.R_edge, R(n_r-1), R(n_r), M(n_r-1), M(n_r));

  // Replace the edges of the star.
  P(n_r) = tov.pfloor;
  M(n_r) = tov.M_edge;
  alp(n_r) = Interpolate(tov.R_edge, R(n_r-1), R(n_r), alp(n_r-1), alp(n_r));
  R(n_r) = tov.R_edge;
  R_iso(n_r) = Interpolate(tov.R_edge, R(n_r-1), R(n_r), R_iso(n_r-1), R_iso(n_r));

  for (int i = 0; i <= n_r; i++) {
    alp(i) = exp(alp(i));
  }

  // Rescale alpha so that it matches the Schwarzschild metric at the boundary.
  // We also need to rescale the isotropic radius to agree at the boundary.
  Real rs = 2.0*tov.M_edge;
  Real bound = sqrt(1.0 - rs/tov.R_edge);
  Real scale = bound/alp(n_r);
  tov.R_edge_iso = 0.5*(R(n_r) - M(n_r) + sqrt(R(n_r)*(R(n_r) - 2.0*M(n_r))));
  Real iso_scale = tov.R_edge_iso/R_iso(n_r);
  for (int i = 0; i <= n_r; i++) {
    alp(i) = alp(i)*scale;
    R_iso(i) = R_iso(i)*iso_scale;
  }

  // Print out details of the calculation
  if (global_variable::my_rank == 0) {
    std::cout << "\nTOV INITIAL DATA\n"
              << "----------------\n";
    std::cout << "Total points in buffer: " << tov.npoints << "\n";
    std::cout << "Radial step: " << tov.dr << "\n";
    std::cout << "Radius (Schwarzschild): " << tov.R_edge << "\n";
    std::cout << "Radius (Isotropic): " << tov.R_edge_iso << "\n";
    std::cout << "Mass: " << tov.M_edge << "\n\n";
  }

  // Sync the views to the GPU
  tov.R.template modify<HostMemSpace>();
  tov.R_iso.template modify<HostMemSpace>();
  tov.M.template modify<HostMemSpace>();
  tov.alpha.template modify<HostMemSpace>();
  tov.P.template modify<HostMemSpace>();

  tov.R.template sync<DevExeSpace>();
  tov.R_iso.template sync<DevExeSpace>();
  tov.M.template sync<DevExeSpace>();
  tov.alpha.template sync<DevExeSpace>();
  tov.P.template sync<DevExeSpace>();

  return tov;
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
void TOVStar::GetPrimitivesAtPoint(const TOVEOS& eos, Real r,
                                 Real &rho, Real &p, Real &m, Real &alp) const {
  // Check if we're past the edge of the star.
  // If so, we just return atmosphere with Schwarzschild.
  if (r >= R_edge) {
    rho = 0.0;
    p = 0.0;
    m = M_edge;
    alp = sqrt(1.0 - 2.0*m/r);
    return;
  }
  // Get the lower index for where our point must be located.
  int idx = static_cast<int>(r/dr);
  const auto &R_l = R.d_view;
  const auto &Ps_l = P.d_view;
  const auto &alps_l = alpha.d_view;
  const auto &Ms_l = M.d_view;
  // Interpolate to get the primitive.
  p = Interpolate(r, R_l(idx), R_l(idx+1), Ps_l(idx), Ps_l(idx+1));
  m = Interpolate(r, R_l(idx), R_l(idx+1), Ms_l(idx), Ms_l(idx+1));
  alp = Interpolate(r, R_l(idx), R_l(idx+1), alps_l(idx), alps_l(idx+1));
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
void TOVStar::GetPrimitivesAtIsoPoint(const TOVEOS& eos, Real r_iso,
                                    Real &rho, Real &p, Real &m, Real &alp) const {
  // Check if we're past the edge of the star.
  // If so, we just return atmosphere with Schwarzschild.
  if (r_iso >= R_edge_iso) {
    rho = 0.0;
    p = 0.0;
    m = M_edge;
    alp = (1. - m/(2.*r_iso))/(1. + m/(2.*r_iso));
    return;
  }
  // Because the isotropic coordinates are not evenly spaced, we need to search to find
  // the right index. We can set a lower bound because r_iso <= r, and then we choose the
  // edge of the star as an upper bound.
  const auto &R_iso_l = R_iso.d_view;
  int idx = FindIsotropicIndex(r_iso);
  const auto &Ps_l = P.d_view;
  const auto &alps_l = alpha.d_view;
  const auto &Ms_l = M.d_view;
  if (idx >= npoints || idx < 0) {
    printf("There's a problem with the index!\n" // NOLINT
           " idx = %d\n"
           " r_iso = %g\n"
           " dr = %g\n",idx,r_iso,dr);
  }
  // Interpolate to get the primitive.
  p = Interpolate(r_iso, R_iso_l(idx), R_iso_l(idx+1), Ps_l(idx), Ps_l(idx+1));
  m = Interpolate(r_iso, R_iso_l(idx), R_iso_l(idx+1), Ms_l(idx), Ms_l(idx+1));
  alp = Interpolate(r_iso, R_iso_l(idx), R_iso_l(idx+1), alps_l(idx), alps_l(idx+1));
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
  if (!isfinite(p)) {
    printf("There's a problem with p!\n"); // NOLINT
  }
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
void TOVStar::GetPandRho(const TOVEOS& eos, Real r, Real &rho, Real &p) const {
  if (r >= R_edge) {
    rho = 0.;
    p   = 0.;
    return;
  }
  // Get the lower index for where our point must be located.
  int idx = static_cast<int>(r/dr);
  const auto &R_l = R.d_view;
  const auto &Ps_l = P.d_view;
  // Interpolate to get the pressure
  p = Interpolate(r, R_l(idx), R_l(idx+1), Ps_l(idx), Ps_l(idx+1));
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}

template<class TOVEOS>
KOKKOS_INLINE_FUNCTION
void TOVStar::GetPandRhoIso(const TOVEOS& eos, Real r, Real &rho, Real &p) const {
  if (r >= R_edge_iso) {
    rho = 0.;
    p   = 0.;
    return;
  }
  // We need to search to find the right index because isotropic coordinates aren't
  // evenly spaced.
  int idx = FindIsotropicIndex(r);
  const auto R_iso_l = R_iso.d_view;
  const auto &Ps_l = P.d_view;
  p = Interpolate(r, R_iso_l(idx), R_iso_l(idx+1), Ps_l(idx), Ps_l(idx+1));
  rho = eos.template GetRhoFromP<LocationTag::Device>(p);
}


} // namespace tov

#endif // UTILS_TOV_TOV_HPP_
