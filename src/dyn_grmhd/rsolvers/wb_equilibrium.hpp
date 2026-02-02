#ifndef DYN_GRMHD_RSOLVERS_WB_EQUILIBRIUM_HPP_
#define DYN_GRMHD_RSOLVERS_WB_EQUILIBRIUM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file wb_equilibrium.hpp
//! \brief Functions to calculate equilibrium for well-balancing

#include <math.h>

#include "coordinates/cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/geom_math.hpp"
#include "flux_dyn_grmhd.hpp"

namespace dyngr {

//----------------------------------------------------------------------------------------
//! \fn void PressureEquilibrium
//! \brief inline function for well-balancing pressure at the L/R interfaces of a cell
//! interpolated values at L/R edges of cell i, that is ql(i+1) and qr(i). Works for
//! reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void PressureEquilibrium(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    const Real &alpm1, const Real &alp0, const Real &alpp1, Real gddm1[NSPMETRIC],
    Real gdd0[NSPMETRIC], Real gddp1[NSPMETRIC], const Real &n0, const Real &P0,
    Real Y[MAX_SPECIES], Real &pl_ip1, Real &pr_i) {
  // Constants needed for both the left and right
  Real T0 = eos.ps.GetEOS().GetTemperatureFromP(n0, P0, Y);
  Real e0 = eos.ps.GetEOS().GetEnergy(n0, T0, Y);
  Real guu0[NSPMETRIC];
  Real sdetg0 = Kokkos::sqrt(Primitive::GetDeterminant(gdd0));
  Real chi0 = Kokkos::log(sdetg0);
  Primitive::InvertMatrix(guu0, gdd0, sdetg0*sdetg0);
  constexpr int max_iter = 30;
  constexpr Real tol = 1e-15;

  // Right side of i-1/2
  Real alp12 = 0.5*(alpm1 + alp0);
  Real gdd12[NSPMETRIC];
  Real guu12[NSPMETRIC];
  for (int a = 0; a < NSPMETRIC; a++) {
    gdd12[a] = 0.5*(gddm1[a] + gdd0[a]);
  }
  Real sdetg12 = Kokkos::sqrt(Primitive::GetDeterminant(gdd12));
  Primitive::InvertMatrix(guu12, gdd12, sdetg12*sdetg12);
  Real q0 = guu0[S11]*(gddm1[S11] - gdd0[S11]) + 2.0*guu0[S12]*(gddm1[S12] - gdd0[S12])
         + 2.0*guu0[S13]*(gddm1[S13] - gdd0[S13]) + guu0[S22]*(gddm1[S22] - gdd0[S22])
         + 2.0*guu0[S23]*(gddm1[S23] - gdd0[S23]) + guu0[S33]*(gddm1[S33] - gdd0[S33]);
  Real pe = (alp0*sdetg0*P0*(1.0 + 0.25*q0) - 0.5*sdetg0*e0*(alpm1 - alp0))/(alp12*sdetg12);
  if (pe <= 0.0) {
    pe = P0;
  }
  pr_i = pe;

  // Left side of i+1/2
  alp12 = 0.5*(alpp1 + alp0);
  for (int a = 0; a < NSPMETRIC; a++) {
    gdd12[a] = 0.5*(gddp1[a] + gdd0[a]);
  }
  sdetg12 = Kokkos::sqrt(Primitive::GetDeterminant(gdd12));
  Primitive::InvertMatrix(guu12, gdd12, sdetg12*sdetg12);
  q0 = guu0[S11]*(gddp1[S11] - gdd0[S11]) + 2.0*guu0[S12]*(gddp1[S12] - gdd0[S12])
         + 2.0*guu0[S13]*(gddp1[S13] - gdd0[S13]) + guu0[S22]*(gddp1[S22] - gdd0[S22])
         + 2.0*guu0[S23]*(gddp1[S23] - gdd0[S23]) + guu0[S33]*(gddp1[S33] - gdd0[S33]);
  pe = (alp0*sdetg0*P0*(1.0 + 0.25*q0) - 0.5*sdetg0*e0*(alpp1 - alp0))/(alp12*sdetg12);
  if (pe <= 0.0) {
    pe = P0;
  }
  pl_ip1 = pe;
}

//----------------------------------------------------------------------------------------
//! \fn void PressureEquilibriumX1
//! \brief inline function for well-balancing pressure in the x1-direction
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void PressureEquilibriumX1(TeamMember_t const &member, 
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos, const int nscal,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, const adm::ADM::ADM_vars& adm,
     ScrArray1D<Real> &pl, ScrArray1D<Real> &pr) {
  par_for_inner(member, il, iu, [&](const int i) {
    Real &alp0 = adm.alpha(m, k, j, i);
    Real &alpp1 = adm.alpha(m, k, j, i+1);
    Real &alpm1 = adm.alpha(m, k, j, i-1);
    Real &rho0 = q(m, IDN, k, j, i);
    Real n0 = rho0/eos.ps.GetEOS().GetBaryonMass();
    Real &P0   = q(m, IPR, k, j, i);
    Real Y[MAX_SPECIES];
    for (int s = 0; s < nscal; s++) {
      Y[s] = q(m, IYF+s, k, j, i);
    }
    Real gdd0[NSPMETRIC] = {adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i)};
    Real gddp1[NSPMETRIC] = {adm.g_dd(m, 0, 0, k, j, i+1), adm.g_dd(m, 0, 1, k, j, i+1),
                             adm.g_dd(m, 0, 2, k, j, i+1), adm.g_dd(m, 1, 1, k, j, i+1),
                             adm.g_dd(m, 1, 2, k, j, i+1), adm.g_dd(m, 2, 2, k, j, i+1)};
    Real gddm1[NSPMETRIC] = {adm.g_dd(m, 0, 0, k, j, i-1), adm.g_dd(m, 0, 1, k, j, i-1),
                             adm.g_dd(m, 0, 2, k, j, i-1), adm.g_dd(m, 1, 1, k, j, i-1),
                             adm.g_dd(m, 1, 2, k, j, i-1), adm.g_dd(m, 2, 2, k, j, i-1)};

    PressureEquilibrium(eos, alpm1, alp0, alpp1, gddm1, gdd0, gddp1, n0, P0, Y,
                        pl(i+1), pr(i));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void PressureEquilibriumX2
//! \brief inline function for well-balancing pressure in the x2-direction
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void PressureEquilibriumX2(TeamMember_t const &member, 
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos, const int nscal,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, const adm::ADM::ADM_vars& adm,
     ScrArray1D<Real> &pl_jp1, ScrArray1D<Real> &pr_j) {
  par_for_inner(member, il, iu, [&](const int i) {
    Real &alp0 = adm.alpha(m, k, j, i);
    Real &alpp1 = adm.alpha(m, k, j+1, i);
    Real &alpm1 = adm.alpha(m, k, j-1, i);
    Real &rho0 = q(m, IDN, k, j, i);
    Real n0 = rho0/eos.ps.GetEOS().GetBaryonMass();
    Real &P0   = q(m, IPR, k, j, i);
    Real Y[MAX_SPECIES];
    for (int s = 0; s < nscal; s++) {
      Y[s] = q(m, IYF+s, k, j, i);
    }
    Real gdd0[NSPMETRIC] = {adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i)};
    Real gddp1[NSPMETRIC] = {adm.g_dd(m, 0, 0, k, j+1, i), adm.g_dd(m, 0, 1, k, j+1, i),
                             adm.g_dd(m, 0, 2, k, j+1, i), adm.g_dd(m, 1, 1, k, j+1, i),
                             adm.g_dd(m, 1, 2, k, j+1, i), adm.g_dd(m, 2, 2, k, j+1, i)};
    Real gddm1[NSPMETRIC] = {adm.g_dd(m, 0, 0, k, j-1, i), adm.g_dd(m, 0, 1, k, j-1, i),
                             adm.g_dd(m, 0, 2, k, j-1, i), adm.g_dd(m, 1, 1, k, j-1, i),
                             adm.g_dd(m, 1, 2, k, j-1, i), adm.g_dd(m, 2, 2, k, j-1, i)};

    PressureEquilibrium(eos, alpm1, alp0, alpp1, gddm1, gdd0, gddp1, n0, P0, Y,
                        pl_jp1(i), pr_j(i));
  });
}

//----------------------------------------------------------------------------------------
//! \fn void PressureEquilibriumX3
//! \brief inline function for well-balancing pressure in the x3-direction
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void PressureEquilibriumX3(TeamMember_t const &member, 
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos, const int nscal,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, const adm::ADM::ADM_vars& adm,
     ScrArray1D<Real> &pl_kp1, ScrArray1D<Real> &pr_k) {
  par_for_inner(member, il, iu, [&](const int i) {
    Real &alp0 = adm.alpha(m, k, j, i);
    Real &alpp1 = adm.alpha(m, k, j+1, i);
    Real &alpm1 = adm.alpha(m, k, j-1, i);
    Real &rho0 = q(m, IDN, k, j, i);
    Real n0 = rho0/eos.ps.GetEOS().GetBaryonMass();
    Real &P0   = q(m, IPR, k, j, i);
    Real Y[MAX_SPECIES];
    for (int s = 0; s < nscal; s++) {
      Y[s] = q(m, IYF+s, k, j, i);
    }
    Real gdd0[NSPMETRIC] = {adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i)};
    Real gddp1[NSPMETRIC] = {adm.g_dd(m, 0, 0, k+1, j, i), adm.g_dd(m, 0, 1, k+1, j, i),
                             adm.g_dd(m, 0, 2, k+1, j, i), adm.g_dd(m, 1, 1, k+1, j, i),
                             adm.g_dd(m, 1, 2, k+1, j, i), adm.g_dd(m, 2, 2, k+1, j, i)};
    Real gddm1[NSPMETRIC] = {adm.g_dd(m, 0, 0, k-1, j, i), adm.g_dd(m, 0, 1, k-1, j, i),
                             adm.g_dd(m, 0, 2, k-1, j, i), adm.g_dd(m, 1, 1, k-1, j, i),
                             adm.g_dd(m, 1, 2, k-1, j, i), adm.g_dd(m, 2, 2, k-1, j, i)};

    PressureEquilibrium(eos, alpm1, alp0, alpp1, gddm1, gdd0, gddp1, n0, P0, Y,
                        pl_kp1(i), pr_k(i));
  });
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_WB_EQUILIBRIUM_HPP_
