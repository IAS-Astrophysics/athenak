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
#include "reconstruct/plm.hpp"

namespace dyngr {

//---------------------------------------------------------------------------------------
// A convenience structure for storing an input state for the well-balanced scheme.
struct WBState {
  const Real& n;
  const Real& P;
  const Real& T;
  Real* Y;

  const Real& alp;
  Real* gdd;
};

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
    const Real &T0, Real Y[MAX_SPECIES], Real &pl_ip1, Real &pr_i) {
  // Constants needed for both the left and right
  Real e0 = eos.ps.GetEOS().GetEnergy(n0, T0, Y);
  Real guu0[NSPMETRIC];
  Real sdetg0 = Kokkos::sqrt(Primitive::GetDeterminant(gdd0));
  Primitive::InvertMatrix(guu0, gdd0, sdetg0*sdetg0);

  // Right side of i-1/2
  Real alp12 = 0.5*(alpm1 + alp0);
  Real gdd12[NSPMETRIC];
  for (int a = 0; a < NSPMETRIC; a++) {
    gdd12[a] = 0.5*(gddm1[a] + gdd0[a]);
  }
  Real sdetg12 = Kokkos::sqrt(Primitive::GetDeterminant(gdd12));
  Real q0 = guu0[S11]*(gddm1[S11] - gdd0[S11]) + 2.0*guu0[S12]*(gddm1[S12] - gdd0[S12])
         + 2.0*guu0[S13]*(gddm1[S13] - gdd0[S13]) + guu0[S22]*(gddm1[S22] - gdd0[S22])
         + 2.0*guu0[S23]*(gddm1[S23] - gdd0[S23]) + guu0[S33]*(gddm1[S33] - gdd0[S33]);
  Real pe = (alp0*sdetg0*P0*(1.0 + 0.25*q0) - 0.5*sdetg0*e0*(alpm1 - alp0))/(alp12*sdetg12);
  /*if (pe <= 0.0) {
    pe = P0;
  }*/
  pr_i = pe;

  // Left side of i+1/2
  alp12 = 0.5*(alpp1 + alp0);
  for (int a = 0; a < NSPMETRIC; a++) {
    gdd12[a] = 0.5*(gddp1[a] + gdd0[a]);
  }
  sdetg12 = Kokkos::sqrt(Primitive::GetDeterminant(gdd12));
  q0 = guu0[S11]*(gddp1[S11] - gdd0[S11]) + 2.0*guu0[S12]*(gddp1[S12] - gdd0[S12])
         + 2.0*guu0[S13]*(gddp1[S13] - gdd0[S13]) + guu0[S22]*(gddp1[S22] - gdd0[S22])
         + 2.0*guu0[S23]*(gddp1[S23] - gdd0[S23]) + guu0[S33]*(gddp1[S33] - gdd0[S33]);
  pe = (alp0*sdetg0*P0*(1.0 + 0.25*q0) - 0.5*sdetg0*e0*(alpp1 - alp0))/(alp12*sdetg12);
  /*if (pe <= 0.0) {
    pe = P0;
  }*/
  pl_ip1 = pe;
}

//---------------------------------------------------------------------------------------
//! \fn void PressureEquilibriumExtrap
//! \brief inline function for well-balancing pressure at the L/R interfaces of a cell and
//  extrapolating to the nearest neighbor for a PLM operator.
template<class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
bool PressureEquilibriumExtrap(const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos,
    const WBState& vm1, const WBState& v0, const WBState& vp1, const Real& dalpm,
    const Real& dalpp, const Real dgddm[NSPMETRIC], const Real dgddp[NSPMETRIC],
    Real &p_im1, Real &pr_i, Real &pl_ip1, Real &p_ip1) {
  // Constants needed for both the left and right.
  Real e0 = eos.ps.GetEOS().GetEnergy(v0.n, v0.T, v0.Y);
  Real guu[NSPMETRIC];
  Real sdetg0 = Kokkos::sqrt(Primitive::GetDeterminant(v0.gdd));
  Primitive::InvertMatrix(guu, v0.gdd, sdetg0*sdetg0);

  auto CalcQ = [](const Real guu[NSPMETRIC], const Real dgdd[NSPMETRIC]) -> Real {
    return guu[S11]*dgdd[S11] + 2.0*guu[S12]*dgdd[S12] + 2.0*guu[S13]*dgdd[S13] +
           guu[S22]*dgdd[S22] + 2.0*guu[S23]*dgdd[S23] + guu[S33]*dgdd[S33];
  };
  auto CalcPEq = [](const WBState& v, const Real& dalp, const Real& q,
                    const Real& sdetg, const Real& e) -> Real {
    return sdetg*(v.alp*sdetg*v.P*(1.0 + 0.5*q) - e*dalp);
  };

  // Right side of i-1/2
  Real q0l = CalcQ(guu, dgddm);
  Real pem = CalcPEq(v0, dalpm, q0l, sdetg0, e0);
  if (pem < 0.0) {
    pr_i = p_im1 = pl_ip1 = p_ip1 = v0.P;
    return false;
  }
  {
    // Scoping to try to encourage the compiler to reduce register usage.
    Real gdd12[NSPMETRIC];
    for (int a = 0; a < NSPMETRIC; a++) {
      gdd12[a] = 0.5*(vm1.gdd[a] + v0.gdd[a]);
    }
    Real sdetg12 = Kokkos::sqrt(Primitive::GetDeterminant(gdd12));
    Real alp12 = 0.5*(v0.alp + vm1.alp);
    pr_i = pem/(alp12*sdetg12);
  }

  // Left side of i+1/2
  Real q0r = CalcQ(guu, dgddp);
  Real pep = CalcPEq(v0, dalpp, q0r, sdetg0, e0);
  if (pep < 0.0) {
    pr_i = p_im1 = pl_ip1 = p_ip1 = v0.P;
    return false;
  }
  {
    // Scoping to try to encourage the compiler to reduce register usage.
    Real gdd12[NSPMETRIC];
    for (int a = 0; a < NSPMETRIC; a++) {
      gdd12[a] = 0.5*(vp1.gdd[a] + v0.gdd[a]);
    }
    Real sdetg12 = Kokkos::sqrt(Primitive::GetDeterminant(gdd12));
    Real alp12 = 0.5*(v0.alp + vp1.alp);
    pl_ip1 = pep/(alp12*sdetg12);
  }

  // Extrapolate to i-1
  Real sdetg1 = Kokkos::sqrt(Primitive::GetDeterminant(vm1.gdd));
  Primitive::InvertMatrix(guu, vm1.gdd, sdetg1*sdetg1);
  Real q1 = CalcQ(guu, dgddm);
  // Estimate the solution using a single step of fixed-point iteration.
  Real e1 = eos.ps.GetEOS().GetEnergy(vm1.n, vm1.T, vm1.Y);
  Real pstar = (pem - sdetg1*e1*dalpm)/(vm1.alp*sdetg1*(1.0 - 0.5*q1));
  if (pstar <= 0.0) {
    pr_i = p_im1 = pl_ip1 = p_ip1 = v0.P;
    return false;
  }
  p_im1 = pstar;

  // Extrapolate to i+1
  sdetg1 = Kokkos::sqrt(Primitive::GetDeterminant(vp1.gdd));
  Primitive::InvertMatrix(guu, vp1.gdd, sdetg1*sdetg1);
  q1 = CalcQ(guu, dgddp);
  // Estimate the solution using a single step of fixed-point iteration.
  e1 = eos.ps.GetEOS().GetEnergy(vp1.n, vp1.T, vp1.Y);
  pstar = (pep - sdetg1*e1*dalpp)/(vp1.alp*sdetg1*(1.0 - 0.5*q1));
  if (pstar <= 0.0) {
    pr_i = p_im1 = pl_ip1 = p_ip1 = v0.P;
    return false;
  }
  p_ip1 = pstar;

  return true;
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
     const DvceArray5D<Real> &temp, ScrArray1D<Real> &pl, ScrArray1D<Real> &pr) {
  par_for_inner(member, il, iu, [&](const int i) {
    Real &alp0 = adm.alpha(m, k, j, i);
    Real &alpp1 = adm.alpha(m, k, j, i+1);
    Real &alpm1 = adm.alpha(m, k, j, i-1);
    Real &rho0 = q(m, IDN, k, j, i);
    Real n0 = rho0/eos.ps.GetEOS().GetBaryonMass();
    Real &P0   = q(m, IPR, k, j, i);
    Real &T0   = temp(m, 0, k, j, i);
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

    PressureEquilibrium(eos, alpm1, alp0, alpp1, gddm1, gdd0, gddp1, n0, P0, T0, Y,
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
     const DvceArray5D<Real> &temp, ScrArray1D<Real> &pl_jp1, ScrArray1D<Real> &pr_j) {
  par_for_inner(member, il, iu, [&](const int i) {
    Real &alp0 = adm.alpha(m, k, j, i);
    Real &alpp1 = adm.alpha(m, k, j+1, i);
    Real &alpm1 = adm.alpha(m, k, j-1, i);
    Real &rho0 = q(m, IDN, k, j, i);
    Real n0 = rho0/eos.ps.GetEOS().GetBaryonMass();
    Real &P0   = q(m, IPR, k, j, i);
    Real &T0   = temp(m, 0, k, j, i);
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

    PressureEquilibrium(eos, alpm1, alp0, alpp1, gddm1, gdd0, gddp1, n0, P0, T0, Y,
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
     const DvceArray5D<Real> &temp, ScrArray1D<Real> &pl_kp1, ScrArray1D<Real> &pr_k) {
  par_for_inner(member, il, iu, [&](const int i) {
    Real &alp0 = adm.alpha(m, k, j, i);
    Real &alpp1 = adm.alpha(m, k+1, j, i);
    Real &alpm1 = adm.alpha(m, k-1, j, i);
    Real &rho0 = q(m, IDN, k, j, i);
    Real n0 = rho0/eos.ps.GetEOS().GetBaryonMass();
    Real &P0   = q(m, IPR, k, j, i);
    Real &T0   = temp(m, 0, k, j, i);
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

    PressureEquilibrium(eos, alpm1, alp0, alpp1, gddm1, gdd0, gddp1, n0, P0, T0, Y,
                        pl_kp1(i), pr_k(i));
  });
}

//----------------------------------------------------------------------------------------
//! \fn BalancePressureX1()
//! \brief Wrapper function to apply well-balanced PLM reconstruction to pressure in x1
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
template<int width, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void BalancePressureX1(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos, const int nscal,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, const adm::ADM::ADM_vars& adm,
     const DvceArray5D<Real> &temp, ScrArray2D<Real> &ql, ScrArray2D<Real> &qr) {
  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  par_for_inner(member, il, iu, [&](const int i) {
    const Real &alp0 = adm.alpha(m, k, j, i);
    const Real &alpp1 = adm.alpha(m, k, j, i+1);
    const Real &alpm1 = adm.alpha(m, k, j, i-1);

    const Real n0 = q(m, IDN, k, j, i)/mb;
    const Real nm1 = q(m, IDN, k, j, i-1)/mb;
    const Real np1 = q(m, IDN, k, j, i+1)/mb;

    const Real &P0 = q(m, IPR, k, j, i);
    const Real &Pm1 = q(m, IPR, k, j, i-1);
    const Real &Pp1 = q(m, IPR, k, j, i+1);
    
    const Real &T0 = temp(m, 0, k, j, i);
    const Real &Tm1 = temp(m, 0, k, j, i-1);
    const Real &Tp1 = temp(m, 0, k, j, i+1);

    Real Y0[MAX_SPECIES], Ym1[MAX_SPECIES], Yp1[MAX_SPECIES];
    for (int s = 0; s < nscal; s++) {
      Y0[s] = q(m, IYF+s, k, j, i);
      Yp1[s] = q(m, IYF+s, k, j, i+1);
      Ym1[s] = q(m, IYF+s, k, j, i-1);
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
    WBState v0{n0, P0, T0, Y0, alp0, gdd0};
    WBState vm1{nm1, Pm1, Tm1, Ym1, alpm1, gddm1};
    WBState vp1{np1, Pp1, Tp1, Yp1, alpp1, gddp1};
    Real p_im1, pr_i, pl_ip1, p_ip1;
    Real dalpm = HalfDifferenceInterface<width,0,-1>(adm.alpha, m, k, j, i);
    Real dalpp = HalfDifferenceInterface<width,0, 1>(adm.alpha, m, k, j, i);
    Real dgddm[NSPMETRIC] = {HalfDifferenceInterface<width,0,-1>(adm.g_dd,m,0,0,k,j,i),
                             HalfDifferenceInterface<width,0,-1>(adm.g_dd,m,0,1,k,j,i),
                             HalfDifferenceInterface<width,0,-1>(adm.g_dd,m,0,2,k,j,i),
                             HalfDifferenceInterface<width,0,-1>(adm.g_dd,m,1,1,k,j,i),
                             HalfDifferenceInterface<width,0,-1>(adm.g_dd,m,1,2,k,j,i),
                             HalfDifferenceInterface<width,0,-1>(adm.g_dd,m,2,2,k,j,i)};
    Real dgddp[NSPMETRIC] = {HalfDifferenceInterface<width,0, 1>(adm.g_dd,m,0,0,k,j,i),
                             HalfDifferenceInterface<width,0, 1>(adm.g_dd,m,0,1,k,j,i),
                             HalfDifferenceInterface<width,0, 1>(adm.g_dd,m,0,2,k,j,i),
                             HalfDifferenceInterface<width,0, 1>(adm.g_dd,m,1,1,k,j,i),
                             HalfDifferenceInterface<width,0, 1>(adm.g_dd,m,1,2,k,j,i),
                             HalfDifferenceInterface<width,0, 1>(adm.g_dd,m,2,2,k,j,i)};
    bool balanced = PressureEquilibriumExtrap(eos, vm1, v0, vp1, dalpm, dalpp, dgddm,
                                              dgddp, p_im1, pr_i, pl_ip1, p_ip1);
    if (balanced) {
      PLM(q(m,IPR,k,j,i-1)-p_im1, 0.0, q(m,IPR,k,j,i+1)-p_ip1, ql(IPR,i+1), qr(IPR,i));
      ql(IPR,i+1) += pl_ip1;
      qr(IPR,i) += pr_i;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn BalancePressureX2()
//! \brief Wrapper function to apply well-balanced PLM reconstruction to pressure in x2
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
template<int width, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void BalancePressureX2(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos, const int nscal,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, const adm::ADM::ADM_vars& adm,
     const DvceArray5D<Real> &temp, ScrArray2D<Real> &ql_jp1, ScrArray2D<Real> &qr_j) {
  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  par_for_inner(member, il, iu, [&](const int i) {
    const Real &alp0 = adm.alpha(m, k, j, i);
    const Real &alpp1 = adm.alpha(m, k, j+1, i);
    const Real &alpm1 = adm.alpha(m, k, j-1, i);

    const Real n0 = q(m, IDN, k, j, i)/mb;
    const Real nm1 = q(m, IDN, k, j-1, i)/mb;
    const Real np1 = q(m, IDN, k, j+1, i)/mb;

    const Real &P0 = q(m, IPR, k, j, i);
    const Real &Pm1 = q(m, IPR, k, j-1, i);
    const Real &Pp1 = q(m, IPR, k, j+1, i);
    
    const Real &T0 = temp(m, 0, k, j, i);
    const Real &Tm1 = temp(m, 0, k, j-1, i);
    const Real &Tp1 = temp(m, 0, k, j+1, i);

    Real Y0[MAX_SPECIES], Ym1[MAX_SPECIES], Yp1[MAX_SPECIES];
    for (int s = 0; s < nscal; s++) {
      Y0[s] = q(m, IYF+s, k, j, i);
      Yp1[s] = q(m, IYF+s, k, j+1, i);
      Ym1[s] = q(m, IYF+s, k, j-1, i);
    }
    // Metric indices are permuted to try to reduce floating-point symmetry errors.
    Real gdd0[NSPMETRIC] = {adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i),
                            adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 2, 2, k, j, i),
                            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 0, 0, k, j, i)};
    Real gddp1[NSPMETRIC] = {adm.g_dd(m, 1, 1, k, j+1, i), adm.g_dd(m, 1, 2, k, j+1, i),
                             adm.g_dd(m, 0, 1, k, j+1, i), adm.g_dd(m, 2, 2, k, j+1, i),
                             adm.g_dd(m, 0, 2, k, j+1, i), adm.g_dd(m, 0, 0, k, j+1, i)};
    Real gddm1[NSPMETRIC] = {adm.g_dd(m, 1, 1, k, j-1, i), adm.g_dd(m, 1, 2, k, j-1, i),
                             adm.g_dd(m, 0, 1, k, j-1, i), adm.g_dd(m, 2, 2, k, j-1, i),
                             adm.g_dd(m, 0, 2, k, j-1, i), adm.g_dd(m, 0, 0, k, j-1, i)};
    WBState v0{n0, P0, T0, Y0, alp0, gdd0};
    WBState vm1{nm1, Pm1, Tm1, Ym1, alpm1, gddm1};
    WBState vp1{np1, Pp1, Tp1, Yp1, alpp1, gddp1};
    Real p_jm1, pr_j, pl_jp1, p_jp1;
    Real dalpm = HalfDifferenceInterface<width,1,-1>(adm.alpha, m, k, j, i);
    Real dalpp = HalfDifferenceInterface<width,1, 1>(adm.alpha, m, k, j, i);
    Real dgddm[NSPMETRIC] = {HalfDifferenceInterface<width,1,-1>(adm.g_dd,m,1,1,k,j,i),
                             HalfDifferenceInterface<width,1,-1>(adm.g_dd,m,1,2,k,j,i),
                             HalfDifferenceInterface<width,1,-1>(adm.g_dd,m,0,1,k,j,i),
                             HalfDifferenceInterface<width,1,-1>(adm.g_dd,m,2,2,k,j,i),
                             HalfDifferenceInterface<width,1,-1>(adm.g_dd,m,0,2,k,j,i),
                             HalfDifferenceInterface<width,1,-1>(adm.g_dd,m,0,0,k,j,i)};
    Real dgddp[NSPMETRIC] = {HalfDifferenceInterface<width,1, 1>(adm.g_dd,m,1,1,k,j,i),
                             HalfDifferenceInterface<width,1, 1>(adm.g_dd,m,1,2,k,j,i),
                             HalfDifferenceInterface<width,1, 1>(adm.g_dd,m,0,1,k,j,i),
                             HalfDifferenceInterface<width,1, 1>(adm.g_dd,m,2,2,k,j,i),
                             HalfDifferenceInterface<width,1, 1>(adm.g_dd,m,0,2,k,j,i),
                             HalfDifferenceInterface<width,1, 1>(adm.g_dd,m,0,0,k,j,i)};
    bool balanced = PressureEquilibriumExtrap(eos, vm1, v0, vp1, dalpm, dalpp, dgddm,
                                              dgddp, p_jm1, pr_j, pl_jp1, p_jp1);
    if (balanced) {
      PLM(q(m,IPR,k,j-1,i)-p_jm1, 0.0, q(m,IPR,k,j+1,i)-p_jp1, ql_jp1(IPR,i), qr_j(IPR,i));
      ql_jp1(IPR,i) += pl_jp1;
      qr_j(IPR,i) += pr_j;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn BalancePressureX3()
//! \brief Wrapper function to apply well-balanced PLM reconstruction to pressure in x3
//! This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
template<int width, class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION
void BalancePressureX3(TeamMember_t const &member,
     const PrimitiveSolverHydro<EOSPolicy, ErrorPolicy>& eos, const int nscal,
     const int m, const int k, const int j, const int il, const int iu,
     const DvceArray5D<Real> &q, const adm::ADM::ADM_vars& adm,
     const DvceArray5D<Real> &temp, ScrArray2D<Real> &ql_kp1, ScrArray2D<Real> &qr_k) {
  const Real mb = eos.ps.GetEOS().GetBaryonMass();
  par_for_inner(member, il, iu, [&](const int i) {
    const Real &alp0 = adm.alpha(m, k, j, i);
    const Real &alpp1 = adm.alpha(m, k+1, j, i);
    const Real &alpm1 = adm.alpha(m, k-1, j, i);

    const Real n0 = q(m, IDN, k, j, i)/mb;
    const Real nm1 = q(m, IDN, k-1, j, i)/mb;
    const Real np1 = q(m, IDN, k+1, j, i)/mb;

    const Real &P0 = q(m, IPR, k, j, i);
    const Real &Pm1 = q(m, IPR, k-1, j, i);
    const Real &Pp1 = q(m, IPR, k+1, j, i);
    
    const Real &T0 = temp(m, 0, k, j, i);
    const Real &Tm1 = temp(m, 0, k-1, j, i);
    const Real &Tp1 = temp(m, 0, k+1, j, i);

    Real Y0[MAX_SPECIES], Ym1[MAX_SPECIES], Yp1[MAX_SPECIES];
    for (int s = 0; s < nscal; s++) {
      Y0[s] = q(m, IYF+s, k, j, i);
      Yp1[s] = q(m, IYF+s, k+1, j, i);
      Ym1[s] = q(m, IYF+s, k-1, j, i);
    }
    // Metric indices are permuted to try to reduce floating-point symmetry errors.
    Real gdd0[NSPMETRIC] = {adm.g_dd(m, 2, 2, k, j, i), adm.g_dd(m, 0, 2, k, j, i),
                            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 0, 0, k, j, i),
                            adm.g_dd(m, 0, 1, k, j, i), adm.g_dd(m, 1, 1, k, j, i)};
    Real gddp1[NSPMETRIC] = {adm.g_dd(m, 2, 2, k+1, j, i), adm.g_dd(m, 0, 2, k+1, j, i),
                             adm.g_dd(m, 1, 2, k+1, j, i), adm.g_dd(m, 0, 0, k+1, j, i),
                             adm.g_dd(m, 0, 1, k+1, j, i), adm.g_dd(m, 1, 1, k+1, j, i)};
    Real gddm1[NSPMETRIC] = {adm.g_dd(m, 2, 2, k-1, j, i), adm.g_dd(m, 0, 2, k-1, j, i),
                             adm.g_dd(m, 1, 2, k-1, j, i), adm.g_dd(m, 0, 0, k-1, j, i),
                             adm.g_dd(m, 0, 1, k-1, j, i), adm.g_dd(m, 1, 1, k-1, j, i)};
    WBState v0{n0, P0, T0, Y0, alp0, gdd0};
    WBState vm1{nm1, Pm1, Tm1, Ym1, alpm1, gddm1};
    WBState vp1{np1, Pp1, Tp1, Yp1, alpp1, gddp1};
    Real p_km1, pr_k, pl_kp1, p_kp1;
    Real dalpm = HalfDifferenceInterface<width,2,-1>(adm.alpha, m, k, j, i);
    Real dalpp = HalfDifferenceInterface<width,2, 1>(adm.alpha, m, k, j, i);
    Real dgddm[NSPMETRIC] = {HalfDifferenceInterface<width,2,-1>(adm.g_dd,m,2,2,k,j,i),
                             HalfDifferenceInterface<width,2,-1>(adm.g_dd,m,0,2,k,j,i),
                             HalfDifferenceInterface<width,2,-1>(adm.g_dd,m,1,2,k,j,i),
                             HalfDifferenceInterface<width,2,-1>(adm.g_dd,m,0,0,k,j,i),
                             HalfDifferenceInterface<width,2,-1>(adm.g_dd,m,0,1,k,j,i),
                             HalfDifferenceInterface<width,2,-1>(adm.g_dd,m,1,1,k,j,i)};
    Real dgddp[NSPMETRIC] = {HalfDifferenceInterface<width,2, 1>(adm.g_dd,m,2,2,k,j,i),
                             HalfDifferenceInterface<width,2, 1>(adm.g_dd,m,0,2,k,j,i),
                             HalfDifferenceInterface<width,2, 1>(adm.g_dd,m,1,2,k,j,i),
                             HalfDifferenceInterface<width,2, 1>(adm.g_dd,m,0,0,k,j,i),
                             HalfDifferenceInterface<width,2, 1>(adm.g_dd,m,0,1,k,j,i),
                             HalfDifferenceInterface<width,2, 1>(adm.g_dd,m,1,1,k,j,i)};
    bool balanced = PressureEquilibriumExtrap(eos, vm1, v0, vp1, dalpm, dalpp, dgddm,
                                              dgddp, p_km1, pr_k, pl_kp1, p_kp1);
    if (balanced) {
      PLM(q(m,IPR,k-1,j,i)-p_km1, 0.0, q(m,IPR,k+1,j,i)-p_kp1, ql_kp1(IPR,i), qr_k(IPR,i));
      ql_kp1(IPR,i) += pl_kp1;
      qr_k(IPR,i) += pr_k;
    }
  });
}

} // namespace dyngr

#endif  // DYN_GRMHD_RSOLVERS_WB_EQUILIBRIUM_HPP_
