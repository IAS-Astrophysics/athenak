//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_srhyd.cpp
//! \brief derived class that implements ideal gas EOS in special relativistic hydro
//! Conserved to primitive variable inversion using algorithm described in Appendix C
//! of Galeazzi et al., PhysRevD, 88, 064009 (2013). Equation refs are to this paper.

#include <float.h>

#include "athena.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealSRHydro::IdealSRHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("hydro","gamma_max",(FLT_MAX));  // gamma ceiling
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables for an ideal gas in SR hydro.
//! Implementation follows Wolfgang Kastaun's algorithm described in Appendix C of
//! Galeazzi et al., PhysRevD, 88, 064009 (2013).  Roots of "master function" (eq. C22)
//! found by false position method.
//!
//! In SR hydrodynamics, the conserved variables are: (D, E - D, m^i), where
//!    D = \gamma \rho is the density in the lab frame,
//!    \gamma = (1 + u^2)^{1/2} = (1 - v^2)^{-1/2} is the Lorentz factor,
//!    u^i = \gamma v^i are the spatial components of the 4-velocity (v^i is 3-vel),
//!    \rho is the comoving/fluid frame mass density,
//!    E = \gamma^2 w - P_g is the total energy,
//!    w = \rho + [\Gamma / (\Gamma - 1)] P_g is the total enthalpy,
//!    \Gamma is the adiabatic index, P_g is the gas pressure
//!    m^i = \gamma w u^i are components of the momentum in the lab frame.
//! Note we evolve (E-D). This improves accuracy/stability in high-density regions.
//!
//! In SR hydrodynamics, the primitive variables are: (\rho, P_gas, u^i).
//! Note components of the 4-velocity (not 3-velocity) are stored in the primitive
//! variables because tests show it is better to reconstruct the 4-vel.
//!
//! This function operates over range of cells given in argument list.

void IdealSRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                              const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->phydro->fofc;
  auto eos = eos_data;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nceilv_=0, nfail_=0, maxit_=0;
  Kokkos::parallel_reduce("srhyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumv, int &sumf, int &max_it) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // Compute (S^i S_i) (eqn C2)
    Real s2 = SQR(u.mx) + SQR(u.my) + SQR(u.mz);

    // call c2p function
    // (inline function in ideal_c2p_hyd.hpp file)
    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    bool vceiling_used=false, c2p_failure=false;
    int iter_used=0;
    SingleC2P_IdealSRHyd(u, eos, s2, w,
                         dfloor_used, efloor_used, c2p_failure, iter_used);
    // apply velocity ceiling if necessary
    Real lor = sqrt(1.0+SQR(w.vx)+SQR(w.vy)+SQR(w.vz));
    if (lor > eos.gamma_max) {
      vceiling_used = true;
      Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
      w.vx *= factor;
      w.vy *= factor;
      w.vz *= factor;
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (vceiling_used) {sumv++;}
      if (c2p_failure) {sumf++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // reset conserved variables if floor, ceiling, or failure encountered
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        SingleP2C_IdealSRHyd(w, eos.gamma, u);
        cons(m,IDN,k,j,i) = u.d;
        cons(m,IM1,k,j,i) = u.mx;
        cons(m,IM2,k,j,i) = u.my;
        cons(m,IM3,k,j,i) = u.mz;
        cons(m,IEN,k,j,i) = u.e;
      }
      // convert scalars (if any)
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nceilv_),
     Kokkos::Sum<int>(nfail_), Kokkos::Max<int>(maxit_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
    pmy_pack->pmesh->ecounter.neos_fail   += nfail_;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for SR hydrodynamics. Operates
//! over range of cells given in argument list.
//! Recall in SR hydrodynamics the conserved variables are: (D, E-D, m^i),
//!                        and the primitive variables are: (\rho, P_gas, u^i).

void IdealSRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;

  par_for("srhyd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Load single state of primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealSRHyd(w, gamma, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
