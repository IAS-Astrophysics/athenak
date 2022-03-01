//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_hyd.cpp
//! \brief derived class that implements ideal gas EOS in nonrelativistic hydro

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealHydro::IdealHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables. Operates over range of cells given
//! in argument list.

void IdealHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gm1 = (eos_data.gamma - 1.0);
  Real igm1 = 1.0/(gm1);

  Real &dfloor_ = eos_data.dfloor;
  Real efloor = eos_data.pfloor/gm1;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0;
  Kokkos::parallel_reduce("hyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum_d, int &sum_e) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    const Real& u_m1 = cons(m,IM1,k,j,i);
    const Real& u_m2 = cons(m,IM2,k,j,i);
    const Real& u_m3 = cons(m,IM3,k,j,i);

    Real& w_d  = prim(m,IDN,k,j,i);
    Real& w_vx = prim(m,IVX,k,j,i);
    Real& w_vy = prim(m,IVY,k,j,i);
    Real& w_vz = prim(m,IVZ,k,j,i);
    Real& w_e  = prim(m,IEN,k,j,i);

    // apply density floor, without changing momentum or energy
    if (u_d < dfloor_) {
      u_d = dfloor_;
      sum_d++;
    }
    w_d = u_d;

    Real di = 1.0/u_d;
    w_vx = u_m1*di;
    w_vy = u_m2*di;
    w_vz = u_m3*di;

    // set internal energy, apply floor, correcting total energy
    Real e_k = 0.5*di*(u_m1*u_m1 + u_m2*u_m2 + u_m3*u_m3);
    w_e = (u_e - e_k);
    if (w_e < efloor) {
      w_e = efloor;
      u_e = efloor + e_k;
      sum_e++;
    }

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      prim(m,n,k,j,i) = cons(m,n,k,j,i)/u_d;
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_));

  // store counters
  pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
  pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts conserved into primitive variables. Operates over range of cells given
//! in argument list.

void IdealHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real igm1 = 1.0/(eos_data.gamma - 1.0);

  par_for("hyd_prim2con", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real& u_d  = cons(m,IDN,k,j,i);
    Real& u_e  = cons(m,IEN,k,j,i);
    Real& u_m1 = cons(m,IM1,k,j,i);
    Real& u_m2 = cons(m,IM2,k,j,i);
    Real& u_m3 = cons(m,IM3,k,j,i);

    const Real& w_d  = prim(m,IDN,k,j,i);
    const Real& w_vx = prim(m,IVX,k,j,i);
    const Real& w_vy = prim(m,IVY,k,j,i);
    const Real& w_vz = prim(m,IVZ,k,j,i);

    u_d  = w_d;
    u_m1 = w_vx*w_d;
    u_m2 = w_vy*w_d;
    u_m3 = w_vz*w_d;
    const Real& w_e  = prim(m,IEN,k,j,i);
    u_e = w_e + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = prim(m,n,k,j,i)*w_d;
    }
  });

  return;
}
