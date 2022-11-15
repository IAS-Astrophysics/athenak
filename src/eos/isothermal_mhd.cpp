//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file isothermal_mhd.cpp
//! \brief derived class that implements isothermal EOS for nonrelativistic mhd

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IsothermalMHD::IsothermalMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = false;
  eos_data.iso_cs = pin->GetReal("mhd","iso_sound_speed");
  eos_data.gamma = 0.0;
  eos_data.use_e = false;
  eos_data.use_t = false;
}

//----------------------------------------------------------------------------------------
//! \!fn void SingleC2P_IsothermalMHD()
//! \brief Converts conserved into primitive variables.  Operates over range of cells
//! given in argument list.

KOKKOS_INLINE_FUNCTION
void SingleC2P_IsothermalMHD(MHDCons1D &u, const Real &dfloor_,
                             HydPrim1D &w, bool &dfloor_used) {
  // apply density floor, without changing momentum
  if (u.d < dfloor_) {
    u.d = dfloor_;
    dfloor_used = true;
  }
  w.d = u.d;
  // compute velocities
  Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IsothermalMHD()
//! \brief Converts single state of primitive variables into conserved variables for
//! non-relativistic MHD with an isothermal EOS.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IsothermalMHD(const HydPrim1D &w, HydCons1D &u) {
  u.d  = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void ConsToPrim()
//! \brief Converts conserved into primitive variables. Operates over range of cells given
//! in argument list.
//! Note that the primitive variables contain the cell-centered magnetic fields, so that
//! W contains (nmhd+3+nscalars) elements, while U contains (nmhd+nscalars)

void IsothermalMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                               DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                               const bool only_testfloors,
                               const int il, const int iu, const int jl, const int ju,
                               const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  Real dfloor = eos_data.dfloor;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0;
  Kokkos::parallel_reduce("isomhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);

    // load cell-centered fields into conserved state
    // (simple linear average of face-centered fields)
    u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
    u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
    u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

    // call c2p function
    HydPrim1D w;
    bool dfloor_used = false;
    SingleC2P_IsothermalMHD(u, dfloor, w, dfloor_used);
    // update counter, reset conserved if floor was hit
    if (dfloor_used) {
      cons(m,IDN,k,j,i) = u.d;
      sumd++;
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used) {fofc_(m,k,j,i) = true;}
    } else {
      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;
      // convert scalars (if any), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list. Does not change cell- or face-centered magnetic fields.

void IsothermalMHD::PrimToCons(const DvceArray5D<Real> &prim,const DvceArray5D<Real> &bcc,
                               DvceArray5D<Real> &cons, const int il, const int iu,
                               const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  par_for("isomhd_prim2cons", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // load single state primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IsothermalMHD(w, u);

    // store conserved state in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;

    // convert scalars (if any), always stored at end of cons and prim arrays.
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
