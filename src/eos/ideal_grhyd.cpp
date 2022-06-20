//========================================================================================
// Athena++ (Kokkos version) astrophysical MHD code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grhyd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic hydro
//! Uses the same algorithm as implemented for SR hydro.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"
#include "eos/ideal_hyd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRHydro::IdealGRHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables for an ideal gas in GR hydro.
//! Operates over range of cells given in argument list.

void IdealGRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                              const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->phydro->fofc;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &horizon_mask_ = pmy_pack->pcoord->cc_mask;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, maxit_=0;
  Kokkos::parallel_reduce("grhyd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &max_it) {
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

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    int iter_used=0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (horizon_mask_(m,k,j,i)) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = pexcise_/gm1;
        excised = true;
      }
    }

    if (!(excised)) {
      // calculate SR conserved quantities
      HydCons1D u_sr;

      // Need to multiply the conserved density by alpha, so that it
      // contains a lorentz factor
      Real alpha = sqrt(-1.0/gupper[0][0]);
      u_sr.d = u.d*alpha;

      // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
      // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
      // We are also evolving T^t_t + D as conserved variable, so must convert to E
      u_sr.e = gupper[0][0]*(u.e - u.d) +
               gupper[0][1]*u.mx + gupper[0][2]*u.my + gupper[0][3]*u.mz;

      // This is only true if sqrt{-g}=1!
      u_sr.e *= (-1./gupper[0][0]);  // Multiply by alpha^2

      // Subtract density for consistency with the rest of the algorithm
      u_sr.e -= u_sr.d;

      // Need to treat the conserved momenta. Also they lack an alpha
      // This is only true if sqrt{-g}=1!
      Real m1l = u.mx*alpha;
      Real m2l = u.my*alpha;
      Real m3l = u.mz*alpha;

      // Need to raise indices on u_sr, which transforms using the spatial 3-metric.
      // This is slightly more involved
      //
      // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
      //       g^0i = beta^i/alpha^2
      //       g^00 = -1/ alpha^2
      // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
      u_sr.mx = ((gupper[1][1] - gupper[0][1]*gupper[0][1]/gupper[0][0])*m1l +
                 (gupper[1][2] - gupper[0][1]*gupper[0][2]/gupper[0][0])*m2l +
                 (gupper[1][3] - gupper[0][1]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

      u_sr.my = ((gupper[2][1] - gupper[0][2]*gupper[0][1]/gupper[0][0])*m1l +
                 (gupper[2][2] - gupper[0][2]*gupper[0][2]/gupper[0][0])*m2l +
                 (gupper[2][3] - gupper[0][2]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

      u_sr.mz = ((gupper[3][1] - gupper[0][3]*gupper[0][1]/gupper[0][0])*m1l +
                 (gupper[3][2] - gupper[0][3]*gupper[0][2]/gupper[0][0])*m2l +
                 (gupper[3][3] - gupper[0][3]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

      // Compute (S^i S_i) (eqn C2)
      Real s2 = (m1l*u_sr.mx) + (m2l*u_sr.my) + (m3l*u_sr.mz);

      // call c2p function
      SingleC2P_IdealSRHyd(u_sr, eos, s2, w, dfloor_used, efloor_used, iter_used);
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;
      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;
      // reset conserved variables if floor is hit or if horizon excised
      if ((dfloor_used || efloor_used) || excised) {
        SingleP2C_IdealGRHyd(glower, gupper, w, eos.gamma, u);
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
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Max<int>(maxit_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for GR hydrodynamics.  Operates
//! over range of cells given in argument list.

void IdealGRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;

  par_for("grhyd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Load single state of primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRHyd(glower, gupper, w, gamma, u);

    // Set conserved quantities
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
