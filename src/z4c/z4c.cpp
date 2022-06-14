//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c.cpp
//! \brief implementation of Z4c class constructor and assorted other functions

#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "z4c/z4c.hpp"

namespace z4c {

char const * const Z4c::Z4c_names[Z4c::N_Z4c] = {
  "z4c_chi",
  "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz", "z4c_gzz",
  "z4c_Khat",
  "z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz",
  "z4c_Gamx", "z4c_Gamy", "z4c_Gamz",
  "z4c_Theta",
  "z4c_alpha",
  "z4c_betax", "z4c_betay", "z4c_betaz",
};

char const * const Z4c::Constraint_names[Z4c::N_CON] = {
  "con_C",
  "con_H",
  "con_M",
  "con_Z",
  "con_Mx", "con_My", "con_Mz",
};

char const * const Z4c::Matter_names[Z4c::N_MAT] = {
  "mat_rho",
  "mat_Sx", "mat_Sy", "mat_Sz",
  "mat_Sxx", "mat_Sxy", "mat_Sxz", "mat_Syy", "mat_Syz", "mat_Szz",
};

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters


Z4c::Z4c(MeshBlockPack *ppack, ParameterInput *pin) :
  pmy_pack(ppack),
  u_con("u_con",1,1,1,1,1),
  u_mat("u_mat",1,1,1,1,1),
  u0("u0 z4c",1,1,1,1,1),
  coarse_u0("coarse u0 z4c",1,1,1,1,1),
  u1("u1 z4c",1,1,1,1,1),
  u_rhs("u_rhs z4c",1,1,1,1,1),
  NDIM(3) 
  {
  // (1) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::Profiling::pushRegion("Tensor fields");
  Kokkos::realloc(u_con, nmb, (N_CON), ncells3, ncells2, ncells1);
  // Matter commented out
  // kokkos::realloc(u_mat, nmb, (N_MAT), ncells3, ncells2, ncells1);
  Kokkos::realloc(u0,    nmb, (N_Z4c), ncells3, ncells2, ncells1);
  Kokkos::realloc(u1,    nmb, (N_Z4c), ncells3, ncells2, ncells1);
  Kokkos::realloc(u_rhs, nmb, (N_Z4c), ncells3, ncells2, ncells1);

  con.C.InitWithShallowSlice(u_con, I_CON_C);
  con.H.InitWithShallowSlice(u_con, I_CON_H);
  con.M.InitWithShallowSlice(u_con, I_CON_M);
  con.Z.InitWithShallowSlice(u_con, I_CON_Z);
  con.M_d.InitWithShallowSlice(u_con, I_CON_Mx, I_CON_Mz);

  // Matter commented out
  //mat.rho.InitWithShallowSlice(u_mat, I_MAT_rho);
  //mat.S_d.InitWithShallowSlice(u_mat, I_MAT_Sx, I_MAT_Sz);
  //mat.S_dd.InitWithShallowSlice(u_mat, I_MAT_Sxx, I_MAT_Szz);

  z4c.alpha.InitWithShallowSlice (u0, I_Z4c_alpha);
  z4c.beta_u.InitWithShallowSlice(u0, I_Z4c_betax, I_Z4c_betaz);
  z4c.chi.InitWithShallowSlice   (u0, I_Z4c_chi);
  z4c.Khat.InitWithShallowSlice  (u0, I_Z4c_Khat);
  z4c.Theta.InitWithShallowSlice (u0, I_Z4c_Theta);
  z4c.Gam_u.InitWithShallowSlice (u0, I_Z4c_Gamx, I_Z4c_Gamz);
  z4c.g_dd.InitWithShallowSlice  (u0, I_Z4c_gxx, I_Z4c_gzz);
  z4c.A_dd.InitWithShallowSlice  (u0, I_Z4c_Axx, I_Z4c_Azz);
  
  rhs.alpha.InitWithShallowSlice (u_rhs, I_Z4c_alpha);
  rhs.beta_u.InitWithShallowSlice(u_rhs, I_Z4c_betax, I_Z4c_betaz);
  rhs.chi.InitWithShallowSlice   (u_rhs, I_Z4c_chi);
  rhs.Khat.InitWithShallowSlice  (u_rhs, I_Z4c_Khat);
  rhs.Theta.InitWithShallowSlice (u_rhs, I_Z4c_Theta);
  rhs.Gam_u.InitWithShallowSlice (u_rhs, I_Z4c_Gamx, I_Z4c_Gamz);
  rhs.g_dd.InitWithShallowSlice  (u_rhs, I_Z4c_gxx, I_Z4c_gzz);
  rhs.A_dd.InitWithShallowSlice  (u_rhs, I_Z4c_Axx, I_Z4c_Azz);
  
  
  
  opt.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opt.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  opt.diss = pin->GetOrAddReal("z4c", "diss", 0.0);
  opt.eps_floor = pin->GetOrAddReal("z4c", "eps_floor", 1e-12);
  opt.damp_kappa1 = pin->GetOrAddReal("z4c", "damp_kappa1", 0.0);
  opt.damp_kappa2 = pin->GetOrAddReal("z4c", "damp_kappa2", 0.0);
  // Gauge conditions (default to moving puncture gauge)
  opt.lapse_harmonicf = pin->GetOrAddReal("z4c", "lapse_harmonicf", 1.0);
  opt.lapse_harmonic = pin->GetOrAddReal("z4c", "lapse_harmonic", 0.0);
  opt.lapse_oplog = pin->GetOrAddReal("z4c", "lapse_oplog", 2.0);
  opt.lapse_advect = pin->GetOrAddReal("z4c", "lapse_advect", 1.0);
  opt.shift_Gamma = pin->GetOrAddReal("z4c", "shift_Gamma", 1.0);
  opt.shift_advect = pin->GetOrAddReal("z4c", "shift_advect", 1.0);

  opt.shift_alpha2Gamma = pin->GetOrAddReal("z4c", "shift_alpha2Gamma", 0.0);
  opt.shift_H = pin->GetOrAddReal("z4c", "shift_H", 0.0);

  opt.shift_eta = pin->GetOrAddReal("z4c", "shift_eta", 2.0);


  diss = opt.diss*pow(2., -2.*indcs.ng)*(indcs.ng % 2 == 0 ? -1. : 1.);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (N_Z4c), nccells3, nccells2, nccells1);
  }
  Kokkos::Profiling::popRegion();

  // allocate boundary buffers for conserved (cell-centered) variables
  Kokkos::Profiling::pushRegion("Buffers");
  pbval_u = new BoundaryValuesCC(ppack, pin);
  pbval_u->InitializeBuffers((N_Z4c));
  Kokkos::Profiling::popRegion();

}

//----------------------------------------------------------------------------------------
// \!fn void Z4c::AlgConstr(AthenaArray<Real> & u)
// \brief algebraic constraints projection
//
// This function operates on all grid points of the MeshBlock
void Z4c::AlgConstr(MeshBlockPack *pmbp)
{ 
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  auto &opt = pmbp->pz4c->opt;
  int &NDIM = pmbp->pz4c->NDIM;
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*3;
  par_for_outer("Alg constr loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> detg;
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> oopsi4;
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> A;
    
      detg.NewAthenaTensor(member, scr_level, ncells1);
    oopsi4.NewAthenaTensor(member, scr_level, ncells1);
         A.NewAthenaTensor(member, scr_level, ncells1);
    par_for_inner(member, isg, ieg, [&](const int i) {
      detg(i) = SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                           z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
      
      detg(i) = detg(i) > 0. ? detg(i) : 1.;
      Real eps = detg(i) - 1.;
      oopsi4(i) = (eps < opt.eps_floor) ? (1. - opt.eps_floor/3.) : (std::pow(1./detg(i), 1./3.));
    });
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        z4c.g_dd(m,a,b,k,j,i) *= oopsi4(i);
      });
    }
    // compute trace of A
    // note: here we are assuming that det g = 1, which we enforced above
    par_for_inner(member, isg, ieg, [&](const int i) {
      A(i) = Trace(1.0,
                   z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                   z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                   z4c.A_dd(m,0,0,k,j,i), z4c.A_dd(m,0,1,k,j,i), z4c.A_dd(m,0,2,k,j,i),
                   z4c.A_dd(m,1,1,k,j,i), z4c.A_dd(m,1,2,k,j,i), z4c.A_dd(m,2,2,k,j,i));
    });
    // enforce trace of A to be zero
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        z4c.A_dd(m,a,b,k,j,i) -= (1.0/3.0) * A(i) * z4c.g_dd(m,a,b,k,j,i);
      });
    }
  });
}
//----------------------------------------------------------------------------------------
// destructor
Z4c::~Z4c() {
  delete pbval_u;
}

} // namespace z4c
