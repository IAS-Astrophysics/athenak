//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm_z4c.cpp
//! \brief implementation of functions in the Z4c class related to ADM decomposition


// C++ standard headers
#include <cmath> // pow
#include <iostream>
#include <fstream>

// Athena++ headers
#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

//! \fn void Z4c::ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Compute Z4c variables from ADM variables
//
// p  = detgbar^(-1/3)
// p0 = psi^(-4)
//
// gtilde_ij = p gbar_ij
// Ktilde_ij = p p0 K_ij
//
// phi = - log(p) / 4
// K   = gtildeinv^ij Ktilde_ij
// Atilde_ij = Ktilde_ij - gtilde_ij K / 3
//
// G^i = - del_j gtildeinv^ji
//
// BAM: Z4c_init()
// https://git.tpi.uni-jena.de/bamdev/z4
// https://git.tpi.uni-jena.de/bamdev/z4/blob/master/z4_init.m
//
// The Z4c variables will be set on the whole MeshBlock with the exception of
// the Gamma's that can only be set in the interior of the MeshBlock.
template <int NGHOST>
void Z4c::ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin) {
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
  auto &adm = pmbp->padm->adm;
  auto &opt = pmbp->pz4c->opt;
  int scr_level = 0;
  // 2 1D scratch array and 1 2D scratch array
  Kokkos::Profiling::pushRegion("Region1");
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*2     // 0 tensors
                  + ScrArray2D<Real>::shmem_size(6,ncells1);  // 2D tensor with symm
  par_for_outer("initialize z4c fields",DevExeSpace(),
  scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> detg;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> oopsi4;

      detg.NewAthenaScratchTensor(member, scr_level, ncells1);
    oopsi4.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Kt_dd;
    Kt_dd.NewAthenaScratchTensor(member, scr_level, ncells1);

    par_for_inner(member, isg, ieg, [&](const int i) {
      detg(i) = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      oopsi4(i) = std::pow(detg(i), -1./3.);
      z4c.chi(m,k,j,i) = std::pow(detg(i), 1./12.*opt.chi_psi_power);
    });
    member.team_barrier();

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        z4c.g_dd(m,a,b,k,j,i) = oopsi4(i) * adm.g_dd(m,a,b,k,j,i);
        Kt_dd(a,b,i)          = oopsi4(i) * adm.vK_dd(m,a,b,k,j,i);
      });
    }
    member.team_barrier();

    par_for_inner(member, isg, ieg, [&](const int i) {
      detg(i) = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
      z4c.vKhat(m,k,j,i) = adm::Trace(1.0/detg(i),
                                z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                                Kt_dd(0,0,i), Kt_dd(0,1,i), Kt_dd(0,2,i),
                                Kt_dd(1,1,i), Kt_dd(1,2,i), Kt_dd(2,2,i));
    });
    member.team_barrier();

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        z4c.vA_dd(m,a,b,k,j,i) = Kt_dd(a,b,i) - (1./3.) *
                                  z4c.vKhat(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
      });
    }
  });
  Kokkos::Profiling::popRegion();

  DvceArray5D<Real> g_uu("g_uu", nmb, 6, ncells3, ncells2, ncells1);
  // GLOOP
  scr_size = ScrArray1D<Real>::shmem_size(ncells1);
  par_for_outer("invert z4c metric",DevExeSpace(),
  scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> detg;
    detg.NewAthenaScratchTensor(member, scr_level, ncells1);

    par_for_inner(member, isg, ieg, [&](const int i) {
      detg(i) = adm::SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                                z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                                z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    });
    member.team_barrier();

    par_for_inner(member, isg, ieg, [&](const int i) {
      adm::SpatialInv(1.0/detg(i),
                 z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                 z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                 &g_uu(m,0,k,j,i), &g_uu(m,1,k,j,i), &g_uu(m,2,k,j,i),
                 &g_uu(m,3,k,j,i), &g_uu(m,4,k,j,i), &g_uu(m,5,k,j,i));
    });
  });
  // Compute Gammas
  // Compute only for internal points
  // ILOOP
  int const &IZ4CGAMX = pmbp->pz4c->I_Z4C_GAMX;
  int const &IZ4CGAMY = pmbp->pz4c->I_Z4C_GAMY;
  int const &IZ4CGAMZ = pmbp->pz4c->I_Z4C_GAMZ;
  auto              &u0 = pmbp->pz4c->u0;
  sub_DvceArray5D_0D g_00 = Kokkos::subview(g_uu, Kokkos::ALL, 0,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_01 = Kokkos::subview(g_uu, Kokkos::ALL, 1,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_02 = Kokkos::subview(g_uu, Kokkos::ALL, 2,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_11 = Kokkos::subview(g_uu, Kokkos::ALL, 3,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_12 = Kokkos::subview(g_uu, Kokkos::ALL, 4,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  sub_DvceArray5D_0D g_22 = Kokkos::subview(g_uu, Kokkos::ALL, 5,
                            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  par_for_outer("initialize Gamma",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    // Usage of Dx: pmbp->pz4c->Dx(blockn, posvar, k,j,i, dir, nghost, dx, quantity);
    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    sub_DvceArray5D_0D aux = Kokkos::subview(g_uu,
    Kokkos::ALL, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    par_for_inner(member, is, ie, [&](const int i) {
      u0(m,IZ4CGAMX,k,j,i) = -Dx<NGHOST>(0, idx, g_00, m, k, j, i)  // d/dx g00
                               -Dx<NGHOST>(1, idx, g_01, m, k, j, i)  // d/dy g01
                               -Dx<NGHOST>(2, idx, g_02, m, k, j, i); // d/dz g02
      u0(m,IZ4CGAMY,k,j,i) = -Dx<NGHOST>(0, idx, g_01, m, k, j, i)  // d/dx g01
                               -Dx<NGHOST>(1, idx, g_11, m, k, j, i)  // d/dy g11
                               -Dx<NGHOST>(2, idx, g_12, m, k, j, i); // d/dz g12
      u0(m,IZ4CGAMZ,k,j,i) = -Dx<NGHOST>(0, idx, g_02, m, k, j, i)  // d/dx g01
                               -Dx<NGHOST>(1, idx, g_12, m, k, j, i)  // d/dy g11
                               -Dx<NGHOST>(2, idx, g_22, m, k, j, i); // d/dz g12
    });
  });
  AlgConstr(pmbp);
  return;
}
template void Z4c::ADMToZ4c<2>(MeshBlockPack *pmbp, ParameterInput *pin);
template void Z4c::ADMToZ4c<3>(MeshBlockPack *pmbp, ParameterInput *pin);
template void Z4c::ADMToZ4c<4>(MeshBlockPack *pmbp, ParameterInput *pin);
//----------------------------------------------------------------------------------------
//! \fn void Z4c::Z4cToADM(MeshBlockPack *pmbp)
//! \brief Compute ADM Psi4, g_ij, and K_ij from Z4c variables
//
// This sets the ADM variables everywhere in the MeshBlock
void Z4c::Z4cToADM(MeshBlockPack *pmbp) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;
  auto &opt = pmbp->pz4c->opt;
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);
  par_for_outer("initialize z4c fields",DevExeSpace(),
  scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    par_for_inner(member, isg, ieg, [&](const int i) {
      adm.psi4(m,k,j,i) = std::pow(z4c.chi(m,k,j,i), 4./opt.chi_psi_power);
    });
    member.team_barrier();

    // g_ab
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        adm.g_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
      });
    }
    member.team_barrier();

    // K_ab
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) {
        adm.vK_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.vA_dd(m,a,b,k,j,i) +
          (1./3.) * (z4c.vKhat(m,k,j,i) + 2.*z4c.vTheta(m,k,j,i)) * adm.g_dd(m,a,b,k,j,i);
      });
    }
  });
  return;
}
//----------------------------------------------------------------------------------------
//! \fn void Z4c::ADMConstraints(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat)
//! \brief compute constraints ADM vars
//
// Note: we are assuming that u_adm has been initialized with the correct
// metric and matter quantities
//
// BAM: adm_constraints_N()
// https://git.tpi.uni-jena.de/bamdev/adm
// https://git.tpi.uni-jena.de/bamdev/adm/blob/master/adm_constraints_N.m
//
// The constraints are set only in the MeshBlock interior, because derivatives
// of the ADM quantities are neded to compute them.
template <int NGHOST>
void Z4c::ADMConstraints(MeshBlockPack *pmbp) {
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  //For GLOOPS

  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  auto &adm = pmbp->padm->adm;
  auto &u_con = pmbp->pz4c->u_con;
  Kokkos::deep_copy(u_con, 0.);
  auto &con = pmbp->pz4c->con;
  int scr_level = 1;
  // 2 1D scratch array and 1 2D scratch array
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*4     // 0 tensors
                  + ScrArray2D<Real>::shmem_size(3,ncells1)*2  // vectors
                  + ScrArray2D<Real>::shmem_size(6,ncells1)*3  // 2D tensor with symm
                  + ScrArray2D<Real>::shmem_size(18,ncells1)*6 // 3D tensor with symm
                  + ScrArray2D<Real>::shmem_size(36,ncells1); // 3D tensor with symm
  par_for_outer("ADM constraints loop",DevExeSpace(),
  scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> R;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> K;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> KK;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> detg;

       R.NewAthenaScratchTensor(member, scr_level, ncells1);
       K.NewAthenaScratchTensor(member, scr_level, ncells1);
      KK.NewAthenaScratchTensor(member, scr_level, ncells1);
    detg.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> M_u;

    Gamma_u.NewAthenaScratchTensor(member, scr_level, ncells1);
        M_u.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> K_ud;

    g_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
    R_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
    K_ud.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dK_ddd;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> DK_udd;

       dg_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
       dK_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
    Gamma_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
    Gamma_udd.NewAthenaScratchTensor(member, scr_level, ncells1);
       DK_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
       DK_udd.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;

    ddg_dddd.NewAthenaScratchTensor(member, scr_level, ncells1);

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        dg_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
        dK_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.vK_dd, m,a,b,k,j,i);
      });
    }
    member.team_barrier();

    // second derivatives of g
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = c; d < 3; ++d) {
      if(a == b) {
        par_for_inner(member, is, ie, [&](const int i) {
          ddg_dddd(a,a,c,d,i) = Dxx<NGHOST>(a, idx, adm.g_dd, m,c,d,k,j,i);
        });
      } else {
        par_for_inner(member, is, ie, [&](const int i) {
          ddg_dddd(a,b,c,d,i) = Dxy<NGHOST>(a, b, idx, adm.g_dd, m,c,d,k,j,i);
        });
      }
    }
    member.team_barrier();

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    par_for_inner(member, is, ie, [&](const int i) {
      detg(i) = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      adm::SpatialInv(1./detg(i),
                 adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                 adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                 &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
                 &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    });
    member.team_barrier();

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_ddd(c,a,b,i) = 0.5*(dg_ddd(a,b,c,i) + dg_ddd(b,a,c,i) - dg_ddd(c,a,b,i));
      });
    }
    member.team_barrier();

    Gamma_udd.ZeroClear();
    member.team_barrier();
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int d = 0; d < 3; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      });
    }
    member.team_barrier();

    Gamma_u.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      });
    }
    member.team_barrier();

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //
    R.ZeroClear();
    R_dd.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        // Part with the Christoffel symbols
        for(int e = 0; e < 3; ++e) {
          par_for_inner(member, is, ie, [&](const int i) {
            R_dd(a,b,i) += g_uu(c,d,i) * Gamma_udd(e,a,c,i) * Gamma_ddd(e,b,d,i);
            R_dd(a,b,i) -= g_uu(c,d,i) * Gamma_udd(e,a,b,i) * Gamma_ddd(e,c,d,i);
          });
        }
        // Wave operator part of the Ricci
        par_for_inner(member, is, ie, [&](const int i) {
          R_dd(a,b,i) += 0.5*g_uu(c,d,i)*(
              - ddg_dddd(c,d,a,b,i) - ddg_dddd(a,b,c,d,i) +
                ddg_dddd(a,c,b,d,i) + ddg_dddd(b,c,a,d,i));
        });
      }
      member.team_barrier();
      par_for_inner(member, is, ie, [&](const int i) {
        R(i) += g_uu(a,b,i) * R_dd(a,b,i);
      });
    }
    member.team_barrier();

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    K.ZeroClear();
    K_ud.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a) {
      for(int b = a; b < 3; ++b) {
        for(int c = 0; c < 3; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            K_ud(a,b,i) += g_uu(a,c,i) * adm.vK_dd(m,c,b,k,j,i);
          });
        }
      }
      member.team_barrier();
      par_for_inner(member, is, ie, [&](const int i) {
        K(i) += K_ud(a,a,i);
      });
    }
    member.team_barrier();

    // K^a_b K^b_a
    KK.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        KK(i) += K_ud(a,b,i) * K_ud(b,a,i);
      });
    }
    member.team_barrier();
    // Covariant derivative of K
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = b; c < 3; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        DK_ddd(a,b,c,i) = dK_ddd(a,b,c,i);
      });
      for(int d = 0; d < 3; ++d) {
        par_for_inner(member, is, ie, [&](const int i) {
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,b,i) * adm.vK_dd(m,d,c,k,j,i);
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,c,i) * adm.vK_dd(m,b,d,k,j,i);
        });
      }
    }
    member.team_barrier();

    DK_udd.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = b; c < 3; ++c)
    for(int d = 0; d < 3; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        DK_udd(a,b,c,i) += g_uu(a,d,i) * DK_ddd(d,b,c,i);
      });
    }
    member.team_barrier();

    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Hamiltonian constraint
    //
    par_for_inner(member, is, ie, [&](const int i) {
      con.H(m,k,j,i) = R(i) + SQR(K(i)) - KK(i);// - 16*M_PI * mat.rho(k,j,i);
    });
    member.team_barrier();

    // Momentum constraint (contravariant)
    //
    M_u.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      //M_u(a,i) -= 8*M_PI * g_uu(a,b,i) * mat.S_d(b,k,j,i);
      for(int c = 0; c < 3; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          M_u(a,i) += g_uu(a,b,i) * DK_udd(c,b,c,i);
          M_u(a,i) -= g_uu(b,c,i) * DK_udd(a,b,c,i);
        });
      }
    }
    member.team_barrier();

    // Momentum constraint (covariant)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        con.M_d(m,a,k,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(b,i);
      });
    }
    // Momentum constraint (norm squared)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        con.M(m,k,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(a,i) * M_u(b,i);
      });
    }
    // Constraint violation Z (norm squared)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        con.Z(m,k,j,i) += 0.25*adm.g_dd(m,a,b,k,j,i)
                          *(z4c.vGam_u(m,a,k,j,i) - Gamma_u(a,i))
                          *(z4c.vGam_u(m,b,k,j,i) - Gamma_u(b,i));
      });
    }
    member.team_barrier();
    // Constraint violation monitor C^2
    par_for_inner(member, is, ie, [&](const int i) {
      con.C(m,k,j,i) = SQR(con.H(m,k,j,i)) + con.M(m,k,j,i) +
      SQR(z4c.vTheta(m,k,j,i)) + 4.0*con.Z(m,k,j,i);
    });
});
}
template void Z4c::ADMConstraints<2>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<3>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<4>(MeshBlockPack *pmbp);
} // namespace z4c
