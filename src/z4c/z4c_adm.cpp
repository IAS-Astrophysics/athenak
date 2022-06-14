//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm_z4c.cpp
//  \brief implementation of functions in the Z4c class related to ADM decomposition


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

template <typename TYPE>
KOKKOS_INLINE_FUNCTION
Real Dx_(int const ind, int const nghost, Real const &idx, const TYPE &quant, MeshBlockPack *pmbp)
{
return (-0.5*quant(ind-1)+0.5*quant(ind+1))*idx; //Multiply by inverse
}

// \!fn void Z4c::ADMToZ4c(MeshBlockPack *pmbp, ParameterInput *pin)
// \brief Compute Z4c variables from ADM variables
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
  int &NDIM = pmbp->pz4c->NDIM; 
  int scr_level = 0;
  // 2 1D scratch array and 1 2D scratch array
  Kokkos::Profiling::pushRegion("Region1");
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*2     // 0 tensors
                  + ScrArray2D<Real>::shmem_size(6,ncells1);  // 2D tensor with symm
  par_for_outer("initialize z4c fields",DevExeSpace(),scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {  
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> detg;
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> oopsi4;
    
      detg.NewAthenaTensor(member, scr_level, ncells1);
    oopsi4.NewAthenaTensor(member, scr_level, ncells1);
    
    AthenaTensor<Real, TensorSymm::SYM2, 1, 2> Kt_dd;
    Kt_dd.NewAthenaTensor(member, scr_level, ncells1);

    par_for_inner(member, isg, ieg, [&](const int i) { 
      detg(i) = SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                           adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      oopsi4(i) = std::pow(detg(i), -1./3.);
      z4c.chi(m,k,j,i) = std::pow(detg(i), 1./12.*opt.chi_psi_power);
    });

    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) { 
        z4c.g_dd(m,a,b,k,j,i) = oopsi4(i) * adm.g_dd(m,a,b,k,j,i);
        Kt_dd(a,b,i)          = oopsi4(i) * adm.K_dd(m,a,b,k,j,i);
      });
    }

    par_for_inner(member, isg, ieg, [&](const int i) {
      detg(i) = SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                           z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
      z4c.Khat(m,k,j,i) = Trace(1.0/detg(i),
                                z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                                z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                                Kt_dd(0,0,i), Kt_dd(0,1,i), Kt_dd(0,2,i),
                                Kt_dd(1,1,i), Kt_dd(1,2,i), Kt_dd(2,2,i));
    });

    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) { 
        z4c.A_dd(m,a,b,k,j,i) = Kt_dd(a,b,i) - (1./3.) * z4c.Khat(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
      });
    }
  });
  Kokkos::Profiling::popRegion();

  DvceArray5D<Real> g_uu("g_uu", nmb, 6, ncells3, ncells2, ncells1);
  // GLOOP
  scr_size = ScrArray1D<Real>::shmem_size(ncells1); 
  par_for_outer("invert z4c metric",DevExeSpace(),scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {  
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> detg;
    
      detg.NewAthenaTensor(member, scr_level, ncells1);

    par_for_inner(member, isg, ieg, [&](const int i) { 
      detg(i) = SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                           z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
    });
    par_for_inner(member, isg, ieg, [&](const int i) {
      SpatialInv(1.0/detg(i),
                 z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                 z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
                 &g_uu(m,0,k,j,i), &g_uu(m,1,k,j,i), &g_uu(m,2,k,j,i),
                 &g_uu(m,3,k,j,i), &g_uu(m,4,k,j,i), &g_uu(m,5,k,j,i));
    });
  });
  // Compute Gammas
  // Compute only for internal points
  // ILOOP
  int const &I_Z4c_Gamx = pmbp->pz4c->I_Z4c_Gamx;
  int const &I_Z4c_Gamy = pmbp->pz4c->I_Z4c_Gamy;
  int const &I_Z4c_Gamz = pmbp->pz4c->I_Z4c_Gamz;
  auto              &u0 = pmbp->pz4c->u0;
  par_for_outer("initialize Gamma",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {  
    // Usage of Dx: pmbp->pz4c->Dx(blockn, posvar, k,j,i, dir, nghost, dx, quantity);
    Real &idx1 = size.d_view(m).idx1;
    Real &idx2 = size.d_view(m).idx2;
    Real &idx3 = size.d_view(m).idx3;
    par_for_inner(member, is, ie, [&](const int i) { 
      u0(m,I_Z4c_Gamx,k,j,i) = -Dx_(i, indcs.ng-1, idx1, 
                                    Kokkos::subview(g_uu, m, 0, k, j, Kokkos::ALL), pmbp)  // d/dx g00
                               -Dx_(j, indcs.ng, idx2, 
                                    Kokkos::subview(g_uu, m, 1, k, Kokkos::ALL, i), pmbp)  // d/dy g01
                               -Dx_(k, indcs.ng, idx3, 
                                    Kokkos::subview(g_uu, m, 2, Kokkos::ALL, j, i), pmbp); // d/dz g02
      u0(m,I_Z4c_Gamy,k,j,i) = -Dx_(i, indcs.ng-1, idx1,
                                    Kokkos::subview(g_uu, m, 1, k, j, Kokkos::ALL), pmbp)  // d/dx g01
                               -Dx_(j, indcs.ng-1, idx2, 
                                    Kokkos::subview(g_uu, m, 3, k, Kokkos::ALL, i), pmbp)  // d/dy g11
                               -Dx_(k, indcs.ng-1, idx3, 
                                    Kokkos::subview(g_uu, m, 4, Kokkos::ALL, j, i), pmbp); // d/dz g12
      u0(m,I_Z4c_Gamz,k,j,i) = -Dx_(i, indcs.ng-1, idx1, 
                                    Kokkos::subview(g_uu, m, 2, k, j, Kokkos::ALL), pmbp)  // d/dx g02
                               -Dx_(j, indcs.ng-1, idx2, 
                                    Kokkos::subview(g_uu, m, 4, k, Kokkos::ALL, i), pmbp)  // d/dy g12
                               -Dx_(k, indcs.ng-1, idx3, 
                                    Kokkos::subview(g_uu, m, 5, Kokkos::ALL, j, i), pmbp); // d/dz g22
    });
  });
  AlgConstr(pmbp);
  return;
}
//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cToADM(MeshBlockPack *pmbp)
// \brief Compute ADM Psi4, g_ij, and K_ij from Z4c variables
//
// This sets the ADM variables everywhere in the MeshBlock
void Z4c::Z4cToADM(MeshBlockPack *pmbp) {
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
  int &NDIM = pmbp->pz4c->NDIM; 
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);
  par_for_outer("initialize z4c fields",DevExeSpace(),scr_size,scr_level,0,nmb-1,ksg,keg,jsg,jeg,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {  
    par_for_inner(member, isg, ieg, [&](const int i) { 
      adm.psi4(m,k,j,i) = std::pow(z4c.chi(m,k,j,i), 4./opt.chi_psi_power);
    });
    // g_ab
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      par_for_inner(member, isg, ieg, [&](const int i) { 
        adm.g_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.g_dd(m,a,b,k,j,i);
      });
    }
    // K_ab 
    for(int a = 0; a < NDIM; ++a) 
    for(int b = a; b < NDIM; ++b) { 
      par_for_inner(member, isg, ieg, [&](const int i) { 
        adm.K_dd(m,a,b,k,j,i) = adm.psi4(m,k,j,i) * z4c.A_dd(m,a,b,k,j,i) + 
          (1./3.) * (z4c.Khat(m,k,j,i) + 2.*z4c.Theta(m,k,j,i)) * adm.g_dd(m,a,b,k,j,i); 
      });
    }
  });
  return;
}
//----------------------------------------------------------------------------------------
// \!fn void Z4c::ADMConstraints(AthenaArray<Real> & u_adm, AthenaArray<Real> & u_mat)
// \brief compute constraints ADM vars
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
  auto &u_con = pmbp->pz4c->u_con;
  Kokkos::deep_copy(u_con, 0.);
  auto &con = pmbp->pz4c->con;
  int &NDIM = pmbp->pz4c->NDIM;
  int scr_level = 1;
  // 2 1D scratch array and 1 2D scratch array
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*4     // 0 tensors
                  + ScrArray2D<Real>::shmem_size(3,ncells1)*2  // vectors
                  + ScrArray2D<Real>::shmem_size(6,ncells1)*3  // 2D tensor with symm
                  + ScrArray2D<Real>::shmem_size(18,ncells1)*6 // 3D tensor with symm
                  + ScrArray2D<Real>::shmem_size(36,ncells1); // 3D tensor with symm
  par_for_outer("ADM constraints loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {  
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> R;
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> K;
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> KK;
    AthenaTensor<Real, TensorSymm::NONE, 1, 0> detg;
    
       R.NewAthenaTensor(member, scr_level, ncells1);
       K.NewAthenaTensor(member, scr_level, ncells1);
      KK.NewAthenaTensor(member, scr_level, ncells1);
    detg.NewAthenaTensor(member, scr_level, ncells1);

    AthenaTensor<Real, TensorSymm::NONE, 1, 1> Gamma_u;
    AthenaTensor<Real, TensorSymm::NONE, 1, 1> M_u;

    Gamma_u.NewAthenaTensor(member, scr_level, ncells1);
        M_u.NewAthenaTensor(member, scr_level, ncells1);
    
    AthenaTensor<Real, TensorSymm::SYM2, 1, 2> g_uu;
    AthenaTensor<Real, TensorSymm::SYM2, 1, 2> R_dd;
    AthenaTensor<Real, TensorSymm::SYM2, 1, 2> K_ud;
    
    g_uu.NewAthenaTensor(member, scr_level, ncells1);
    R_dd.NewAthenaTensor(member, scr_level, ncells1);
    K_ud.NewAthenaTensor(member, scr_level, ncells1);
    
    AthenaTensor<Real, TensorSymm::SYM2, 1, 3> dg_ddd;
    AthenaTensor<Real, TensorSymm::SYM2, 1, 3> dK_ddd;
    AthenaTensor<Real, TensorSymm::SYM2, 1, 3> Gamma_ddd;
    AthenaTensor<Real, TensorSymm::SYM2, 1, 3> Gamma_udd;
    AthenaTensor<Real, TensorSymm::SYM2, 1, 3> DK_ddd;
    AthenaTensor<Real, TensorSymm::SYM2, 1, 3> DK_udd;

       dg_ddd.NewAthenaTensor(member, scr_level, ncells1);
       dK_ddd.NewAthenaTensor(member, scr_level, ncells1);
    Gamma_ddd.NewAthenaTensor(member, scr_level, ncells1);
    Gamma_udd.NewAthenaTensor(member, scr_level, ncells1);
       DK_ddd.NewAthenaTensor(member, scr_level, ncells1);
       DK_udd.NewAthenaTensor(member, scr_level, ncells1);
    
    AthenaTensor<Real, TensorSymm::SYM22, 1, 4> ddg_dddd;

    ddg_dddd.NewAthenaTensor(member, scr_level, ncells1);
    
    
    Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
    int ord_der = indcs.ng;
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        dg_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
        dK_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.K_dd, m,a,b,k,j,i);
      });
    }
    
    // second derivatives of g
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c)
    for(int d = c; d < NDIM; ++d) {
      if(a == b) {
        par_for_inner(member, is, ie, [&](const int i) {
          ddg_dddd(a,a,c,d,i) = Dxx<NGHOST>(a, idx, adm.g_dd, m,c,d,k,j,i);
        });
      }
      else {
        par_for_inner(member, is, ie, [&](const int i) {
          ddg_dddd(a,b,c,d,i) = Dxy<NGHOST>(a, b, idx, adm.g_dd, m,c,d,k,j,i);
        });
      }
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    par_for_inner(member, is, ie, [&](const int i) {
      detg(i) = SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                           adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      SpatialInv(1./detg(i),
                 adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                 adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                 &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
                 &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    
    });

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_ddd(c,a,b,i) = 0.5*(dg_ddd(a,b,c,i) + dg_ddd(b,a,c,i) - dg_ddd(c,a,b,i));
      });
    }
    
    Gamma_udd.ZeroClear();
    for(int c = 0; c < NDIM; ++c)
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b)
    for(int d = 0; d < NDIM; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      });
    }

    Gamma_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = 0; c < NDIM; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      });
    }

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //
    R.ZeroClear();
    R_dd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = a; b < NDIM; ++b) {
      for(int c = 0; c < NDIM; ++c)
      for(int d = 0; d < NDIM; ++d) {
        // Part with the Christoffel symbols
        for(int e = 0; e < NDIM; ++e) {
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
      par_for_inner(member, is, ie, [&](const int i) {
        R(i) += g_uu(a,b,i) * R_dd(a,b,i);
      });
    }
    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    K.ZeroClear();
    K_ud.ZeroClear();
    for(int a = 0; a < NDIM; ++a) {
      for(int b = a; b < NDIM; ++b) {
        for(int c = 0; c < NDIM; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            K_ud(a,b,i) += g_uu(a,c,i) * adm.K_dd(m,c,b,k,j,i);
          });
        }
      }
      par_for_inner(member, is, ie, [&](const int i) {
        K(i) += K_ud(a,a,i);
      });
    }
    // K^a_b K^b_a
    KK.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        KK(i) += K_ud(a,b,i) * K_ud(b,a,i);
      });
    }
    // Covariant derivative of K
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = b; c < NDIM; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        DK_ddd(a,b,c,i) = dK_ddd(a,b,c,i);
      });
      for(int d = 0; d < NDIM; ++d) {
        par_for_inner(member, is, ie, [&](const int i) {
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,b,i) * adm.K_dd(m,d,c,k,j,i);
          DK_ddd(a,b,c,i) -= Gamma_udd(d,a,c,i) * adm.K_dd(m,b,d,k,j,i);
        });
      }
    }
    DK_udd.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b)
    for(int c = b; c < NDIM; ++c)
    for(int d = 0; d < NDIM; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        DK_udd(a,b,c,i) += g_uu(a,d,i) * DK_ddd(d,b,c,i);
      });
    }
    // -----------------------------------------------------------------------------------
    // Actual constraints
    //
    // Hamiltonian constraint
    //
    par_for_inner(member, is, ie, [&](const int i) {
      con.H(m,k,j,i) = R(i) + SQR(K(i)) - KK(i);// - 16*M_PI * mat.rho(k,j,i);
    });
    // Momentum constraint (contravariant)
    //
    M_u.ZeroClear();
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      //M_u(a,i) -= 8*M_PI * g_uu(a,b,i) * mat.S_d(b,k,j,i);
      for(int c = 0; c < NDIM; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          M_u(a,i) += g_uu(a,b,i) * DK_udd(c,b,c,i);
          M_u(a,i) -= g_uu(b,c,i) * DK_udd(a,b,c,i);
        });
      }
    }
    // Momentum constraint (covariant)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        con.M_d(m,a,k,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(b,i);
      });
    }
    // Momentum constraint (norm squared)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        con.M(m,k,j,i) += adm.g_dd(m,a,b,k,j,i) * M_u(a,i) * M_u(b,i);
      });
    }
    // Constraint violation Z (norm squared)
    for(int a = 0; a < NDIM; ++a)
    for(int b = 0; b < NDIM; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        con.Z(m,k,j,i) += 0.25*adm.g_dd(m,a,b,k,j,i)*(z4c.Gam_u(m,a,k,j,i) - Gamma_u(a,i))
                                                    *(z4c.Gam_u(m,b,k,j,i) - Gamma_u(b,i));
      });
    }
    // Constraint violation monitor C^2
    par_for_inner(member, is, ie, [&](const int i) {
      con.C(m,k,j,i) = SQR(con.H(m,k,j,i)) + con.M(m,k,j,i) + SQR(z4c.Theta(m,k,j,i)) + 4.0*con.Z(m,k,j,i);
    });
});
}
template void Z4c::ADMConstraints<2>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<3>(MeshBlockPack *pmbp);
template void Z4c::ADMConstraints<4>(MeshBlockPack *pmbp);
}
