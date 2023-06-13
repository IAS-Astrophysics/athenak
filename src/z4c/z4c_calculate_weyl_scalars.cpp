//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class related to calculation of Weyl scalars

// C++ standard headers
//#include <iostream>
#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "adm/adm.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {
//----------------------------------------------------------------------------------------
// \!fn void Z4c::Z4cWeyl(MeshBlockPack *pmbp)
// \brief compute the weyl scalars given the adm variables and matter state
//
// This function operates only on the interior points of the MeshBlock
template <int NGHOST>
void Z4c::Z4cWeyl(MeshBlockPack *pmbp) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1+indcs.ng; // Align scratch buffers with variables
  int ncells2 = indcs.nx2;
  int ncells3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;

  auto &z4c = pmy_pack->pz4c->z4c;
  auto &adm = pmy_pack->pz4c->adm;
  auto &opt = pmy_pack->pz4c->opt;
  int scr_level = 1;
  // 2 1D scratch array and 1 2D scratch array
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*6   // 0 tensors
                  + ScrArray2D<Real>::shmem_size(3,ncells1)*4 // vectors
                  + ScrArray2D<Real>::shmem_size(6,ncells1)*4 // 2D tensor with symm
                  + ScrArray2D<Real>::shmem_size(9,ncells1)*0  // 2D tensor with no symm
                  + ScrArray2D<Real>::shmem_size(18,ncells1)*6 // 3D tensor with symm
                  + ScrArray2D<Real>::shmem_size(36,ncells1)*1;  // 4D tensor with symm
  // Check symmetries of Riemann tensor! Now it is NONE!
  par_for_outer("z4c_weyl_scalar",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    // Simplify constants (2 & sqrt 2 factors) featured in re/im[psi4]
    const Real FR4 = 0.25;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> detg;         // det(g)
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> R;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> dotp1;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> dotp2;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> K;            // trace of extrinsic curvature
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> KK;           // K^a_b K^b_a


    detg.NewAthenaScratchTensor(member, scr_level, ncells1);
       R.NewAthenaScratchTensor(member, scr_level, ncells1);
       K.NewAthenaScratchTensor(member, scr_level, ncells1);
   dotp1.NewAthenaScratchTensor(member, scr_level, ncells1);
   dotp2.NewAthenaScratchTensor(member, scr_level, ncells1);
      KK.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> uvec;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> vvec;
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> wvec;

 Gamma_u.NewAthenaScratchTensor(member, scr_level, ncells1);
    uvec.NewAthenaScratchTensor(member, scr_level, ncells1);
    vvec.NewAthenaScratchTensor(member, scr_level, ncells1);
    wvec.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;        // inverse of conf. metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;        // Ricci tensor
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> K_ud;        // extrinsic curvature
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> Riemm4_dd;   // 4D Riemann *n^a*n^c


         g_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
         R_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
         K_ud.NewAthenaScratchTensor(member, scr_level, ncells1);
    Riemm4_dd.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2,  3, 3> dg_ddd;      // metric 1st drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2,  3, 3> dK_ddd;      // K 1st drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;   // Christoffel symbols of 1st kind
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;   // Christoffel symbols of 2nd kind
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;      // differential of K
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> DK_udd;      // differential of K
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 3> Riemm4_ddd;  // 4D Riemann * n^a


       dg_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
       dK_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
    Gamma_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
    Gamma_udd.NewAthenaScratchTensor(member, scr_level, ncells1);
       DK_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
       DK_udd.NewAthenaScratchTensor(member, scr_level, ncells1);
   Riemm4_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);

     AthenaScratchTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;   // metric 2nd drvts
     AthenaScratchTensor<Real, TensorSymm::NONE, 3, 4> Riem3_dddd;  // 3D Riemann tensor
     AthenaScratchTensor<Real, TensorSymm::NONE, 3, 4> Riemm4_dddd; // 4D Riemann tensor

      ddg_dddd.NewAthenaScratchTensor(member, scr_level, ncells1);    
    Riem3_dddd.NewAthenaScratchTensor(member, scr_level, ncells1);    
   Riemm4_dddd.NewAthenaScratchTensor(member, scr_level, ncells1);    

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};

    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        dg_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
        dK_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.vK_dd, m,a,b,k,j,i);
      });
    }
    // second derivatives of g
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = 0; d < 3; ++d) {
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
      detg(i) = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                           adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      adm::SpatialInv(1.0/detg(i),
                 adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                 adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                 &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
                 &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    });


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

    Gamma_u.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      });
    }

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
      par_for_inner(member, is, ie, [&](const int i) {
        R(i) += g_uu(a,b,i) * R_dd(a,b,i);
      });
    }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //
    K.ZeroClear();
    K_ud.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        for(int c = 0; c < 3; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            K_ud(a,b,i) += g_uu(a,c,i) * adm.vK_dd(m,c,b,k,j,i);
          });
        }
      }
      par_for_inner(member, is, ie, [&](const int i) {
        K(i) += K_ud(a,a,i);
      });
    }
    // K^a_b K^b_a
    KK.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        KK(i) += K_ud(a,b,i) * K_ud(b,a,i);
      });
    }
    // Covariant derivative of K
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
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
    DK_udd.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = 0; d < 3; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        DK_udd(a,b,c,i) += g_uu(a,d,i) * DK_ddd(d,b,c,i);
      });
    }

    //------------------------------------------------------------------------------------
    //     Construct tetrad
    //
    //     Initial tetrad guess. NB, aligned with z axis - possible problem if points lie on z axis
    //     theta and phi vectors degenerate
    //     Like BAM start with phi vector
    //     uvec = radial vec
    //     vvec = theta vec
    //     wvec = phi vec
    par_for_inner(member, is, ie, [&](const int i){
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real xx = x1v;
      if(SQR(x1v) +  SQR(x2v) < 1e-10)
        xx = xx + 1e-8;
      uvec(0,i) = xx;
      uvec(1,i) = x2v;
      uvec(2,i) = x3v;
      vvec(0,i) = xx*x3v;
      vvec(1,i) = x2v*x3v;
      vvec(2,i) = -SQR(xx)-SQR(x2v);
      wvec(0,i) = x2v*-1.0;
      wvec(1,i) = xx;
      wvec(2,i) = 0.0;
    });


    //Gram-Schmidt orthonormalisation with spacetime metric.
    //
    dotp1.ZeroClear();
    member.team_barrier();
    for(int a = 0; a<3; ++a){
	    for(int b = 0; b<3; ++b){
	      par_for_inner(member, is, ie, [&](const int i){
          dotp1(i) += adm.g_dd(m,a,b,k,j,i)*wvec(a,i)*wvec(b,i);
	      });
	    }
    }
    for(int a =0; a<3; ++a){
      par_for_inner(member, is, ie, [&](const int i){
	      wvec(a,i) = wvec(a,i)/std::sqrt(dotp1(i));
      });
    }

    dotp1.ZeroClear();
    member.team_barrier();
    for(int a = 0; a<3; ++a){
      for( int b = 0; b<3; ++b){
	      par_for_inner(member, is, ie, [&](const int i){
	        dotp1(i) += adm.g_dd(m,a,b,k,j,i)*wvec(a,i)*uvec(b,i);
	      });
      }
    }
    for(int a = 0; a<3; ++a){
      par_for_inner(member, is, ie, [&](const int i){
	      uvec(a,i) -= dotp1(i)*wvec(a,i);
	    });
    }
    dotp1.ZeroClear();
    member.team_barrier();
    for(int a = 0; a<3; ++a){
	    for(int b = 0; b<3; ++b) {
	      par_for_inner(member, is, ie, [&](const int i){
	        dotp1(i) += adm.g_dd(m,a,b,k,j,i)*uvec(a,i)*uvec(b,i);
	      });
	    }
    }

    for(int a =0; a<3; ++a){
	    par_for_inner(member, is, ie, [&](const int i){
	      uvec(a,i) = uvec(a,i)/std::sqrt(dotp1(i));
	    });
    }

    dotp1.ZeroClear();
    member.team_barrier();
    for(int a = 0; a<3; ++a){
      for(int b = 0; b<3; ++b) {
	      par_for_inner(member, is, ie, [&](const int i){
	        dotp1(i) += adm.g_dd(m,a,b,k,j,i)*wvec(a,i)*vvec(b,i);
	      });
	    }
    }
    dotp2.ZeroClear();
    member.team_barrier();
    for(int a = 0; a<3; ++a){
	    for( int b = 0; b<3; ++b) {
	      par_for_inner(member, is, ie, [&](const int i){
	        dotp2(i) += adm.g_dd(m,a,b,k,j,i)*uvec(a,i)*vvec(b,i);
	      });
	    }
    }

    for(int a = 0; a<3; ++a){
	    par_for_inner(member, is, ie, [&](const int i){
	      vvec(a,i) -= dotp1(i)*wvec(a,i)+dotp2(i)*uvec(a,i);
	    });
    }

    dotp1.ZeroClear();
    member.team_barrier();
    for(int a = 0; a<3; ++a){
	    for( int b = 0; b<3; ++b) {
	      par_for_inner(member, is, ie, [&](const int i){
	        dotp1(i) += adm.g_dd(m,a,b,k,j,i)*vvec(a,i)*vvec(b,i);
	      });
	    }
    }

    for(int a =0; a<3; ++a){
	    par_for_inner(member, is, ie, [&](const int i){
	      vvec(a,i) = vvec(a,i)/std::sqrt(dotp1(i));
	    });
    }

    //   Riem3_dddd = Riemann tensor of spacelike hypersurface
    //   Riemm4_dddd = Riemann tensor of 4D spacetime
    //   Riemm4_ddd  = Riemann tensor of 4D spacetime contracted once with n
    //   Riemm4_dd  = Riemann tensor of 4D spacetime contracted twice with n
    Riem3_dddd.ZeroClear();
    Riemm4_dddd.ZeroClear();
    Riemm4_ddd.ZeroClear();
    Riemm4_dd.ZeroClear();
    member.team_barrier();
    for(int a = 0; a < 3; ++a){
      for(int b = 0; b < 3; ++b){
        for(int c = 0; c < 3; ++c){
          for(int d = 0; d < 3; ++d){
            par_for_inner(member, is, ie, [&](const int i){
              Riem3_dddd(a,b,c,d,i) = adm.g_dd(m,a,c,k,j,i)*R_dd(b,d,i) 
	                            + adm.g_dd(m,b,d,k,j,i)*R_dd(a,c,i)
                                    - adm.g_dd(m,a,d,k,j,i)*R_dd(b,c,i) 
				    - adm.g_dd(m,b,c,k,j,i)*R_dd(a,d,i)
                                    - 0.5*R(i)*adm.g_dd(m,a,c,k,j,i)*adm.g_dd(m,b,d,k,j,i)
                                    + 0.5*R(i)*adm.g_dd(m,a,d,k,j,i)*adm.g_dd(m,b,c,k,j,i);
              Riemm4_dddd(a,b,c,d,i) = Riem3_dddd(a,b,c,d,i) 
	                             + adm.vK_dd(m,a,c,k,j,i)*adm.vK_dd(m,b,d,k,j,i)
                                     - adm.vK_dd(m,a,d,k,j,i)*adm.vK_dd(m,b,c,k,j,i);
            });
          }
        }
      }
    }

    for(int a = 0; a < 3; ++a){
      for(int b = 0; b < 3; ++b){
        for(int c = 0; c < 3; ++c){
          par_for_inner(member, is, ie, [&](const int i){
            Riemm4_ddd(a,b,c,i) = - (DK_ddd(c,a,b,i) - DK_ddd(b,a,c,i));
          });
        }
      }
    }


    for(int a = 0; a < 3; ++a){
      for(int b = 0; b < 3; ++b){
        par_for_inner(member, is, ie, [&](const int i){
          Riemm4_dd(a,b,i) = R_dd(a,b,i) + K(i)*adm.vK_dd(m,a,b,k,j,i);
        });
        for(int c = 0; c < 3; ++c){
          for(int d = 0; d < 3; ++d){
            par_for_inner(member, is, ie, [&](const int i){
              Riemm4_dd(a,b,i) += - g_uu(c,d,i)*adm.vK_dd(m,a,c,k,j,i)
	                                       *adm.vK_dd(m,d,b,k,j,i);
            });
          }
        }
      }
    }

    for(int a = 0; a < 3; ++a){
      for(int b = 0; b < 3; ++b){
        par_for_inner(member, is, ie, [&](const int i){
          weyl.rpsi4(m,k,j,i) += - FR4 * Riemm4_dd(a,b,i) * (
            vvec(a,i) * vvec(b,i) - (-wvec(a,i) * (-wvec(b,i)))
          );
          weyl.ipsi4(m,k,j,i) += - FR4 * Riemm4_dd(a,b,i) * (
            -vvec(a,i) * wvec(b,i) - wvec(a,i)*vvec(b,i)
          );
        });
        for(int c = 0; c < 3; ++c){
          par_for_inner(member, is, ie, [&](const int i){
            weyl.rpsi4(m,k,j,i) += 0.5 * Riemm4_ddd(a,c,b,i) * uvec(c,i) * (
              vvec(a,i) * vvec(b,i) - (-wvec(a,i)*(-wvec(b,i)))
            );
            weyl.ipsi4(m,k,j,i) += 0.5 * Riemm4_ddd(a,c,b,i) * uvec(c,i) * (
              -vvec(a,i) * wvec(b,i) - wvec(a,i)*vvec(b,i)
            );
          });
          for(int d = 0; d < 3; ++d){
            par_for_inner(member, is, ie, [&](const int i){
              weyl.rpsi4(m,k,j,i) += -FR4 * (Riemm4_dddd(d,a,c,b,i) * uvec(d,i) * uvec(c,i)) * (
                vvec(a,i) * vvec(b,i) - (-wvec(a,i)*(-wvec(b,i)))
              );
              weyl.ipsi4(m,k,j,i) += -FR4 * (Riemm4_dddd(d,a,c,b,i) * uvec(d,i) * uvec(c,i)) * (
                -vvec(a,i) * wvec(b,i) - wvec(a,i)*vvec(b,i)
              );
            });
          }
        }
      }
    }
   });
}
template void Z4c::Z4cWeyl<2>(MeshBlockPack *pmbp);
template void Z4c::Z4cWeyl<3>(MeshBlockPack *pmbp);
template void Z4c::Z4cWeyl<4>(MeshBlockPack *pmbp);
} // namespace z4c

