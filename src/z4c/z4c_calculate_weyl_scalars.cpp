//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_weyl_scalars.cpp
//  \brief implementation of functions in the Z4c class related to calculation
//  of Weyl scalars

// C++ standard headers
//#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
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
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  auto &adm = pmbp->padm->adm;
  auto &weyl = pmbp->pz4c->weyl;
  auto &u_weyl = pmbp->pz4c->u_weyl;
  Kokkos::deep_copy(u_weyl, 0.);

  par_for("z4c_weyl_scalar",DevExeSpace(),0,nmb-1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Simplify constants (2 & sqrt 2 factors) featured in re/im[psi4]
    const Real FR4 = 0.25;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Scalars
    Real detg = 0.0;         // det(g)
    Real R = 0.0;
    Real dotp1 = 0.0;
    Real dotp2 = 0.0;
    Real K = 0.0;            // trace of extrinsic curvature
    Real KK = 0.0;           // K^a_b K^b_a

    // Vectors
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> uvec;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> vvec;
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> wvec;
    for (int a = 0; a < 3; ++a) {
      Gamma_u(a) = 0.0;
      uvec(a) = 0.0;
      vvec(a) = 0.0;
      wvec(a) = 0.0;
    }

    // Symmetric tensors
    // Rank 2
    // inverse of conf. metric
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;        // Ricci tensor
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> K_ud;        // extrinsic curvature

    // Rank 3
    AthenaPointTensor<Real, TensorSymm::SYM2,  3, 3> dg_ddd;      // metric 1st drvts
    AthenaPointTensor<Real, TensorSymm::SYM2,  3, 3> dK_ddd;      // K 1st drvts
    // Christoffel symbols of 1st kind
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
    // Christoffel symbols of 2nd kind
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;      // differential of K
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> DK_udd;      // differential of K

    // Rank 4
    AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;   // metric 2nd drvts
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      g_uu(a,b) = 0.0;
      R_dd(a,b) = 0.0;
      for (int c = 0; c < 3; ++c) {
        dg_ddd(c,a,b) = 0.0;
        dK_ddd(c,a,b) = 0.0;
        Gamma_ddd(c,a,b) = 0.0;
        Gamma_udd(c,a,b) = 0.0;
        DK_ddd(c,a,b) = 0.0;
        DK_udd(c,a,b) = 0.0;
        for (int d = c; d < 3; ++d) {
          ddg_dddd(c,d,a,b) = 0.0;
        }
      }
    }

    // Generic tensors
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> Riemm4_dd;   // 4D Riemann *n^a*n^c
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 3> Riemm4_ddd;  // 4D Riemann * n^a
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 4> Riem3_dddd;  // 3D Riemann tensor
    AthenaPointTensor<Real, TensorSymm::NONE, 3, 4> Riemm4_dddd; // 4D Riemann tensor
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      Riemm4_dd(a,b) = 0.0;
      for (int c = 0; c < 3; ++c) {
        Riemm4_ddd(c,a,b) = 0.0;
        for (int d = 0; d < 3; ++d) {
          Riem3_dddd(a,b,c,d) = 0.0;
          Riemm4_dddd(a,b,c,d) = 0.0;
        }
      }
    }

    Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
    // -----------------------------------------------------------------------------------
    // derivatives
    //
    // first derivatives of g and K
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      dg_ddd(c,a,b) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
      dK_ddd(c,a,b) = Dx<NGHOST>(c, idx, adm.vK_dd, m,a,b,k,j,i);
    }
    // second derivatives of g
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = c; d < 3; ++d) {
      if(a == b) {
        ddg_dddd(a,b,c,d) = Dxx<NGHOST>(a, idx, adm.g_dd, m,c,d,k,j,i);
      } else {
        ddg_dddd(a,b,c,d) = Dxy<NGHOST>(a, b, idx, adm.g_dd, m,c,d,k,j,i);
      }
    }

    // -----------------------------------------------------------------------------------
    // inverse metric
    //
    detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                           adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                           adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
    adm::SpatialInv(1.0/detg,
                adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
                &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

    // -----------------------------------------------------------------------------------
    // Christoffel symbols
    //
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      Gamma_ddd(c,a,b) = 0.5*(dg_ddd(a,b,c) + dg_ddd(b,a,c) - dg_ddd(c,a,b));
    }

    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int d = 0; d < 3; ++d) {
      Gamma_udd(c,a,b) += g_uu(c,d)*Gamma_ddd(d,a,b);
    }

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
      Gamma_u(a) += g_uu(b,c)*Gamma_udd(a,b,c);
    }

    // -----------------------------------------------------------------------------------
    // Ricci tensor and Ricci scalar
    //
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        // Part with the Christoffel symbols
        for(int e = 0; e < 3; ++e) {
          R_dd(a,b) += g_uu(c,d) * Gamma_udd(e,a,c) * Gamma_ddd(e,b,d);
          R_dd(a,b) -= g_uu(c,d) * Gamma_udd(e,a,b) * Gamma_ddd(e,c,d);
        }
        // Wave operator part of the Ricci
        R_dd(a,b) += 0.5*g_uu(c,d)*(
            - ddg_dddd(c,d,a,b) - ddg_dddd(a,b,c,d) +
              ddg_dddd(a,c,b,d) + ddg_dddd(b,c,a,d));
      }
    }

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      R += g_uu(a,b) * R_dd(a,b);
    }

    // -----------------------------------------------------------------------------------
    // Extrinsic curvature: traces and derivatives
    //

    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        K_ud(a,b) = 0.0;
        for(int c = 0; c < 3; ++c) {
          K_ud(a,b) += g_uu(a,c) * adm.vK_dd(m,c,b,k,j,i);
        }
      }
      K += K_ud(a,a);
    }
    // K^a_b K^b_a
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      KK += K_ud(a,b) * K_ud(b,a);
    }
    // Covariant derivative of K
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
        DK_ddd(a,b,c) = dK_ddd(a,b,c);
      for(int d = 0; d < 3; ++d) {
          DK_ddd(a,b,c) -= Gamma_udd(d,a,b) * adm.vK_dd(m,d,c,k,j,i);
          DK_ddd(a,b,c) -= Gamma_udd(d,a,c) * adm.vK_dd(m,b,d,k,j,i);
      }
    }
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = 0; d < 3; ++d) {
      DK_udd(a,b,c) += g_uu(a,d) * DK_ddd(d,b,c);
    }

    //------------------------------------------------------------------------------------
    //     Construct tetrad
    //
    //     Initial tetrad guess. NB, aligned with z axis - possible problem if points
    //     lie on z axis theta and phi vectors degenerate
    //     Like BAM start with phi vector
    //     uvec = radial vec
    //     vvec = theta vec
    //     wvec = phi vec
    Real xx = x1v;
    if(SQR(x1v) +  SQR(x2v) < 1e-10)
      xx = xx + 1e-8;
    uvec(0) = xx;
    uvec(1) = x2v;
    uvec(2) = x3v;
    vvec(0) = xx*x3v;
    vvec(1) = x2v*x3v;
    vvec(2) = -SQR(xx)-SQR(x2v);
    wvec(0) = x2v*-1.0;
    wvec(1) = xx;
    wvec(2) = 0.0;

    //Gram-Schmidt orthonormalisation with spacetime metric.

    // (1) normalize phi vec
    for(int a = 0; a<3; ++a) {
      for(int b = 0; b<3; ++b) {
          dotp1 += adm.g_dd(m,a,b,k,j,i)*wvec(a)*wvec(b);
      }
    }
    for(int a =0; a<3; ++a) {
        wvec(a) = wvec(a)/std::sqrt(dotp1);
    }

    // (2) make radial vec orthogonal to phi vec
    dotp1 = 0;
    for(int a = 0; a<3; ++a) {
      for( int b = 0; b<3; ++b) {
        dotp1 += adm.g_dd(m,a,b,k,j,i)*wvec(a)*uvec(b);
      }
    }
    for(int a = 0; a<3; ++a) {
      uvec(a) -= dotp1*wvec(a);
    }

    // (3) normalize radial vec
    dotp1 = 0;
    for(int a = 0; a<3; ++a) {
      for(int b = 0; b<3; ++b) {
          dotp1 += adm.g_dd(m,a,b,k,j,i)*uvec(a)*uvec(b);
      }
    }

    for(int a =0; a<3; ++a) {
        uvec(a) = uvec(a)/std::sqrt(dotp1);
    }

    // (4) make theta vec orthogonal to both radial and phi vec
    dotp1 = 0;
    for(int a = 0; a<3; ++a) {
      for(int b = 0; b<3; ++b) {
        dotp1 += adm.g_dd(m,a,b,k,j,i)*wvec(a)*vvec(b);
      }
    }

    dotp2 = 0;
    for(int a = 0; a<3; ++a) {
      for( int b = 0; b<3; ++b) {
        dotp2 += adm.g_dd(m,a,b,k,j,i)*uvec(a)*vvec(b);
      }
    }

    for(int a = 0; a<3; ++a) {
      vvec(a) -= dotp1*wvec(a)+dotp2*uvec(a);
    }

    // (3) normalize theta vec
    dotp1 = 0;
    for(int a = 0; a<3; ++a) {
      for( int b = 0; b<3; ++b) {
        dotp1 += adm.g_dd(m,a,b,k,j,i)*vvec(a)*vvec(b);
      }
    }

    for(int a =0; a<3; ++a) {
      vvec(a) = vvec(a)/std::sqrt(dotp1);
    }

    //   Riem3_dddd = Riemann tensor of spacelike hypersurface
    //   Riemm4_dddd = Riemann tensor of 4D spacetime
    //   Riemm4_ddd  = Riemann tensor of 4D spacetime contracted once with n
    //   Riemm4_dd  = Riemann tensor of 4D spacetime contracted twice with n

    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = 0; d < 3; ++d) {
      Riem3_dddd(a,b,c,d) = adm.g_dd(m,a,c,k,j,i)*R_dd(b,d)
                            + adm.g_dd(m,b,d,k,j,i)*R_dd(a,c)
                            - adm.g_dd(m,a,d,k,j,i)*R_dd(b,c)
                            - adm.g_dd(m,b,c,k,j,i)*R_dd(a,d)
                            - 0.5*R*adm.g_dd(m,a,c,k,j,i)*adm.g_dd(m,b,d,k,j,i)
                            + 0.5*R*adm.g_dd(m,a,d,k,j,i)*adm.g_dd(m,b,c,k,j,i);
      Riemm4_dddd(a,b,c,d) = Riem3_dddd(a,b,c,d)
                            + adm.vK_dd(m,a,c,k,j,i)*adm.vK_dd(m,b,d,k,j,i)
                            - adm.vK_dd(m,a,d,k,j,i)*adm.vK_dd(m,b,c,k,j,i);
    }

    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        for(int c = 0; c < 3; ++c) {
          Riemm4_ddd(a,b,c) = - (DK_ddd(c,a,b) - DK_ddd(b,a,c));
        }
      }
    }


    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        Riemm4_dd(a,b) = R_dd(a,b) + K*adm.vK_dd(m,a,b,k,j,i);
        for(int c = 0; c < 3; ++c) {
          for(int d = 0; d < 3; ++d) {
            Riemm4_dd(a,b) += - g_uu(c,d)*adm.vK_dd(m,a,c,k,j,i)
                                        *adm.vK_dd(m,d,b,k,j,i);
          }
        }
      }
    }

    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        weyl.rpsi4(m,k,j,i) += - FR4 * Riemm4_dd(a,b) * (
          vvec(a) * vvec(b) - (-wvec(a) * (-wvec(b)))
        );
        weyl.ipsi4(m,k,j,i) += - FR4 * Riemm4_dd(a,b) * (
          -vvec(a) * wvec(b) - wvec(a)*vvec(b)
        );
        for(int c = 0; c < 3; ++c) {
          weyl.rpsi4(m,k,j,i) += 0.5 * Riemm4_ddd(a,c,b) * uvec(c) * (
            vvec(a) * vvec(b) - (-wvec(a)*(-wvec(b)))
          );
          weyl.ipsi4(m,k,j,i) += 0.5 * Riemm4_ddd(a,c,b) * uvec(c) * (
            -vvec(a) * wvec(b) - wvec(a)*vvec(b)
          );
          for(int d = 0; d < 3; ++d) {
            weyl.rpsi4(m,k,j,i) += -FR4 * (Riemm4_dddd(d,a,c,b) * uvec(d) * uvec(c)) * (
              vvec(a) * vvec(b) - (-wvec(a)*(-wvec(b)))
            );
            weyl.ipsi4(m,k,j,i) += -FR4 * (Riemm4_dddd(d,a,c,b) * uvec(d) * uvec(c)) * (
              -vvec(a) * wvec(b) - wvec(a)*vvec(b)
            );
          }
        }
      }
    }
    Real r = std::sqrt(SQR(x1v) +  SQR(x2v) + SQR(x3v));
    weyl.rpsi4(m,k,j,i) *= r;
    weyl.ipsi4(m,k,j,i) *= r;
  });
}

template void Z4c::Z4cWeyl<2>(MeshBlockPack *pmbp);
template void Z4c::Z4cWeyl<3>(MeshBlockPack *pmbp);
template void Z4c::Z4cWeyl<4>(MeshBlockPack *pmbp);
} // namespace z4c
