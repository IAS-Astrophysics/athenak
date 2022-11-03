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
//! \fn TaskStatus Z4c::CalcRHS
//! \brief Computes the wave equation RHS
template <int NGHOST>
TaskStatus Z4c::CalcRHS(Driver *pdriver, int stage)
{
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
  auto &rhs = pmy_pack->pz4c->rhs;
  auto &opt = pmy_pack->pz4c->opt;
  int scr_level = 1;
  // 2 1D scratch array and 1 2D scratch array
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*17   // 0 tensors
                  + ScrArray2D<Real>::shmem_size(3,ncells1)*13 // vectors
                  + ScrArray2D<Real>::shmem_size(6,ncells1)*13 // 2D tensor with symm
                  + ScrArray2D<Real>::shmem_size(9,ncells1)*2  // 2D tensor with no symm
                  + ScrArray2D<Real>::shmem_size(18,ncells1)*8 // 3D tensor with symm
                  + ScrArray2D<Real>::shmem_size(36,ncells1);  // 4D tensor with symm
  par_for_outer("z4c rhs loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    // Define scratch arrays to be used in the following calculations
    // These are spatially 1-D arrays with different ranks for symmetries
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> r;            // radial coordinate
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> detg;         // det(g)
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> chi_guarded;  // bounded version of chi
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> oopsi4;       // 1/psi4
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> A;            // trace of A
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> AA;           // trace of AA
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> R;            // Ricci scalar
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> Ht;           // tilde H
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> K;            // trace of extrinsic curvature
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> KK;           // K^a_b K^b_a
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> S;            // Trace of S_ik
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> Ddalpha;      // Trace of Ddalpha_dd

              r.NewAthenaScratchTensor(member, scr_level, ncells1);
           detg.NewAthenaScratchTensor(member, scr_level, ncells1);
    chi_guarded.NewAthenaScratchTensor(member, scr_level, ncells1);
         oopsi4.NewAthenaScratchTensor(member, scr_level, ncells1);
              A.NewAthenaScratchTensor(member, scr_level, ncells1);
             AA.NewAthenaScratchTensor(member, scr_level, ncells1);
              R.NewAthenaScratchTensor(member, scr_level, ncells1);
             Ht.NewAthenaScratchTensor(member, scr_level, ncells1);
              K.NewAthenaScratchTensor(member, scr_level, ncells1);
             KK.NewAthenaScratchTensor(member, scr_level, ncells1);
              S.NewAthenaScratchTensor(member, scr_level, ncells1);
        Ddalpha.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> M_u;         // momentum constraint
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;     // Gamma computed from the metric
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> DA_u;        // Covariant derivative of A
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> s_u;         // x^i/r where r is the coord. radius

        M_u.NewAthenaScratchTensor(member, scr_level, ncells1);
    Gamma_u.NewAthenaScratchTensor(member, scr_level, ncells1);
       DA_u.NewAthenaScratchTensor(member, scr_level, ncells1);
        s_u.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;        // inverse of conf. metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> A_uu;        // inverse of A
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> AA_dd;       // g^cd A_ac A_db
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;        // Ricci tensor
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Rphi_dd;     // Ricci tensor, conformal contribution
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Kt_dd;       // conformal extrinsic curvature
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> K_ud;        // extrinsic curvature
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Ddalpha_dd;  // 2nd differential of the lapse
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Ddphi_dd;    // 2nd differential of phi


          g_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
          A_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
         AA_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
          R_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
       Rphi_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
         Kt_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
          K_ud.NewAthenaScratchTensor(member, scr_level, ncells1);
    Ddalpha_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
      Ddphi_dd.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;   // Christoffel symbols of 1st kind
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;   // Christoffel symbols of 2nd kind
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> DK_ddd;      // differential of K
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> DK_udd;      // differential of K

    Gamma_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
    Gamma_udd.NewAthenaScratchTensor(member, scr_level, ncells1);
       DK_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
       DK_udd.NewAthenaScratchTensor(member, scr_level, ncells1);

    // auxiliary derivatives
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> dbeta;       // d_a beta^a

    dbeta.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;    // lapse 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> ddbeta_d;    // 2nd "divergence" of beta
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;      // chi 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dphi_d;      // phi 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dK_d;        // K 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;     // Khat 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;    // Theta 1st drvts

    dalpha_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    ddbeta_d.NewAthenaScratchTensor(member, scr_level, ncells1);
      dchi_d.NewAthenaScratchTensor(member, scr_level, ncells1);
      dphi_d.NewAthenaScratchTensor(member, scr_level, ncells1);
        dK_d.NewAthenaScratchTensor(member, scr_level, ncells1);
     dKhat_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    dTheta_d.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> ddalpha_dd;  // lapse 2nd drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;    // shift 1st drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> ddchi_dd;    // chi 2nd drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;     // Gamma 1st drvts

    ddalpha_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
      dbeta_du.NewAthenaScratchTensor(member, scr_level, ncells1);
      ddchi_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
       dGam_du.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2,  3, 3> dg_ddd;      // metric 1st drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2,  3, 3> dK_ddd;      // K 1st drvts
    AthenaScratchTensor<Real, TensorSymm::SYM2,  3, 3> dA_ddd;      // A 1st drvts
    AthenaScratchTensor<Real, TensorSymm::ISYM2, 3, 3> ddbeta_ddu; // shift 2nd drvts

        dg_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
        dK_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
        dA_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);
    ddbeta_ddu.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;   // metric 2nd drvts

    ddg_dddd.NewAthenaScratchTensor(member, scr_level, ncells1);

    // auxiliary Lie derivatives along the shift vector
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> Lchi;        // Lie derivative of chi
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> LKhat;       // Lie derivative of Khat
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> LTheta;      // Lie derivative of Theta
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> Lalpha;      // Lie derivative of the lapse

      Lchi.NewAthenaScratchTensor(member, scr_level, ncells1);
     LKhat.NewAthenaScratchTensor(member, scr_level, ncells1);
    LTheta.NewAthenaScratchTensor(member, scr_level, ncells1);
    Lalpha.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> LGam_u;      // Lie derivative of Gamma
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> Lbeta_u;     // Lie derivative of the shift

     LGam_u.NewAthenaScratchTensor(member, scr_level, ncells1);
    Lbeta_u.NewAthenaScratchTensor(member, scr_level, ncells1);

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> Lg_dd;       // Lie derivative of conf. 3-metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> LA_dd;       // Lie derivative of A

    Lg_dd.NewAthenaScratchTensor(member, scr_level, ncells1);
    LA_dd.NewAthenaScratchTensor(member, scr_level, ncells1);

    Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
    int der_ghost = indcs.ng;
    // -----------------------------------------------------------------------------------
    // 1st derivatives
    //
    // Scalars
    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {
        dalpha_d(a,i) = Dx<NGHOST>(a, idx, z4c.alpha, m,k,j,i);
        dchi_d  (a,i) = Dx<NGHOST>(a, idx, z4c.chi,   m,k,j,i);
        dKhat_d (a,i) = Dx<NGHOST>(a, idx, z4c.Khat,  m,k,j,i);
        dTheta_d(a,i) = Dx<NGHOST>(a, idx, z4c.Theta, m,k,j,i);
      });
    }
    // Vectors
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        dbeta_du(b,a,i) = Dx<NGHOST>(b, idx, z4c.beta_u, m,a,k,j,i);
         dGam_du(b,a,i) = Dx<NGHOST>(b, idx, z4c.Gam_u,  m,a,k,j,i);
      });
    }
    // Tensors
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        dg_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, z4c.g_dd, m,a,b,k,j,i);
        dA_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, z4c.A_dd, m,a,b,k,j,i);
      });
    }

    // -----------------------------------------------------------------------------------
    // 2nd derivatives
    //
    // Scalars
    ddalpha_dd.ZeroClear();
    ddchi_dd.ZeroClear();
    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {
        ddalpha_dd(a,a,i) = Dxx<NGHOST>(a, idx, z4c.alpha, m,k,j,i);
          ddchi_dd(a,a,i) = Dxx<NGHOST>(a, idx, z4c.chi,   m,k,j,i);
      });
      for(int b = a + 1; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          ddalpha_dd(a,b,i) = Dxy<NGHOST>(a, b, idx, z4c.alpha, m,k,j,i);
            ddchi_dd(a,b,i) = Dxy<NGHOST>(a, b, idx, z4c.chi,   m,k,j,i);
        });
      }
    }
    // Vectors
    ddbeta_ddu.ZeroClear();
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      if(a == b) {
        par_for_inner(member, is, ie, [&](const int i) {
          ddbeta_ddu(a,b,c,i) = Dxx<NGHOST>(a, idx, z4c.beta_u, m,c,k,j,i);
        });
      }
      else {
        par_for_inner(member, is, ie, [&](const int i) {
          ddbeta_ddu(a,b,c,i) = Dxy<NGHOST>(a, b, idx, z4c.beta_u, m,c,k,j,i);
        });
      }
    }
    // Tensors
    ddg_dddd.ZeroClear();
    for(int c = 0; c < 3; ++c)
    for(int d = c; d < 3; ++d)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      if(a == b) {
        par_for_inner(member, is, ie, [&](const int i) {
          ddg_dddd(a,b,c,d,i) = Dxx<NGHOST>(a, idx, z4c.g_dd, m,c,d,k,j,i);
        });
      }
      else {
        par_for_inner(member, is, ie, [&](const int i) {
          ddg_dddd(a,b,c,d,i) = Dxy<NGHOST>(a, b, idx, z4c.g_dd, m,c,d,k,j,i);
        });
      }
    }
    // -----------------------------------------------------------------------------------
    // Advective derivatives
    //
    // Scalars
    Lalpha.ZeroClear();
    Lchi.ZeroClear();
    LKhat.ZeroClear();
    LTheta.ZeroClear();
    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {  
        Lalpha(i) += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.alpha, m,a,k,j,i);
        Lchi(i)   += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.chi,   m,a,k,j,i);
        LKhat(i)  += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.Khat,  m,a,k,j,i);
        LTheta(i) += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.Theta, m,a,k,j,i);
      });
    }
    // Vectors
    Lbeta_u.ZeroClear();
    LGam_u.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Lbeta_u(b,i) += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.beta_u, m,a,b,k,j,i);
        LGam_u(b,i)  += Lx<NGHOST>(a, idx, z4c.beta_u, z4c.Gam_u,  m,a,b,k,j,i);
      });
    }
    // Tensors
    Lg_dd.ZeroClear();
    LA_dd.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        Lg_dd(a,b,i) += Lx<NGHOST>(c, idx, z4c.beta_u, z4c.g_dd, m,c,a,b,k,j,i);
        LA_dd(a,b,i) += Lx<NGHOST>(c, idx, z4c.beta_u, z4c.A_dd, m,c,a,b,k,j,i);
      });
    }
    // -----------------------------------------------------------------------------------
    // Get K from Khat
    //
    par_for_inner(member, is, ie, [&](const int i) {
      K(i) = z4c.Khat(m,k,j,i) + 2.*z4c.Theta(m,k,j,i);
    });


    // -----------------------------------------------------------------------------------
    // Inverse metric
    //
    par_for_inner(member, is, ie, [&](const int i) {
      detg(i) = SpatialDet(z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                           z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i));
      SpatialInv(1.0/detg(i),
                 z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i), z4c.g_dd(m,0,2,k,j,i),
                 z4c.g_dd(m,1,1,k,j,i), z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i),
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
    for(int c = 0; c < 3; ++c)
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int d = 0; d < 3; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_udd(c,a,b,i) += g_uu(c,d,i)*Gamma_ddd(d,a,b,i);
      });
    }
    // Gamma's computed from the conformal metric (not evolved)
    Gamma_u.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b)
    for(int c = 0; c < 3; ++c) {
      par_for_inner(member, is, ie, [&](const int i) {
        Gamma_u(a,i) += g_uu(b,c,i)*Gamma_udd(a,b,c,i);
      });
    }

    // -----------------------------------------------------------------------------------
    // Curvature of conformal metric
    //
    R_dd.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      for(int c = 0; c < 3; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          R_dd(a,b,i) += 0.5*(z4c.g_dd(m,c,a,k,j,i)*dGam_du(b,c,i) +
                              z4c.g_dd(m,c,b,k,j,i)*dGam_du(a,c,i) +
                              Gamma_u(c,i)*(Gamma_ddd(a,b,c,i) + Gamma_ddd(b,a,c,i)));
        });
      }
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        par_for_inner(member, is, ie, [&](const int i) {
          R_dd(a,b,i) -= 0.5*g_uu(c,d,i)*ddg_dddd(c,d,a,b,i);
        });
      }
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d)
      for(int e = 0; e < 3; ++e) {
        par_for_inner(member, is, ie, [&](const int i) {
          R_dd(a,b,i) += g_uu(c,d,i)*(
              Gamma_udd(e,c,a,i)*Gamma_ddd(b,e,d,i) +
              Gamma_udd(e,c,b,i)*Gamma_ddd(a,e,d,i) +
              Gamma_udd(e,a,d,i)*Gamma_ddd(e,c,b,i));
        });
      }
    }

    // -----------------------------------------------------------------------------------
    // Derivatives of conformal factor phi
    //
    par_for_inner(member, is, ie, [&](const int i) {
      chi_guarded(i) = (z4c.chi(m,k,j,i)>opt.chi_div_floor) ? z4c.chi(m,k,j,i) : opt.chi_div_floor;
      oopsi4(i) = std::pow(chi_guarded(i), -4./opt.chi_psi_power);
    });

    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {
        dphi_d(a,i) = dchi_d(a,i)/(chi_guarded(i) * opt.chi_psi_power);
      });
    }
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Real const ddphi_ab = ddchi_dd(a,b,i)/(chi_guarded(i) * opt.chi_psi_power) -
          opt.chi_psi_power * dphi_d(a,i) * dphi_d(b,i);
        Ddphi_dd(a,b,i) = ddphi_ab;
      });
      for(int c = 0; c < 3; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          Ddphi_dd(a,b,i) -= Gamma_udd(c,a,b,i)*dphi_d(c,i);
        });
      }
    }

    // -----------------------------------------------------------------------------------
    // Curvature contribution from conformal factor
    //
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Rphi_dd(a,b,i) = 4.*dphi_d(a,i)*dphi_d(b,i) - 2.*Ddphi_dd(a,b,i);
      });
      for(int c = 0; c < 3; ++c)
      for(int d = 0; d < 3; ++d) {
        par_for_inner(member, is, ie, [&](const int i) {
          Rphi_dd(a,b,i) -= 2.*z4c.g_dd(m,a,b,k,j,i) * g_uu(c,d,i)*(Ddphi_dd(c,d,i) +
              2.*dphi_d(c,i)*dphi_d(d,i));
        });
      }
    }

    // -----------------------------------------------------------------------------------
    // Trace of the matter stress tensor
    //
    // Matter commented out
    //S.ZeroClear();
    //for(int a = 0; a < 3; ++a)
    //for(int b = 0; b < 3; ++b) {
    //  ILOOP1(i) {
    //    S(i) += oopsi4(i) * g_uu(a,b,i) * mat.S_dd(m,a,b,k,j,i);
    //  }
    //}

    // -----------------------------------------------------------------------------------
    // 2nd covariant derivative of the lapse
    //
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Ddalpha_dd(a,b,i) = ddalpha_dd(a,b,i)
                         - 2.*(dphi_d(a,i)*dalpha_d(b,i) + dphi_d(b,i)*dalpha_d(a,i));
      });
      for(int c = 0; c < 3; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          Ddalpha_dd(a,b,i) -= Gamma_udd(c,a,b,i)*dalpha_d(c,i);
        });
        for(int d = 0; d < 3; ++d) {
          par_for_inner(member, is, ie, [&](const int i) {
            Ddalpha_dd(a,b,i) += 2.*z4c.g_dd(m,a,b,k,j,i) * g_uu(c,d,i) * dphi_d(c,i) * dalpha_d(d,i);
          });
        }
      }
    }

    Ddalpha.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Ddalpha(i) += oopsi4(i) * g_uu(a,b,i) * Ddalpha_dd(a,b,i);
      });
    }

    // -----------------------------------------------------------------------------------
    // Contractions of A_ab, inverse, and derivatives
    //
    AA_dd.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = 0; d < 3; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        AA_dd(a,b,i) += g_uu(c,d,i) * z4c.A_dd(m,a,c,k,j,i) * z4c.A_dd(m,d,b,k,j,i);
      });
    }
    AA.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        AA(i) += g_uu(a,b,i) * AA_dd(a,b,i);
      });
    }
    A_uu.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b)
    for(int c = 0; c < 3; ++c)
    for(int d = 0; d < 3; ++d) {
      par_for_inner(member, is, ie, [&](const int i) {
        A_uu(a,b,i) += g_uu(a,c,i) * g_uu(b,d,i) * z4c.A_dd(m,c,d,k,j,i);
      });
    }
    DA_u.ZeroClear();
    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          DA_u(a,i) -= (3./2.) * A_uu(a,b,i) * dchi_d(b,i) / chi_guarded(i);
          DA_u(a,i) -= (1./3.) * g_uu(a,b,i) * (2.*dKhat_d(b,i) + dTheta_d(b,i));
        });
      }
      for(int b = 0; b < 3; ++b)
      for(int c = 0; c < 3; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          DA_u(a,i) += Gamma_udd(a,b,c,i) * A_uu(b,c,i);
        });
      }
    }

    // -----------------------------------------------------------------------------------
    // Ricci scalar
    //
    R.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        R(i) += oopsi4(i) * g_uu(a,b,i) * (R_dd(a,b,i) + Rphi_dd(a,b,i));
      });
    }

    // -----------------------------------------------------------------------------------
    // Hamiltonian constraint
    //
    par_for_inner(member, is, ie, [&](const int i) {
      Ht(i) = R(i) + (2./3.)*SQR(K(i)) - AA(i);
    });
    // -----------------------------------------------------------------------------------
    // Finalize advective (Lie) derivatives
    //
    // Shift vector contractions
    dbeta.ZeroClear();
    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {
        dbeta(i) += dbeta_du(a,a,i);
      });
    }
    ddbeta_d.ZeroClear();
    for(int a = 0; a < 3; ++a)
    for(int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        ddbeta_d(a,i) += (1./3.) * ddbeta_ddu(a,b,b,i);
      });
    }
    // Finalize Lchi
    par_for_inner(member, is, ie, [&](const int i) {
      Lchi(i) += (1./6.) * opt.chi_psi_power * chi_guarded(i) * dbeta(i);
    });
    // Finalize LGam_u (note that this is not a real Lie derivative)
    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {
        LGam_u(a,i) += (2./3.) * Gamma_u(a,i) * dbeta(i);
      });
      for(int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          LGam_u(a,i) += g_uu(a,b,i) * ddbeta_d(b,i) - Gamma_u(b,i) * dbeta_du(b,a,i);
        });
        for(int c = 0; c < 3; ++c) {
          par_for_inner(member, is, ie, [&](const int i) {
            LGam_u(a,i) += g_uu(b,c,i) * ddbeta_ddu(b,c,a,i);
          });
        }
      }
    }
    // Finalize Lg_dd and LA_dd
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        Lg_dd(a,b,i) -= (2./3.) * z4c.g_dd(m,a,b,k,j,i) * dbeta(i);
        LA_dd(a,b,i) -= (2./3.) * z4c.A_dd(m,a,b,k,j,i) * dbeta(i);
      });
      for(int c = 0; c < 3; ++c) {
        par_for_inner(member, is, ie, [&](const int i) {
          Lg_dd(a,b,i) += dbeta_du(a,c,i) * z4c.g_dd(m,b,c,k,j,i);
          LA_dd(a,b,i) += dbeta_du(b,c,i) * z4c.A_dd(m,a,c,k,j,i);
          Lg_dd(a,b,i) += dbeta_du(b,c,i) * z4c.g_dd(m,a,c,k,j,i);
          LA_dd(a,b,i) += dbeta_du(a,c,i) * z4c.A_dd(m,b,c,k,j,i);
        });
      }
    }
    // -----------------------------------------------------------------------------------
    // Assemble RHS
    //
    // Khat, chi, and Theta
    par_for_inner(member, is, ie, [&](const int i) {
      rhs.Khat(m,k,j,i) = - Ddalpha(i) + z4c.alpha(m,k,j,i) * (AA(i) + (1./3.)*SQR(K(i))) +
        LKhat(i) + opt.damp_kappa1*(1 - opt.damp_kappa2) * z4c.alpha(m,k,j,i) * z4c.Theta(m,k,j,i);
    // Matter commented out
      //rhs.Khat(m,k,j,i) += 4*M_PI * z4c.alpha(m,k,j,i) * (S(i) + mat.rho(m,k,j,i));
      rhs.chi(m,k,j,i) = Lchi(i) - (1./6.) * opt.chi_psi_power *
        chi_guarded(i) * z4c.alpha(m,k,j,i) * K(i);
      rhs.Theta(m,k,j,i) = LTheta(i) + z4c.alpha(m,k,j,i) * (
          0.5*Ht(i) - (2. + opt.damp_kappa2) * opt.damp_kappa1 * z4c.Theta(m,k,j,i));
    // Matter commented out
      //rhs.Theta(m,k,j,i) -= 8.*M_PI * z4c.alpha(m,k,j,i) * mat.rho(m,k,j,i);
    });
    // Gamma's
    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.Gam_u(m,a,k,j,i) = 2.*z4c.alpha(m,k,j,i)*DA_u(a,i) + LGam_u(a,i);
        rhs.Gam_u(m,a,k,j,i) -= 2.*z4c.alpha(m,k,j,i) * opt.damp_kappa1 *
            (z4c.Gam_u(m,a,k,j,i) - Gamma_u(a,i));
      });
      for(int b = 0; b < 3; ++b) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.Gam_u(m,a,k,j,i) -= 2. * A_uu(a,b,i) * dalpha_d(b,i);
    // Matter commented out
        //rhs.Gam_u(m,a,k,j,i) -= 16.*M_PI * z4c.alpha(m,k,j,i) 
	//                      * g_uu(a,b) * mat.S_d(m,b,k,j,i);
        });
      }
    }
    // g and A
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.g_dd(m,a,b,k,j,i) = - 2. * z4c.alpha(m,k,j,i) * z4c.A_dd(m,a,b,k,j,i) 
	                      + Lg_dd(a,b,i);
        rhs.A_dd(m,a,b,k,j,i) = oopsi4(i) *
            (-Ddalpha_dd(a,b,i) + z4c.alpha(m,k,j,i) * (R_dd(a,b,i) + Rphi_dd(a,b,i)));
        rhs.A_dd(m,a,b,k,j,i) -= (1./3.) * z4c.g_dd(m,a,b,k,j,i)
                               * (-Ddalpha(i) + z4c.alpha(m,k,j,i)*R(i));
        rhs.A_dd(m,a,b,k,j,i) += z4c.alpha(m,k,j,i) * (K(i)*z4c.A_dd(m,a,b,k,j,i)
                               - 2.*AA_dd(a,b,i));
        rhs.A_dd(m,a,b,k,j,i) += LA_dd(a,b,i);
    // Matter commented out
        //rhs.A_dd(m,a,b,k,j,i) -= 8.*M_PI * z4c.alpha(m,k,j,i) *
        // (oopsi4*mat.S_dd(m,a,b,k,j,i) - (1./3.)*S(i)*z4c.g_dd(m,a,b,k,j,i));
      });
    }
    // lapse function
    par_for_inner(member, is, ie, [&](const int i) {
      Real const f = opt.lapse_oplog * opt.lapse_harmonicf
                   + opt.lapse_harmonic * z4c.alpha(m,k,j,i);
      rhs.alpha(m,k,j,i) = opt.lapse_advect * Lalpha(i)
                         - f * z4c.alpha(m,k,j,i) * z4c.Khat(m,k,j,i);
    });

    // shift vector
    for(int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](const int i) {
        rhs.beta_u(m,a,k,j,i) = opt.shift_Gamma * z4c.Gam_u(m,a,k,j,i)
                              + opt.shift_advect * Lbeta_u(a,i);
        rhs.beta_u(m,a,k,j,i) -= opt.shift_eta * z4c.beta_u(m,a,k,j,i);
        // FORCE beta = 0
        //rhs.beta_u(m,a,k,j,i) = 0;
      });
    }

    // harmonic gauge terms
    if (std::fabs(opt.shift_alpha2Gamma) > 0.0) {
      for(int a = 0; a < 3; ++a) {
        par_for_inner(member, is, ie, [&](const int i) {
          rhs.beta_u(m,a,k,j,i) += opt.shift_alpha2Gamma *
                               SQR(z4c.alpha(m,k,j,i)) * z4c.Gam_u(m,a,k,j,i);
        });
      }
    }
    if (std::fabs(opt.shift_H) > 0.0) {
      for(int a = 0; a < 3; ++a) {
        for(int b = 0; b < 3; ++b) {
          par_for_inner(member, is, ie, [&](const int i) {
            rhs.beta_u(m,a,k,j,i) += opt.shift_H * z4c.alpha(m,k,j,i) *
            chi_guarded(i) * (0.5 * z4c.alpha(m,k,j,i) * dchi_d(b,i) -
                              dalpha_d(b,i)) * g_uu(a,b,i);
          });
        }
      }
    }
  });
  // ===================================================================================
  // Add dissipation for stability
  //
  Real &diss = pmy_pack->pz4c->diss;
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u_rhs = pmy_pack->pz4c->u_rhs;
  par_for_outer("K-O Dissipation",
  DevExeSpace(),scr_size,scr_level,0,nmb-1,0,N_Z4c-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member,
  const int m, const int n, const int k, const int j) {
  Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
  for(int a = 0; a < 3; ++a) {
    par_for_inner(member, is, ie, [&](const int i) {
      u_rhs(m,n,k,j,i) += Diss<NGHOST>(a, idx, u0, m, n, k, j, i)*diss;
    });
  }
  });
  return TaskStatus::complete;
}

template TaskStatus Z4c::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<4>(Driver *pdriver, int stage);
} // namespace z4c
