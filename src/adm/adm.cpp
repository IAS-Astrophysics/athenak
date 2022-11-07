//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.cpp
//  \brief implementation of ADM class

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "parameter_input.hpp"
#include "adm/adm.hpp"
#include "eos/eos.hpp"
#include "mesh/mesh.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/z4c.hpp"

char const * const ADM::ADM_names[ADM::N_ADM] = {
  "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
  "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
  "adm_psi4",
  "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
};

//----------------------------------------------------------------------------------------
// constructor: initializes data structures and parameters
ADM::ADM(MeshBlockPack *ppack, ParameterInput *pin):
  pmy_pack(ppack),
  u_adm("u_adm",1,1,1,1,1)
{
  int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  if (nullptr == pmy_pack->pz4c) {
    Kokkos::realloc(u_adm, nmb, N_ADM, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(u_adm, I_ADM_alpha);
    adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_betax, I_ADM_betaz);
  }
  else {
    // Lapse and shift are stored in the Z4c class
    z4c::Z4c * pz4c = pmy_pack->pz4c;
    Kokkos::realloc(u_adm, nmb, N_ADM - 4, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4c_alpha);
    adm.beta_u.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4c_betax, pz4c->I_Z4c_betaz);
  }
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_psi4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_gxx, I_ADM_gzz);
  adm.K_dd.InitWithShallowSlice(u_adm, I_ADM_Kxx, I_ADM_Kzz);
}

//----------------------------------------------------------------------------------------
//! \fn
// Coordinate (geometric) source term function for GR hydrodynamics
template<int NGHOST>
void ADM::AddCoordTerms(const DvceArray5D<Real> &prim, const EOS_Data &eos,
                        const Real dt, DvceArray5D<Real> &rhs)
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  int ncells1 = indcs.nx1+indcs.ng; // Align scratch buffers with variables
  int nmb = pmy_pack->nmb_thispack;

  auto &adm = pmy_pack->padm->adm;

  // DR: EOS call here
  Real const gamma_prime = eos.gamma / (eos.gamma - 1.0);

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1)*2       // scalars
                  + ScrArray2D<Real>::shmem_size(3, ncells1)*2    // vectors
                  + ScrArray2D<Real>::shmem_size(6, ncells1)*2    // symmetric 2 tensors
                  + ScrArray2D<Real>::shmem_size(9, ncells1)*1    // general 2 tensors
                  + ScrArray2D<Real>::shmem_size(18, ncells1)*1;  // symmetric 3 tensors
  par_for_outer("adm_coord_terms_loop",DevExeSpace(),scr_size,scr_level,0,nmb-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    //
    // Scratch space
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> vol;         // sqrt of determinant of spatial metric
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 0> E;           // fluid energy density

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;    // lapse 1st drvts
    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 1> S_d;         // matter momentum

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;        // inverse metric
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> S_uu;        // spatial component of stress tensor

    AthenaScratchTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;    // derivatives of the shift

    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;      // metric 1st drvts

         vol.NewAthenaScratchTensor(member, scr_level, ncells1);
           E.NewAthenaScratchTensor(member, scr_level, ncells1);
    dalpha_d.NewAthenaScratchTensor(member, scr_level, ncells1);
         S_d.NewAthenaScratchTensor(member, scr_level, ncells1);
    dbeta_du.NewAthenaScratchTensor(member, scr_level, ncells1);
        g_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
        S_uu.NewAthenaScratchTensor(member, scr_level, ncells1);
      dg_ddd.NewAthenaScratchTensor(member, scr_level, ncells1);

    //
    // Metric quantities
    par_for_inner(member, is, ie, [&](int const i) {
      Real detg = SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                             adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
      vol(i) = sqrt(detg);
    });

    par_for_inner(member, is, ie, [&](int const i) {
      SpatialInv(1.0/SQR(vol(i)),
                 adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i), adm.g_dd(m,0,2,k,j,i),
                 adm.g_dd(m,1,1,k,j,i), adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i),
                 &g_uu(0,0,i), &g_uu(0,1,i), &g_uu(0,2,i),
                 &g_uu(1,1,i), &g_uu(1,2,i), &g_uu(2,2,i));
    });

    //
    // Metric derivatives
    Real idx[] = {size.d_view(m).idx1, size.d_view(m).idx2, size.d_view(m).idx3};
    for (int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](int const i) {
        dalpha_d(a,i) = Dx<NGHOST>(a, idx, adm.alpha, m,k,j,i);
      });
    }
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](int const i) {
        dbeta_du(b,a,i) = Dx<NGHOST>(b, idx, adm.beta_u, m,a,k,j,i);
      });
    }
    for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b)
    for (int c = 0; c < 3; ++c) {
      par_for_inner(member, is, ie, [&](int const i) {
        dg_ddd(c,a,b,i) = Dx<NGHOST>(c, idx, adm.g_dd, m,a,b,k,j,i);
      });
    }

    //
    // Fluid quantities
    par_for_inner(member, is, ie, [&](int const i) {
      Real const & rho  = prim(m,IDN,k,j,i);
      Real const z_u[3] = {
        prim(m,IVX,k,j,i), prim(m,IVY,k,j,i), prim(m,IVZ,k,j,i)
      };

      // DR: EOS calls here
      Real const press = eos.IdealGasPressure(prim(m,IEN,k,j,i));
      Real const rho_h = rho + gamma_prime*press;

      Real z2 = 0;
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
          z2 += adm.g_dd(m,a,b,k,j,i)*z_u[a]*z_u[b];
      }
      Real const W2 = 1.0 + z2;
      Real const W = sqrt(W2);
      Real const v_u[3] = {z_u[0]/W, z_u[1]/W, z_u[2]/W};

      E(i) = rho_h*W2 - press;

      for (int a = 0; a < 3; ++a) {
        S_d(a,i) = 0;
        for (int b = 0; b < 3; ++b) {
          S_d(a,i) += rho_h*W2*adm.g_dd(m,a,b,k,j,i)*v_u[b];
        }
      }

      for (int a = 0; a < 3; ++a)
      for (int b = a; b < 3; ++b) {
        S_uu(a,b,i) = rho_h*W2*v_u[a]*v_u[b] + press*g_uu(a,b,i);
      }
    });

    //
    // Assemble energy RHS
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      par_for_inner(member, is, ie, [&](int const i) {
        rhs(m,IEN,k,j,i) += dt * vol(i) * (
           adm.alpha(m,k,j,i) * adm.K_dd(m,a,b,k,j,i) * S_uu(a,b,i) -
           g_uu(a,b,i) * S_d(a,i) * dalpha_d(b,i) );
      });
    }

    //
    // Assemble momentum RHS
    for (int a = 0; a < 3; ++a) {
      par_for_inner(member, is, ie, [&](int const i) {
        for (int b = 0; b < 3; ++b) {
          for (int c = 0; c < 3; ++c) {
            rhs(m,IM1+a,k,j,i) += 0.5 * dt * adm.alpha(m,k,j,i) * vol(i) *
              S_uu(b,c,i) * dg_ddd(a,b,c,i);
          }
          rhs(m,IM1+a,k,j,i) += dt * vol(i) * S_d(b,i) * dbeta_du(a,b,i);
        }
        rhs(m,IM1+a,k,j,i) -= dt * vol(i) * E(i) * dalpha_d(a,i);
      });
    }
  });
}

template
void ADM::AddCoordTerms<2>(const DvceArray5D<Real> &prim, const EOS_Data &eos,
                           const Real dt, DvceArray5D<Real> &rhs);
template
void ADM::AddCoordTerms<3>(const DvceArray5D<Real> &prim, const EOS_Data &eos,
                           const Real dt, DvceArray5D<Real> &rhs);
template
void ADM::AddCoordTerms<4>(const DvceArray5D<Real> &prim, const EOS_Data &eos,
                           const Real dt, DvceArray5D<Real> &rhs);

//----------------------------------------------------------------------------------------
// destructor
ADM::~ADM() {}
