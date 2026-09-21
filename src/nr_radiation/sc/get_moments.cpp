//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file get_moments.cpp
//! \brief Radiation moment quadrature (Davis 2012 Eq. 17-19): J every sweep for the
//! iteration, H and K once per solve; and the coupling
//! Q = crat * prat * sigma_a * (J - brad) (Davis Eq. 27 absorption form).
//! Moments array ordering (Athena++ convention):
//!   moments: 0:J, 1-3:H_1,H_2,H_3, 4:K_11, 5:K_22, 6:K_33, 7:K_12, 8:K_13, 9:K_23
//! -- Athena++ rad_mom order (IER, IFR1-3, IPR...), with the symmetric six K components

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeJ
//! \brief Mean intensity only, J = Sum w I (Eq. 17), into moments slot 0. The cheap one
//! iteration runs after every sweep (residual) and MeshRefinement after a remesh.

void SC::ComputeJ() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto mom_ = moments;

  par_for("sc_compute_j", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real J = 0.0;
    for (int angg = 0; angg < nang_tot_; ++angg) {
      int a = angg % nang;
      J += wmu.d_view(a) * ir_(m, angg, k, j, i);
    }
    mom_(m, 0, k, j, i) = J;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeHK
//! \brief The first and second moments H_i, K_ij into moments slots 1-9: one par_for
//! over cells with an inner serial loop over all nang_tot angles, nine accumulators in
//! registers. Slots in Athena++ order: H1(0), H2(1), H3(2), K11(3), K22(4), K33(5),
//! K12(6), K13(7), K23(8). Run once per solve. Not gated on a consumer: slot 0 shares
//! the array and a restart rebuilds it from the restored ir before the outputs exist, so
//! the array cannot be reallocated later and there is nothing to defer. Consumers today
//! are the outputs; a radiation force or an Eddington-tensor closure would be next. J
//! itself is slot 0 (ComputeJ, per sweep).

void SC::ComputeHK() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nang = pang->nang;
  int nang_tot_ = nang_tot;
  auto &mu = pang->mu;
  auto &wmu = pang->wmu;
  auto ir_ = ir;
  auto mom_ = moments;

  par_for("sc_compute_hk", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real H1 = 0.0, H2 = 0.0, H3 = 0.0;
    Real K11 = 0.0, K22 = 0.0, K33 = 0.0;
    Real K12 = 0.0, K13 = 0.0, K23 = 0.0;

    for (int angg = 0; angg < nang_tot_; ++angg) {
      int oct = angg / nang;
      int a = angg - oct * nang;
      Real w = wmu.d_view(a);
      Real nx = mu.d_view(oct, a, 0);
      Real ny = mu.d_view(oct, a, 1);
      Real nz = mu.d_view(oct, a, 2);
      Real wI = w * ir_(m,angg,k,j,i);

      H1  += nx * wI;
      H2  += ny * wI;
      H3  += nz * wI;
      K11 += nx * nx * wI;
      K22 += ny * ny * wI;
      K33 += nz * nz * wI;
      K12 += nx * ny * wI;
      K13 += nx * nz * wI;
      K23 += ny * nz * wI;
    }

    mom_(m, 1, k, j, i) = H1;
    mom_(m, 2, k, j, i) = H2;
    mom_(m, 3, k, j, i) = H3;
    mom_(m, 4, k, j, i) = K11;
    mom_(m, 5, k, j, i) = K22;
    mom_(m, 6, k, j, i) = K33;
    mom_(m, 7, k, j, i) = K12;
    mom_(m, 8, k, j, i) = K13;
    mom_(m, 9, k, j, i) = K23;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SC::ComputeQrad
//! \brief Gas-radiation coupling, Davis Eq. 27 in its absorption form:
//!   Q = crat * prat * sigma_a * (J - brad)
//!
//! Eq. 27 gives the two as an identity, 4 pi chi_tot (J - S) = 4 pi sigma_a (J - B):
//! with S = eps B + (1 - eps) J and eps = sigma_a/chi_tot, the scattering contribution
//! cancels exactly. So scattering is in the transfer equation and in srad, and is
//! absent from Q -- not because the reference omits it, but because it cancels.
//!
//! The ABSORPTION coefficient, then, not chi. The two arrays are equal today, so this
//! reads as a free choice; it is not. Once sigma_s exists chi = sigma_a + sigma_s, and
//! chi here would overstate heating and cooling by the scattering fraction, with no
//! crash and nothing obviously wrong in the output. Of the two valid forms this one is
//! also the better conditioned: when scattering dominates, eps -> 0 and S -> J, so
//! chi_tot (J - S) becomes a small difference of large numbers, while sigma_a (J - B)
//! does not. newdt.cpp already draws the same distinction, using sigma_a for the
//! coupling and chi only in the diffusion denominator.
//!
//! Both forms lose precision in the opposite limit -- optically thick and near
//! equilibrium, where J -> B. That is what Davis Eq. 28, Q = -4 pi div H, is for,
//! switched on chi*dx > 1. Not implemented here: this module has only the integral
//! form, so the thick regime is a known gap against the reference.

void SC::ComputeQrad() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto sigma_a_ = sigma_a;
  auto brad_ = brad;
  auto mom_ = moments;
  auto qrad_ = qrad;
  const Real crat_prat = crat * prat;

  par_for("sc_compute_qrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    qrad_(m,k,j,i) = crat_prat * sigma_a_(m,0,k,j,i)
                     * (mom_(m,0,k,j,i) - brad_(m,k,j,i));
  });
}

}  // namespace nr_radiation
