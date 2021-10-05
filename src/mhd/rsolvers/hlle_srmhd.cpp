//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_srmhd.cpp
//! \brief Implements HLLE Riemann solver for special relativistic MHD.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void HLLE
//  \brief The HLLE Riemann solver for SR MHD

KOKKOS_INLINE_FUNCTION
void HLLE(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j,  const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez)
{
  int ivy = IVX + ((ivx-IVX) + 1)%3;
  int ivz = IVX + ((ivx-IVX) + 2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  Real igm1 = 1.0/(eos.gamma - 1.0);
  Real iso_cs = eos.iso_cs;

  // Calculate metric if in GR
  int i01(0), i11(0);

  // Transform primitives to locally flat coordinates if in GR
#if GENERAL_RELATIVITY
  {
    switch (ivx) {
      case IVX:
        pmb->pcoord->PrimToLocal1(k, j, il, iu, bb, prim_l, prim_r, bb_normal);
        break;
      case IVY:
        pmb->pcoord->PrimToLocal2(k, j, il, iu, bb, prim_l, prim_r, bb_normal);
        break;
      case IVZ:
        pmb->pcoord->PrimToLocal3(k, j, il, iu, bb, prim_l, prim_r, bb_normal);
        break;
    }
  }
#else  // SR; need to populate 1D normal B array
  {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      bb_normal(i) = bb(k,j,i);
    }
  }
#endif  // GENERAL_RELATIVITY


  // Extract ratio of specific heats
  const Real gamma_adi = pmb->peos->GetGamma();

  // Go through each interface
#pragma omp simd simdlen(SIMD_WIDTH)
  for (int i=il; i<=iu; ++i) {
    // Extract left primitives
    Real rho_l = prim_l(IDN,i);
    Real pgas_l = prim_l(IPR,i);
    Real ux_l = prim_l(ivx,i);
    Real uy_l = prim_l(ivy,i);
    Real uz_l = prim_l(ivz,i);
    Real u_l[4];
    u_l[0] = std::sqrt(1.0 + SQR(ux_l) + SQR(uy_l) + SQR(uz_l));
    u_l[1] = ux_l;
    u_l[2] = uy_l;
    u_l[3] = uz_l;
    Real bb2_l = prim_l(IBY,i);
    Real bb3_l = prim_l(IBZ,i);

    // Extract right primitives
    Real rho_r = prim_r(IDN,i);
    Real pgas_r = prim_r(IPR,i);
    Real ux_r = prim_r(ivx,i);
    Real uy_r = prim_r(ivy,i);
    Real uz_r = prim_r(ivz,i);
    Real u_r[4];
    u_r[0] = std::sqrt(1.0 + SQR(ux_r) + SQR(uy_r) + SQR(uz_r));
    u_r[1] = ux_r;
    u_r[2] = uy_r;
    u_r[3] = uz_r;
    Real bb2_r = prim_r(IBY,i);
    Real bb3_r = prim_r(IBZ,i);

    // Extract normal magnetic field
    Real bb1 = bb_normal(i);

    // Calculate 4-magnetic field in left state
    Real b_l[4];
    b_l[0] = bb1*u_l[1] + bb2_l*u_l[2] + bb3_l*u_l[3];
    b_l[1] = (bb1 + b_l[0] * u_l[1]) / u_l[0];
    b_l[2] = (bb2_l + b_l[0] * u_l[2]) / u_l[0];
    b_l[3] = (bb3_l + b_l[0] * u_l[3]) / u_l[0];
    Real b_sq_l = -SQR(b_l[0]) + SQR(b_l[1]) + SQR(b_l[2]) + SQR(b_l[3]);

    // Calculate 4-magnetic field in right state
    Real b_r[4];
    b_r[0] = bb1*u_r[1] + bb2_r*u_r[2] + bb3_r*u_r[3];
    b_r[1] = (bb1 + b_r[0] * u_r[1]) / u_r[0];
    b_r[2] = (bb2_r + b_r[0] * u_r[2]) / u_r[0];
    b_r[3] = (bb3_r + b_r[0] * u_r[3]) / u_r[0];
    Real b_sq_r = -SQR(b_r[0]) + SQR(b_r[1]) + SQR(b_r[2]) + SQR(b_r[3]);

    // Calculate left wavespeeds
    Real wgas_l = rho_l + gamma_adi / (gamma_adi - 1.0) * pgas_l;
    Real lambda_m_l, lambda_p_l;
    pmb->peos->FastMagnetosonicSpeedsGR(wgas_l, pgas_l, u_l[0], u_l[1], b_sq_l, -1.0, 0.0,
                                        1.0, &lambda_p_l, &lambda_m_l);

    // Calculate right wavespeeds
    Real wgas_r = rho_r + gamma_adi / (gamma_adi - 1.0) * pgas_r;
    Real lambda_m_r, lambda_p_r;
    pmb->peos->FastMagnetosonicSpeedsGR(wgas_r, pgas_r, u_r[0], u_r[1], b_sq_r, -1.0, 0.0,
                                        1.0, &lambda_p_r, &lambda_m_r);

    // Calculate extremal wavespeeds (MB2006 55)
    Real lambda_l = std::min(lambda_m_l, lambda_m_r);
    Real lambda_r = std::max(lambda_p_l, lambda_p_r);

    // Calculate conserved quantities in L region (MUB 8)
    Real cons_l[NWAVE];
    Real wtot_l = wgas_l + b_sq_l;
    Real ptot_l = pgas_l + 0.5*b_sq_l;
    cons_l[IDN] = rho_l * u_l[0];
    cons_l[IEN] = wtot_l * u_l[0] * u_l[0] - b_l[0] * b_l[0] - ptot_l;
    cons_l[ivx] = wtot_l * u_l[1] * u_l[0] - b_l[1] * b_l[0];
    cons_l[ivy] = wtot_l * u_l[2] * u_l[0] - b_l[2] * b_l[0];
    cons_l[ivz] = wtot_l * u_l[3] * u_l[0] - b_l[3] * b_l[0];
    cons_l[IBY] = b_l[2] * u_l[0] - b_l[0] * u_l[2];
    cons_l[IBZ] = b_l[3] * u_l[0] - b_l[0] * u_l[3];

    // Calculate fluxes in L region (MUB 15)
    Real flux_l[NWAVE];
    flux_l[IDN] = rho_l * u_l[1];
    flux_l[IEN] = wtot_l * u_l[0] * u_l[1] - b_l[0] * b_l[1];
    flux_l[ivx] = wtot_l * u_l[1] * u_l[1] - b_l[1] * b_l[1] + ptot_l;
    flux_l[ivy] = wtot_l * u_l[2] * u_l[1] - b_l[2] * b_l[1];
    flux_l[ivz] = wtot_l * u_l[3] * u_l[1] - b_l[3] * b_l[1];
    flux_l[IBY] = b_l[2] * u_l[1] - b_l[1] * u_l[2];
    flux_l[IBZ] = b_l[3] * u_l[1] - b_l[1] * u_l[3];

    // Calculate conserved quantities in R region (MUB 8)
    Real cons_r[NWAVE];
    Real wtot_r = wgas_r + b_sq_r;
    Real ptot_r = pgas_r + 0.5*b_sq_r;
    cons_r[IDN] = rho_r * u_r[0];
    cons_r[IEN] = wtot_r * u_r[0] * u_r[0] - b_r[0] * b_r[0] - ptot_r;
    cons_r[ivx] = wtot_r * u_r[1] * u_r[0] - b_r[1] * b_r[0];
    cons_r[ivy] = wtot_r * u_r[2] * u_r[0] - b_r[2] * b_r[0];
    cons_r[ivz] = wtot_r * u_r[3] * u_r[0] - b_r[3] * b_r[0];
    cons_r[IBY] = b_r[2] * u_r[0] - b_r[0] * u_r[2];
    cons_r[IBZ] = b_r[3] * u_r[0] - b_r[0] * u_r[3];

    // Calculate fluxes in R region (MUB 15)
    Real flux_r[NWAVE];
    flux_r[IDN] = rho_r * u_r[1];
    flux_r[IEN] = wtot_r * u_r[0] * u_r[1] - b_r[0] * b_r[1];
    flux_r[ivx] = wtot_r * u_r[1] * u_r[1] - b_r[1] * b_r[1] + ptot_r;
    flux_r[ivy] = wtot_r * u_r[2] * u_r[1] - b_r[2] * b_r[1];
    flux_r[ivz] = wtot_r * u_r[3] * u_r[1] - b_r[3] * b_r[1];
    flux_r[IBY] = b_r[2] * u_r[1] - b_r[1] * u_r[2];
    flux_r[IBZ] = b_r[3] * u_r[1] - b_r[1] * u_r[3];

    // Calculate conserved quantities in HLL region in GR (MB2005 9)
    Real lambda_diff_inv = 1.0 / (lambda_r-lambda_l);
    Real cons_hll[NWAVE];
    if (GENERAL_RELATIVITY) {
      for (int n = 0; n < NWAVE; ++n) {
        cons_hll[n] = (lambda_r*cons_r[n] - lambda_l*cons_l[n] + flux_l[n] - flux_r[n])
                      * lambda_diff_inv;
      }
    }

    // Calculate fluxes in HLL region (MB2005 11)
    Real flux_hll[NWAVE];
    for (int n = 0; n < NWAVE; ++n) {
      flux_hll[n] = (lambda_r*flux_l[n] - lambda_l*flux_r[n]
                     + lambda_l*lambda_r * (cons_r[n] - cons_l[n])) * lambda_diff_inv;
    }

    // Calculate interface velocity
    Real v_interface = 0.0;
    if (GENERAL_RELATIVITY) {
      v_interface = gi(i01,i) / std::sqrt(SQR(gi(i01,i)) - gi(I00,i)*gi(i11,i));
    }

    // Determine region of wavefan
    Real *cons_interface, *flux_interface;
    if (lambda_l >= v_interface) {  // L region
      cons_interface = cons_l;
      flux_interface = flux_l;
    } else if (lambda_r <= v_interface) { // R region
      cons_interface = cons_r;
      flux_interface = flux_r;
    } else {  // HLL region
      cons_interface = cons_hll;
      flux_interface = flux_hll;
    }

    // Set conserved quantities in GR
    if (GENERAL_RELATIVITY) {
      for (int n = 0; n < NWAVE; ++n) {
        cons(n,i) = cons_interface[n];
      }
    }

    // Set fluxes
    for (int n = 0; n < NHYDRO; ++n) {
      flux(n,k,j,i) = flux_interface[n];
    }
    ey(k,j,i) = -flux_interface[IBY];
    ez(k,j,i) = flux_interface[IBZ];
  }

  return;
}

} // namespace mhd
