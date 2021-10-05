//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srmhd.cpp
//! \brief Implements local Lax-Friedrichs Riemann solver for special relativistic MHD.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for SR MHD

KOKKOS_INLINE_FUNCTION
void LLF(TeamMember_t const &member, const EOS_Data &eos,
     const int m, const int k, const int j,  const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     const DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez)
{
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  Real igm1 = 1.0/(eos.gamma - 1.0);
  Real iso_cs = eos.iso_cs;


  par_for_inner(member, il, iu, [&](const int i)
  {
    //--- Step 1.  Create local references for L/R states (helps compiler vectorize)


#pragma omp simd simdlen(SIMD_WIDTH)
    for (int i=il; i<=iu; ++i) {
      bb_normal(i) = bb(k,j,i);
    }

  // Calculate cyclic permutations of indices

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

    // Calculate extremal wavespeeds
    Real lambda_l = std::min(lambda_m_l, lambda_m_r);  // (MB 55)
    Real lambda_r = std::max(lambda_p_l, lambda_p_r);  // (MB 55)
    Real lambda = std::max(lambda_r, -lambda_l);

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

    // Set conserved quantities in GR
    if (GENERAL_RELATIVITY) {
      for (int n = 0; n < NWAVE; ++n) {
        cons(n,i) = 0.5 * (cons_r[n] + cons_l[n] + (flux_l[n] - flux_r[n]) / lambda);
      }
    }

    // Set fluxes
    for (int n = 0; n < NHYDRO; ++n) {
      flux(n,k,j,i) = 0.5 * (flux_l[n] + flux_r[n] - lambda * (cons_r[n] - cons_l[n]));
    }
    ey(k,j,i) = -0.5 * (flux_l[IBY] + flux_r[IBY] - lambda * (cons_r[IBY] - cons_l[IBY]));
    ez(k,j,i) = 0.5 * (flux_l[IBZ] + flux_r[IBZ] - lambda * (cons_r[IBZ] - cons_l[IBZ]));
  }

  return;
}

} // namespace mhd
