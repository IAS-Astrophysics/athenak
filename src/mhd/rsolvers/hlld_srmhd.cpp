//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlld_srmhd.cpp
//! \brief Implements HLLD Riemann solver for special relativistic MHD.
//   implements HLLD algorithm from Mignone, Ugliano, & Bodo 2009, MNRAS 393 1141 (MUB)
//   references Mignone & Bodo 2006, MNRAS 368 1040 (MB)
//   references Mignone & McKinney 2007, MNRAS 378 1118 (MM)
//   follows Athena 4.2, hlld_sr.c, in variable choices and magic numbers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // abs(), isfinite(), NAN, sqrt()

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for MHD (both ideal gas and isothermal)

KOKKOS_INLINE_FUNCTION
void HLLD_SR(TeamMember_t const &member, const EOS_Data &eos,
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

  par_for_inner(member, il, iu, [&](const int i)
  {

  // Parameters
  const Real p_transition = 0.01;     // value delineating intial pressure regimes
  const Real vc_extension = 1.0e-6;   // use contact region if Alfven speeds smaller
  const Real delta_kx_aug = 1.0e-12;  // amount to add to \Delta K^x

  // Calculate metric if in GR
  int i01(0), i11(0);
#if GENERAL_RELATIVITY
  {
    switch (ivx) {
      case IVX:
        pmb->pcoord->Face1Metric(k, j, il, iu, g, gi);
        i01 = I01;
        i11 = I11;
        break;
      case IVY:
        pmb->pcoord->Face2Metric(k, j, il, iu, g, gi);
        i01 = I02;
        i11 = I22;
        break;
      case IVZ:
        pmb->pcoord->Face3Metric(k, j, il, iu, g, gi);
        i01 = I03;
        i11 = I33;
        break;
    }
  }
#endif  // GENERAL_RELATIVITY

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

  // Calculate cyclic permutations of indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;

  // Extract ratio of specific heats
  const Real gamma_adi = pmb->peos->GetGamma();
  const Real gamma_prime = gamma_adi/(gamma_adi-1.0);

  // Allocate aligned memory
  Real cons_l[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real flux_l[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real cons_r[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real flux_r[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real r_l[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real r_r[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real cons_hll[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real flux_hll[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real cons_al[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real flux_al[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real cons_ar[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real flux_ar[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real cons_c[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real flux_c[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real cons_interface[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real flux_interface[NWAVE][SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real ptot_init[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real ptot_c[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real ptot_0[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real ptot_1[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real res_0[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real res_1[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real lambda_al[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  Real lambda_ar[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));
  bool switch_to_hlle[SIMD_WIDTH] __attribute__((aligned(CACHELINE_BYTES)));

  // Parameters for secant method
  const int num_secant = 4;
  const Real initial_offset = 1.0e-5;
  const Real tol_res = 1.0e-12;

  // Go through each interface
  for (int i=il; i<=iu; i+=SIMD_WIDTH) {
#pragma omp simd simdlen(SIMD_WIDTH)
    for (int m=0; m<std::min(SIMD_WIDTH, iu-i+1); m++) {
      int ipm = i+m;
      // Extract left primitives
      Real rho_l = prim_l(IDN,ipm);
      Real pgas_l = prim_l(IPR,ipm);
      Real ux_l = prim_l(ivx,ipm);
      Real uy_l = prim_l(ivy,ipm);
      Real uz_l = prim_l(ivz,ipm);
      Real u_l[4];
      u_l[0] = std::sqrt(1.0 + SQR(ux_l) + SQR(uy_l) + SQR(uz_l));
      u_l[1] = ux_l;
      u_l[2] = uy_l;
      u_l[3] = uz_l;
      Real bby_l = prim_l(IBY,ipm);
      Real bbz_l = prim_l(IBZ,ipm);

      // Extract right primitives
      Real rho_r = prim_r(IDN,ipm);
      Real pgas_r = prim_r(IPR,ipm);
      Real u_r[4];
      Real ux_r = prim_r(ivx,ipm);
      Real uy_r = prim_r(ivy,ipm);
      Real uz_r = prim_r(ivz,ipm);
      u_r[0] = std::sqrt(1.0 + SQR(ux_r) + SQR(uy_r) + SQR(uz_r));
      u_r[1] = ux_r;
      u_r[2] = uy_r;
      u_r[3] = uz_r;
      Real bby_r = prim_r(IBY,ipm);
      Real bbz_r = prim_r(IBZ,ipm);

      // Extract normal magnetic field
      Real bbx = bb_normal(ipm);

      // Calculate 4-magnetic field in left state
      Real b_l[4];
      b_l[0] = bbx*u_l[1] + bby_l*u_l[2] + bbz_l*u_l[3];
      b_l[1] = (bbx + b_l[0] * u_l[1]) / u_l[0];
      b_l[2] = (bby_l + b_l[0] * u_l[2]) / u_l[0];
      b_l[3] = (bbz_l + b_l[0] * u_l[3]) / u_l[0];
      Real b_sq_l = -SQR(b_l[0]) + SQR(b_l[1]) + SQR(b_l[2]) + SQR(b_l[3]);

      // Calculate 4-magnetic field in right state
      Real b_r[4];
      b_r[0] = bbx*u_r[1] + bby_r*u_r[2] + bbz_r*u_r[3];
      b_r[1] = (bbx + b_r[0] * u_r[1]) / u_r[0];
      b_r[2] = (bby_r + b_r[0] * u_r[2]) / u_r[0];
      b_r[3] = (bbz_r + b_r[0] * u_r[3]) / u_r[0];
      Real b_sq_r = -SQR(b_r[0]) + SQR(b_r[1]) + SQR(b_r[2]) + SQR(b_r[3]);

      // Calculate left wavespeeds
      Real wgas_l = rho_l + gamma_adi / (gamma_adi - 1.0) * pgas_l;
      pmb->peos->FastMagnetosonicSpeedsGR(wgas_l, pgas_l, u_l[0], u_l[1], b_sq_l, -1.0,
                                          0.0, 1.0, &lambdas_p_l(ipm), &lambdas_m_l(ipm));

      // Calculate right wavespeeds
      Real wgas_r = rho_r + gamma_adi / (gamma_adi - 1.0) * pgas_r;
      pmb->peos->FastMagnetosonicSpeedsGR(wgas_r, pgas_r, u_r[0], u_r[1], b_sq_r, -1.0,
                                          0.0, 1.0, &lambdas_p_r(ipm), &lambdas_m_r(ipm));

      // Calculate extremal wavespeeds (MB 55)
      Real lambda_l = std::min(lambdas_m_l(ipm), lambdas_m_r(ipm));
      Real lambda_r = std::max(lambdas_p_l(ipm), lambdas_p_r(ipm));

      // Calculate conserved quantities in L region (MUB 8)
      Real wtot_l = rho_l + gamma_prime * pgas_l + b_sq_l;
      Real ptot_l = pgas_l + 0.5*b_sq_l;
      cons_l[IDN][m] = rho_l * u_l[0];
      cons_l[IEN][m] = wtot_l * u_l[0] * u_l[0] - b_l[0] * b_l[0] - ptot_l;
      cons_l[ivx][m] = wtot_l * u_l[1] * u_l[0] - b_l[1] * b_l[0];
      cons_l[ivy][m] = wtot_l * u_l[2] * u_l[0] - b_l[2] * b_l[0];
      cons_l[ivz][m] = wtot_l * u_l[3] * u_l[0] - b_l[3] * b_l[0];
      cons_l[IBY][m] = b_l[2] * u_l[0] - b_l[0] * u_l[2];
      cons_l[IBZ][m] = b_l[3] * u_l[0] - b_l[0] * u_l[3];

      // Calculate fluxes in L region (MUB 15)
      flux_l[IDN][m] = rho_l * u_l[1];
      flux_l[IEN][m] = wtot_l * u_l[0] * u_l[1] - b_l[0] * b_l[1];
      flux_l[ivx][m] = wtot_l * u_l[1] * u_l[1] - b_l[1] * b_l[1] + ptot_l;
      flux_l[ivy][m] = wtot_l * u_l[2] * u_l[1] - b_l[2] * b_l[1];
      flux_l[ivz][m] = wtot_l * u_l[3] * u_l[1] - b_l[3] * b_l[1];
      flux_l[IBY][m] = b_l[2] * u_l[1] - b_l[1] * u_l[2];
      flux_l[IBZ][m] = b_l[3] * u_l[1] - b_l[1] * u_l[3];

      // Calculate conserved quantities in R region (MUB 8)
      Real wtot_r = rho_r + gamma_prime * pgas_r + b_sq_r;
      Real ptot_r = pgas_r + 0.5*b_sq_r;
      cons_r[IDN][m] = rho_r * u_r[0];
      cons_r[IEN][m] = wtot_r * u_r[0] * u_r[0] - b_r[0] * b_r[0] - ptot_r;
      cons_r[ivx][m] = wtot_r * u_r[1] * u_r[0] - b_r[1] * b_r[0];
      cons_r[ivy][m] = wtot_r * u_r[2] * u_r[0] - b_r[2] * b_r[0];
      cons_r[ivz][m] = wtot_r * u_r[3] * u_r[0] - b_r[3] * b_r[0];
      cons_r[IBY][m] = b_r[2] * u_r[0] - b_r[0] * u_r[2];
      cons_r[IBZ][m] = b_r[3] * u_r[0] - b_r[0] * u_r[3];

      // Calculate fluxes in R region (MUB 15)
      flux_r[IDN][m] = rho_r * u_r[1];
      flux_r[IEN][m] = wtot_r * u_r[0] * u_r[1] - b_r[0] * b_r[1];
      flux_r[ivx][m] = wtot_r * u_r[1] * u_r[1] - b_r[1] * b_r[1] + ptot_r;
      flux_r[ivy][m] = wtot_r * u_r[2] * u_r[1] - b_r[2] * b_r[1];
      flux_r[ivz][m] = wtot_r * u_r[3] * u_r[1] - b_r[3] * b_r[1];
      flux_r[IBY][m] = b_r[2] * u_r[1] - b_r[1] * u_r[2];
      flux_r[IBZ][m] = b_r[3] * u_r[1] - b_r[1] * u_r[3];

      // Calculate jump quantities across left fast wave (MUB 12)
      for (int n = 0; n < NWAVE; ++n) {
        r_l[n][m] = lambda_l * cons_l[n][m] - flux_l[n][m];
      }

      // Calculate jump quantities across right fast wave (MUB 12)
      for (int n = 0; n < NWAVE; ++n) {
        r_r[n][m] = lambda_r * cons_r[n][m] - flux_r[n][m];
      }

      // Calculate conserved quantities in HLL region (MB 29)
      Real lambda_diff_inv = 1.0 / (lambda_r - lambda_l);
      for (int n = 0; n < NWAVE; ++n) {
        cons_hll[n][m] = (r_r[n][m]-r_l[n][m]) * lambda_diff_inv;
      }

      // Calculate fluxes in HLL region (MB 31)
      for (int n = 0; n < NWAVE; ++n) {
        flux_hll[n][m] = (lambda_l*r_r[n][m] - lambda_r*r_l[n][m]) * lambda_diff_inv;
      }
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int m=0; m<std::min(SIMD_WIDTH, iu-i+1); m++) {
      int ipm = i+m;
      // Extract normal magnetic field
      Real bbx = bb_normal(ipm);
      Real ptot_hll;
      // Calculate total pressure in HLL region
      {
        // Calculate variations on conserved quantities
        Real m_sq = SQR(cons_hll[ivx][m]) + SQR(cons_hll[ivy][m]) + SQR(cons_hll[ivz][m]);
        Real bb_sq = SQR(bbx) + SQR(cons_hll[IBY][m]) + SQR(cons_hll[IBZ][m]);
        Real m_dot_bb = cons_hll[ivx][m]*bbx + cons_hll[ivy][m]*cons_hll[IBY][m]
                        + cons_hll[ivz][m]*cons_hll[IBZ][m];
        Real ss_sq = SQR(m_dot_bb);

        // Construct initial guess for enthalpy W (MM A26-A27)
        Real a1 = 4.0/3.0 * (bb_sq - cons_hll[IEN][m]);
        Real a0 = ONE_3RD * (m_sq + bb_sq * (bb_sq - 2.0*cons_hll[IEN][m]));
        Real s2 = SQR(a1) - 4.0*a0;
        Real s = (s2 < 0.0) ? 0.0 : std::sqrt(s2);
        Real w_init = (s2 >= 0.0 && a1 >= 0.0) ? -2.0*a0/(a1+s) : 0.5*(-a1+s);

        // Apply Newton-Raphson method to find new W
        const int num_nr = 2;
        Real w_new = w_init;
        Real res_new = EResidual(w_new, cons_hll[IDN][m], cons_hll[IEN][m],
                                 m_sq, bb_sq, ss_sq, gamma_prime);
        for (int n = 0; n < num_nr; ++n) {
          Real w_old = w_new;
          Real res_old = res_new;
          Real derivative = EResidualPrime(w_old, cons_hll[IDN][m],
                                           m_sq, bb_sq, ss_sq, gamma_prime);
          Real delta = -res_old / derivative;
          w_new = w_old + delta;
          res_new = EResidual(w_new, cons_hll[IDN][m], cons_hll[IEN][m],
                              m_sq, bb_sq, ss_sq, gamma_prime);
        }
        Real w = w_new;

        // Calculate primitives from W
        Real v_sq = (m_sq + ss_sq/SQR(w) * (2.0*w+bb_sq)) / SQR(w+bb_sq);  // (MM A3)
        Real gamma_lor_sq = 1.0/(1.0-v_sq);
        Real gamma_lor = std::sqrt(gamma_lor_sq);
        Real chi = (1.0-v_sq) * (w - gamma_lor*cons_hll[IDN][m]);  // (MM A11)
        Real pgas = 1.0 / gamma_prime * chi;  // (MM A17)
        Real vx = (cons_hll[ivx][m] + m_dot_bb/w * bbx) / (w+bb_sq);  // (MM A10)
        Real vy = (cons_hll[ivy][m] + m_dot_bb/w * cons_hll[IBY][m])
                  / (w+bb_sq);  // (MM A10)
        Real vz = (cons_hll[ivz][m] + m_dot_bb/w * cons_hll[IBZ][m])
                  / (w+bb_sq);  // (MM A10)

        // Calculate total pressure
        Real v_bb = vx*bbx + vy*cons_hll[IBY][m] + vz*cons_hll[IBZ][m];
        Real b_sq = bb_sq/gamma_lor_sq + SQR(v_bb);  // (MM 2)
        ptot_hll = pgas + 0.5*b_sq;
      }

      // Calculate initial guess for total pressure (MUB 53)
      if (SQR(bbx)/ptot_hll < p_transition) {  // weak magnetic field
        Real a1 = cons_hll[IEN][m] - flux_hll[ivx][m];
        Real a0 = cons_hll[ivx][m]*flux_hll[IEN][m] - flux_hll[ivx][m]*cons_hll[IEN][m];
        Real s2 = SQR(a1) - 4.0*a0;
        Real s = (s2 < 0.0) ? 0.0 : std::sqrt(s2);
        ptot_init[m] = (s2 >= 0.0 && a1 >= 0.0) ?
                       -2.0*a0/(a1+s) : (-a1+s)/2.0;  // (MUB 55)
      } else {  // strong magnetic field
        ptot_init[m] = ptot_hll;
      }

      switch_to_hlle[m] = false;
      if (!std::isfinite(ptot_init[m]) || ptot_init[m] <= 0.0) {
        switch_to_hlle[m] = true;
      }
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int m=0; m<std::min(SIMD_WIDTH, iu-i+1); m++) {
      int ipm = i+m;
      // Extract normal magnetic field
      Real bbx = bb_normal(ipm);
      // Calculate extremal wavespeeds (MB 55)
      Real lambda_l = std::min(lambdas_m_l(ipm), lambdas_m_r(ipm));
      Real lambda_r = std::max(lambdas_p_l(ipm), lambdas_p_r(ipm));

      // Prepare variables that should be preserved from secant iterations
      Real vx_al, vy_al, vz_al, vx_ar, vy_ar, vz_ar;
      Real kx_l, ky_l, kz_l, kx_r, ky_r, kz_r;
      Real eta_l, eta_r;

      // Apply secant method to find total pressure
      Real tol_ptot = 1.0e-8 * ptot_init[m];  // unused variable
      // Calculate initial pressure residual
      ptot_0[m] = ptot_init[m];
      {
        // Calculate v_aL and v_aR
        Real al = r_l[ivx][m] - lambda_l*r_l[IEN][m]
                  + ptot_0[m]*(1.0-SQR(lambda_l));  // (MUB 26)
        Real gl = SQR(r_l[IBY][m]) + SQR(r_l[IBZ][m]);  // (MUB 27)
        Real cl = r_l[ivy][m]*r_l[IBY][m] + r_l[ivz][m]*r_l[IBZ][m];  // (MUB 28)
        Real ql = -al - gl + SQR(bbx)*(1.0-SQR(lambda_l));  // (MUB 29)
        Real xl_inv = 1.0 / (bbx * (al*lambda_l*bbx+cl)
                             - (al+gl) * (lambda_l*ptot_0[m]+r_l[IEN][m]));  // (MUB 30)
        vx_al = (bbx * (al*bbx+lambda_l*cl)
                 - (al+gl) * (ptot_0[m]+r_l[ivx][m])) * xl_inv; // (MUB 23)
        vy_al = (ql*r_l[ivy][m] + r_l[IBY][m] *
                 (cl + bbx * (lambda_l*r_l[ivx][m]-r_l[IEN][m]))) * xl_inv;  // (MUB 24)
        vz_al = (ql*r_l[ivz][m] + r_l[IBZ][m] *
                 (cl + bbx * (lambda_l*r_l[ivx][m]-r_l[IEN][m]))) * xl_inv;  // (MUB 24)
        Real ar = r_r[ivx][m] - lambda_r*r_r[IEN][m]
                  + ptot_0[m]*(1.0-SQR(lambda_r));  // (MUB 26)
        Real gr = SQR(r_r[IBY][m]) + SQR(r_r[IBZ][m]);  // (MUB 27)
        Real cr = r_r[ivy][m]*r_r[IBY][m] + r_r[ivz][m]*r_r[IBZ][m];  // (MUB 28)
        Real qr = -ar - gr + SQR(bbx)*(1.0-SQR(lambda_r));  // (MUB 29)
        Real xr_inv = 1.0 / (bbx * (ar*lambda_r*bbx+cr)
                             - (ar+gr) * (lambda_r*ptot_0[m]+r_r[IEN][m]));  // (MUB 30)
        vx_ar = (bbx * (ar*bbx+lambda_r*cr)
                 - (ar+gr) * (ptot_0[m]+r_r[ivx][m])) * xr_inv;  // (MUB 23)
        vy_ar = (qr*r_r[ivy][m] + r_r[IBY][m] *
                 (cr + bbx * (lambda_r*r_r[ivx][m]-r_r[IEN][m]))) * xr_inv;  // (MUB 24)
        vz_ar = (qr*r_r[ivz][m] + r_r[IBZ][m] *
                 (cr + bbx * (lambda_r*r_r[ivx][m]-r_r[IEN][m]))) * xr_inv;  // (MUB 24)

        // Calculate B_aL and B_aR (MUB 21)
        cons_al[IBY][m] = (r_l[IBY][m] - bbx*vy_al) / (lambda_l-vx_al);
        cons_al[IBZ][m] = (r_l[IBZ][m] - bbx*vz_al) / (lambda_l-vx_al);
        cons_ar[IBY][m] = (r_r[IBY][m] - bbx*vy_ar) / (lambda_r-vx_ar);
        cons_ar[IBZ][m] = (r_r[IBZ][m] - bbx*vz_ar) / (lambda_r-vx_ar);

        // Calculate w_aL and w_aR (MUB 31)
        Real v_rm_l = vx_al*r_l[ivx][m] + vy_al*r_l[ivy][m] + vz_al*r_l[ivz][m];
        Real wtot_al = ptot_0[m] + (r_l[IEN][m]-v_rm_l) / (lambda_l-vx_al);
        Real v_rm_r = vx_ar*r_r[ivx][m] + vy_ar*r_r[ivy][m] + vz_ar*r_r[ivz][m];
        Real wtot_ar = ptot_0[m] + (r_r[IEN][m]-v_rm_r) / (lambda_r-vx_ar);

        // Calculate eta_L and eta_R (MUB 35)
        eta_l = -copysign(std::sqrt(wtot_al), bbx);
        eta_r = copysign(std::sqrt(wtot_ar), bbx);

        // Calculate K_L and K_R (MUB 43)
        Real denom_al_inv = 1.0 / (lambda_l*ptot_0[m] + r_l[IEN][m] + bbx*eta_l);
        kx_l = (r_l[ivx][m] + ptot_0[m] + lambda_l*bbx*eta_l)
               * denom_al_inv;  // R_{B^x} = \lambda B^x
        ky_l = (r_l[ivy][m] + r_l[IBY][m]*eta_l) * denom_al_inv;
        kz_l = (r_l[ivz][m] + r_l[IBZ][m]*eta_l) * denom_al_inv;
        Real denom_ar_inv = 1.0 / (lambda_r*ptot_0[m] + r_r[IEN][m] + bbx*eta_r);
        kx_r = (r_r[ivx][m] + ptot_0[m] + lambda_r*bbx*eta_r)
               * denom_ar_inv;  // R_{B^x} = \lambda B^x
        ky_r = (r_r[ivy][m] + r_r[IBY][m]*eta_r) * denom_ar_inv;
        kz_r = (r_r[ivz][m] + r_r[IBZ][m]*eta_r) * denom_ar_inv;

        // Rename Alfven wavespeeds for what they are
        lambda_al[m] = kx_l;
        lambda_ar[m] = kx_r;

        // Calculate B_c (MUB 45)
        Real delta_kx = kx_r - kx_l + delta_kx_aug;
        Real bbx_c_delta_kx = bbx * delta_kx;
        Real by_c_delta_kx = cons_ar[IBY][m]*(lambda_ar[m]-vx_ar)
                             - cons_al[IBY][m]*(lambda_al[m]-vx_al) + bbx*(vy_ar-vy_al);
        Real bz_c_delta_kx = cons_ar[IBZ][m]*(lambda_ar[m]-vx_ar)
                             - cons_al[IBZ][m]*(lambda_al[m]-vx_al) + bbx*(vz_ar-vz_al);

        // Calculate residual
        Real k_sq_l = SQR(kx_l) + SQR(ky_l) + SQR(kz_l);
        Real k_dot_bc_delta_kx = kx_l*bbx_c_delta_kx + ky_l*by_c_delta_kx
                                 + kz_l*bz_c_delta_kx;
        Real y_l = (1.0-k_sq_l) / (eta_l*delta_kx - k_dot_bc_delta_kx);  // (MUB 49)
        Real k_sq_r = SQR(kx_r) + SQR(ky_r) + SQR(kz_r);
        k_dot_bc_delta_kx = kx_r*bbx_c_delta_kx + ky_r*by_c_delta_kx
                            + kz_r*bz_c_delta_kx;
        Real y_r = (1.0-k_sq_r) / (eta_r*delta_kx - k_dot_bc_delta_kx);  // (MUB 49)
        res_0[m] = delta_kx * (1.0 - bbx * (y_r-y_l));  // (MUB 48)
      }

      // Calculate offset pressure and residual
      ptot_1[m] = ptot_init[m] * (1.0 + initial_offset);
      {
        // Calculate v_aL and v_aR
        Real al = r_l[ivx][m] - lambda_l*r_l[IEN][m]
                  + ptot_1[m]*(1.0-SQR(lambda_l));  // (MUB 26)
        Real gl = SQR(r_l[IBY][m]) + SQR(r_l[IBZ][m]);  // (MUB 27)
        Real cl = r_l[ivy][m]*r_l[IBY][m] + r_l[ivz][m]*r_l[IBZ][m];  // (MUB 28)
        Real ql = -al - gl + SQR(bbx)*(1.0-SQR(lambda_l));  // (MUB 29)
        Real xl_inv = 1.0 / (bbx * (al*lambda_l*bbx+cl)
                             - (al+gl) * (lambda_l*ptot_1[m]+r_l[IEN][m]));  // (MUB 30)
        vx_al = (bbx * (al*bbx+lambda_l*cl)
                 - (al+gl) * (ptot_1[m]+r_l[ivx][m])) * xl_inv;  // (MUB 23)
        vy_al = (ql*r_l[ivy][m] + r_l[IBY][m] *
                 (cl + bbx * (lambda_l*r_l[ivx][m]-r_l[IEN][m]))) * xl_inv;  // (MUB 24)
        vz_al = (ql*r_l[ivz][m] + r_l[IBZ][m] *
                 (cl + bbx * (lambda_l*r_l[ivx][m]-r_l[IEN][m]))) * xl_inv;  // (MUB 24)
        Real ar = r_r[ivx][m] - lambda_r*r_r[IEN][m]
                  + ptot_1[m]*(1.0-SQR(lambda_r));  // (MUB 26)
        Real gr = SQR(r_r[IBY][m]) + SQR(r_r[IBZ][m]);  // (MUB 27)
        Real cr = r_r[ivy][m]*r_r[IBY][m] + r_r[ivz][m]*r_r[IBZ][m];  // (MUB 28)
        Real qr = -ar - gr + SQR(bbx)*(1.0-SQR(lambda_r));  // (MUB 29)
        Real xr_inv = 1.0 / (bbx * (ar*lambda_r*bbx+cr)
                             - (ar+gr) * (lambda_r*ptot_1[m]+r_r[IEN][m]));  // (MUB 30)
        vx_ar = (bbx * (ar*bbx+lambda_r*cr)
                 - (ar+gr) * (ptot_1[m]+r_r[ivx][m])) * xr_inv;  // (MUB 23)
        vy_ar = (qr*r_r[ivy][m] + r_r[IBY][m] *
                 (cr + bbx * (lambda_r*r_r[ivx][m]-r_r[IEN][m]))) * xr_inv;  // (MUB 24)
        vz_ar = (qr*r_r[ivz][m] + r_r[IBZ][m] *
                 (cr + bbx * (lambda_r*r_r[ivx][m]-r_r[IEN][m]))) * xr_inv;  // (MUB 24)

        // Calculate B_aL and B_aR (MUB 21)
        cons_al[IBY][m] = (r_l[IBY][m] - bbx*vy_al) / (lambda_l-vx_al);
        cons_al[IBZ][m] = (r_l[IBZ][m] - bbx*vz_al) / (lambda_l-vx_al);
        cons_ar[IBY][m] = (r_r[IBY][m] - bbx*vy_ar) / (lambda_r-vx_ar);
        cons_ar[IBZ][m] = (r_r[IBZ][m] - bbx*vz_ar) / (lambda_r-vx_ar);

        // Calculate w_aL and w_aR (MUB 31)
        Real v_rm_l = vx_al*r_l[ivx][m] + vy_al*r_l[ivy][m] + vz_al*r_l[ivz][m];
        Real wtot_al = ptot_1[m] + (r_l[IEN][m]-v_rm_l) / (lambda_l-vx_al);
        Real v_rm_r = vx_ar*r_r[ivx][m] + vy_ar*r_r[ivy][m] + vz_ar*r_r[ivz][m];
        Real wtot_ar = ptot_1[m] + (r_r[IEN][m]-v_rm_r) / (lambda_r-vx_ar);

        // Calculate eta_L and eta_R (MUB 35)
        eta_l = -copysign(std::sqrt(wtot_al), bbx);
        eta_r = copysign(std::sqrt(wtot_ar), bbx);

        // Calculate K_L and K_R (MUB 43)
        Real denom_al_inv = 1.0 / (lambda_l*ptot_1[m] + r_l[IEN][m] + bbx*eta_l);
        kx_l = (r_l[ivx][m] + ptot_1[m] + lambda_l*bbx*eta_l)
               * denom_al_inv;  // R_{B^x} = \lambda B^x
        ky_l = (r_l[ivy][m] + r_l[IBY][m]*eta_l) * denom_al_inv;
        kz_l = (r_l[ivz][m] + r_l[IBZ][m]*eta_l) * denom_al_inv;
        Real denom_ar_inv = 1.0 / (lambda_r*ptot_1[m] + r_r[IEN][m] + bbx*eta_r);
        kx_r = (r_r[ivx][m] + ptot_1[m] + lambda_r*bbx*eta_r)
               * denom_ar_inv;  // R_{B^x} = \lambda B^x
        ky_r = (r_r[ivy][m] + r_r[IBY][m]*eta_r) * denom_ar_inv;
        kz_r = (r_r[ivz][m] + r_r[IBZ][m]*eta_r) * denom_ar_inv;

        // Rename Alfven wavespeeds for what they are
        lambda_al[m] = kx_l;
        lambda_ar[m] = kx_r;

        // Calculate B_c (MUB 45)
        Real delta_kx = kx_r - kx_l + delta_kx_aug;
        Real bbx_c_delta_kx = bbx * delta_kx;
        Real by_c_delta_kx = cons_ar[IBY][m]*(lambda_ar[m]-vx_ar)
                             - cons_al[IBY][m]*(lambda_al[m]-vx_al) + bbx*(vy_ar-vy_al);
        Real bz_c_delta_kx = cons_ar[IBZ][m]*(lambda_ar[m]-vx_ar)
                             - cons_al[IBZ][m]*(lambda_al[m]-vx_al) + bbx*(vz_ar-vz_al);

        // Calculate residual
        Real k_sq_l = SQR(kx_l) + SQR(ky_l) + SQR(kz_l);
        Real k_dot_bc_delta_kx = kx_l*bbx_c_delta_kx + ky_l*by_c_delta_kx
                                 + kz_l*bz_c_delta_kx;
        Real y_l = (1.0-k_sq_l) / (eta_l*delta_kx - k_dot_bc_delta_kx);  // (MUB 49)
        Real k_sq_r = SQR(kx_r) + SQR(ky_r) + SQR(kz_r);
        k_dot_bc_delta_kx = kx_r*bbx_c_delta_kx + ky_r*by_c_delta_kx
                            + kz_r*bz_c_delta_kx;
        Real y_r = (1.0-k_sq_r) / (eta_r*delta_kx - k_dot_bc_delta_kx);  // (MUB 49)
        res_1[m] = delta_kx * (1.0 - bbx * (y_r-y_l));  // (MUB 48)
      }
    }

#pragma omp simd simdlen(SIMD_WIDTH)
    for (int m=0; m<std::min(SIMD_WIDTH,iu-i+1); m++) {
      int ipm = i+m;
      // Extract normal magnetic field
      Real bbx = bb_normal(ipm);
      // Calculate extremal wavespeeds (MB 55)
      Real lambda_l = std::min(lambdas_m_l(ipm), lambdas_m_r(ipm));
      Real lambda_r = std::max(lambdas_p_l(ipm), lambdas_p_r(ipm));

      // Iterate via secant method
      Real ptot_last = ptot_0[m];
      Real ptot_n = ptot_1[m];
      Real res_last = res_0[m];
      Real res_n = res_1[m];
      Real tol_ptot = 1.0e-8 * ptot_init[m];

      // Prepare variables that should be preserved from secant iterations
      Real vx_al, vy_al, vz_al, vx_ar, vy_ar, vz_ar;
      Real kx_l, ky_l, kz_l, kx_r, ky_r, kz_r;
      Real eta_l, eta_r;

      for (int n = 0; n < num_secant; ++n) {
        // Calculate new guess for pressure
        Real ptot_old = ptot_last;
        ptot_last = ptot_n;
        Real res_old = res_last;
        res_last = res_n;
        bool need_to_iterate = std::abs(res_last) > tol_res
                               && std::abs(ptot_last-ptot_old) > tol_ptot;
        if (need_to_iterate) {
          ptot_n = (res_last*ptot_old - res_old*ptot_last) / (res_last-res_old);
        } else {
          ptot_n = ptot_last;
        }

        // Calculate v_aL and v_aR
        Real al = r_l[ivx][m] - lambda_l*r_l[IEN][m]
                  + ptot_n*(1.0-SQR(lambda_l));  // (MUB 26)
        Real gl = SQR(r_l[IBY][m]) + SQR(r_l[IBZ][m]);  // (MUB 27)
        Real cl = r_l[ivy][m]*r_l[IBY][m] + r_l[ivz][m]*r_l[IBZ][m];  // (MUB 28)
        Real ql = -al - gl + SQR(bbx)*(1.0-SQR(lambda_l));  // (MUB 29)
        Real xl_inv = 1.0 / (bbx * (al*lambda_l*bbx+cl)
                             - (al+gl) * (lambda_l*ptot_n+r_l[IEN][m]));  // (MUB 30)
        vx_al = (bbx * (al*bbx+lambda_l*cl)
                 - (al+gl) * (ptot_n+r_l[ivx][m])) * xl_inv;  // (MUB 23)
        vy_al = (ql*r_l[ivy][m] + r_l[IBY][m]
                 * (cl + bbx * (lambda_l*r_l[ivx][m] -r_l[IEN][m]))) * xl_inv; // (MUB 24)
        vz_al = (ql*r_l[ivz][m] + r_l[IBZ][m]
                 * (cl + bbx * (lambda_l*r_l[ivx][m] -r_l[IEN][m]))) * xl_inv; // (MUB 24)
        Real ar = r_r[ivx][m] - lambda_r*r_r[IEN][m]
                  + ptot_n*(1.0-SQR(lambda_r));  // (MUB 26)
        Real gr = SQR(r_r[IBY][m]) + SQR(r_r[IBZ][m]);  // (MUB 27)
        Real cr = r_r[ivy][m]*r_r[IBY][m] + r_r[ivz][m]*r_r[IBZ][m];  // (MUB 28)
        Real qr = -ar - gr + SQR(bbx)*(1.0-SQR(lambda_r));  // (MUB 29)
        Real xr_inv = 1.0 / (bbx * (ar*lambda_r*bbx+cr)
                             - (ar+gr) * (lambda_r*ptot_n+r_r[IEN][m]));  // (MUB 30)
        vx_ar = (bbx * (ar*bbx+lambda_r*cr)
                 - (ar+gr) * (ptot_n+r_r[ivx][m])) * xr_inv;  // (MUB 23)
        vy_ar = (qr*r_r[ivy][m] + r_r[IBY][m]
                 * (cr + bbx * (lambda_r*r_r[ivx][m] -r_r[IEN][m]))) * xr_inv; // (MUB 24)
        vz_ar = (qr*r_r[ivz][m] + r_r[IBZ][m]
                 * (cr + bbx * (lambda_r*r_r[ivx][m] -r_r[IEN][m]))) * xr_inv; // (MUB 24)

        // Calculate B_aL and B_aR (MUB 21)
        cons_al[IBY][m] = (r_l[IBY][m] - bbx*vy_al) / (lambda_l-vx_al);
        cons_al[IBZ][m] = (r_l[IBZ][m] - bbx*vz_al) / (lambda_l-vx_al);
        cons_ar[IBY][m] = (r_r[IBY][m] - bbx*vy_ar) / (lambda_r-vx_ar);
        cons_ar[IBZ][m] = (r_r[IBZ][m] - bbx*vz_ar) / (lambda_r-vx_ar);

        // Calculate w_aL and w_aR (MUB 31)
        Real v_rm_l = vx_al*r_l[ivx][m] + vy_al*r_l[ivy][m] + vz_al*r_l[ivz][m];
        Real wtot_al = ptot_n + (r_l[IEN][m]-v_rm_l) / (lambda_l-vx_al);
        Real v_rm_r = vx_ar*r_r[ivx][m] + vy_ar*r_r[ivy][m] + vz_ar*r_r[ivz][m];
        Real wtot_ar = ptot_n + (r_r[IEN][m]-v_rm_r) / (lambda_r-vx_ar);

        // Calculate eta_L and eta_R (MUB 35)
        eta_l = -copysign(std::sqrt(wtot_al), bbx);
        eta_r = copysign(std::sqrt(wtot_ar), bbx);

        // Calculate K_L and K_R (MUB 43)
        Real denom_al_inv = 1.0 / (lambda_l*ptot_n + r_l[IEN][m] + bbx*eta_l);
        kx_l = (r_l[ivx][m] + ptot_n + lambda_l*bbx*eta_l)
               * denom_al_inv;  // R_{B^x} = \lambda B^x
        ky_l = (r_l[ivy][m] + r_l[IBY][m]*eta_l) * denom_al_inv;
        kz_l = (r_l[ivz][m] + r_l[IBZ][m]*eta_l) * denom_al_inv;
        Real denom_ar_inv = 1.0 / (lambda_r*ptot_n + r_r[IEN][m] + bbx*eta_r);
        kx_r = (r_r[ivx][m] + ptot_n + lambda_r*bbx*eta_r)
               * denom_ar_inv;  // R_{B^x} = \lambda B^x
        ky_r = (r_r[ivy][m] + r_r[IBY][m]*eta_r) * denom_ar_inv;
        kz_r = (r_r[ivz][m] + r_r[IBZ][m]*eta_r) * denom_ar_inv;

        // Rename Alfven wavespeeds for what they are
        lambda_al[m] = kx_l;
        lambda_ar[m] = kx_r;

        // Calculate B_c (MUB 45)
        Real delta_kx = kx_r - kx_l + delta_kx_aug;
        Real bbx_c_delta_kx = bbx * delta_kx;
        Real by_c_delta_kx = cons_ar[IBY][m]*(lambda_ar[m]-vx_ar)
                             - cons_al[IBY][m]*(lambda_al[m]-vx_al) + bbx*(vy_ar-vy_al);
        Real bz_c_delta_kx = cons_ar[IBZ][m]*(lambda_ar[m]-vx_ar)
                             - cons_al[IBZ][m]*(lambda_al[m]-vx_al) + bbx*(vz_ar-vz_al);

        // Calculate residual
        Real k_sq_l = SQR(kx_l) + SQR(ky_l) + SQR(kz_l);
        Real k_dot_bc_delta_kx = kx_l*bbx_c_delta_kx + ky_l*by_c_delta_kx
                                 + kz_l*bz_c_delta_kx;
        Real y_l = (1.0-k_sq_l) / (eta_l*delta_kx - k_dot_bc_delta_kx);  // (MUB 49)
        Real k_sq_r = SQR(kx_r) + SQR(ky_r) + SQR(kz_r);
        k_dot_bc_delta_kx = kx_r*bbx_c_delta_kx + ky_r*by_c_delta_kx
                            + kz_r*bz_c_delta_kx;
        Real y_r = (1.0-k_sq_r) / (eta_r*delta_kx - k_dot_bc_delta_kx);  // (MUB 49)
        res_n = delta_kx * (1.0 - bbx * (y_r-y_l));  // (MUB 48)
      }

      // Set total contact pressure
      ptot_c[m] = ptot_n;

      if (!std::isfinite(lambda_al[m]) || !std::isfinite(lambda_ar[m]) ||
          !std::isfinite(ptot_c[m]) || ptot_c[m] <= 0.0) {
        switch_to_hlle[m] = true;
      }

      // Calculate remaining conserved quantities in aL region
      Real v_bb_al = vx_al*bbx + vy_al*cons_al[IBY][m] + vz_al*cons_al[IBZ][m];
      cons_al[IDN][m] = r_l[IDN][m] / (lambda_l-vx_al);  // (MUB 32)
      cons_al[IEN][m] = (r_l[IEN][m] + ptot_c[m]*vx_al - v_bb_al*bbx)
                        / (lambda_l-vx_al);  // (MUB 33)
      cons_al[ivx][m] = (cons_al[IEN][m] + ptot_c[m]) * vx_al
                        - v_bb_al * bbx;  // (MUB 34)
      cons_al[ivy][m] = (cons_al[IEN][m] + ptot_c[m]) * vy_al
                        - v_bb_al * cons_al[IBY][m];  // (MUB 34)
      cons_al[ivz][m] = (cons_al[IEN][m] + ptot_c[m]) * vz_al
                        - v_bb_al * cons_al[IBZ][m]; // (MUB 34)

      // Calculate remaining conserved quantities in aR region
      Real v_bb_ar = vx_ar*bbx + vy_ar*cons_ar[IBY][m] + vz_ar*cons_ar[IBZ][m];
      cons_ar[IDN][m] = r_r[IDN][m] / (lambda_r-vx_ar);  // (MUB 32)
      cons_ar[IEN][m] = (r_r[IEN][m] + ptot_c[m]*vx_ar - v_bb_ar*bbx)
                        / (lambda_r-vx_ar);  // (MUB 33)
      cons_ar[ivx][m] = (cons_ar[IEN][m] + ptot_c[m]) * vx_ar
                        - v_bb_ar * bbx;  // (MUB 34)
      cons_ar[ivy][m] = (cons_ar[IEN][m] + ptot_c[m]) * vy_ar
                        - v_bb_ar * cons_ar[IBY][m];  // (MUB 34)
      cons_ar[ivz][m] = (cons_ar[IEN][m] + ptot_c[m]) * vz_ar
                        - v_bb_ar * cons_ar[IBZ][m];  // (MUB 34)

      // Calculate fluxes in aL region (MUB 11,12)
      for (int n = 0; n < NWAVE; ++n) {
        flux_al[n][m] = lambda_l * cons_al[n][m] - r_l[n][m];
      }

      // Calculate fluxes in aR region (MUB 11,12)
      for (int n = 0; n < NWAVE; ++n) {
        flux_ar[n][m] = lambda_r * cons_ar[n][m] - r_r[n][m];
      }

      // Calculate B_c (MUB 45)
      Real denom_c_inv = 1.0 / (lambda_ar[m] - lambda_al[m] + delta_kx_aug);
      Real numer_al = cons_al[IBY][m] * (lambda_al[m]-vx_al) + bbx*vy_al;
      Real numer_ar = cons_ar[IBY][m] * (lambda_ar[m]-vx_ar) + bbx*vy_ar;
      cons_c[IBY][m] = (numer_ar - numer_al) * denom_c_inv;
      numer_al = cons_al[IBZ][m] * (lambda_al[m]-vx_al) + bbx*vz_al;
      numer_ar = cons_ar[IBZ][m] * (lambda_ar[m]-vx_ar) + bbx*vz_ar;
      cons_c[IBZ][m] = (numer_ar - numer_al) * denom_c_inv;

      // Calculate v_c (MUB 47), averaging left and right values in case of disagreement
      Real k_sq_l = SQR(kx_l) + SQR(ky_l) + SQR(kz_l);
      Real k_bc_l = kx_l*bbx + ky_l*cons_c[IBY][m] + kz_l*cons_c[IBZ][m];
      Real vx_cl = kx_l - bbx * (1.0-k_sq_l) / (eta_l-k_bc_l);
      Real vy_cl = ky_l - cons_c[IBY][m] * (1.0-k_sq_l) / (eta_l-k_bc_l);
      Real vz_cl = kz_l - cons_c[IBZ][m] * (1.0-k_sq_l) / (eta_l-k_bc_l);
      Real k_sq_r = SQR(kx_r) + SQR(ky_r) + SQR(kz_r);
      Real k_bc_r = kx_r*bbx + ky_r*cons_c[IBY][m] + kz_r*cons_c[IBZ][m];
      Real vx_cr = kx_r - bbx * (1.0-k_sq_r) / (eta_r-k_bc_r);
      Real vy_cr = ky_r - cons_c[IBY][m] * (1.0-k_sq_r) / (eta_r-k_bc_r);
      Real vz_cr = kz_r - cons_c[IBZ][m] * (1.0-k_sq_r) / (eta_r-k_bc_r);
      Real vx_c = 0.5 * (vx_cl + vx_cr);
      Real vy_c = 0.5 * (vy_cl + vy_cr);
      Real vz_c = 0.5 * (vz_cl + vz_cr);

      // Calculate remaining conserved quantities in c region
      Real v_bb_c = vx_c*bbx + vy_c*cons_c[IBY][m] + vz_c*cons_c[IBZ][m];
      if (vx_c >= 0.0) {  // cL region
        cons_c[IDN][m] = cons_al[IDN][m] * (lambda_al[m]-vx_al)
                         / (lambda_al[m]-vx_c);  // (MUB 50)
        cons_c[IEN][m] = (lambda_al[m]*cons_al[IEN][m] - cons_al[ivx][m] + ptot_c[m]*vx_c
                          - v_bb_c*bbx) / (lambda_al[m]-vx_c);  // (MUB 51)
      } else {  // cR region
        cons_c[IDN][m] = cons_ar[IDN][m] * (lambda_ar[m]-vx_ar)
                         / (lambda_ar[m]-vx_c);  // (MUB 50)
        cons_c[IEN][m] = (lambda_ar[m]*cons_ar[IEN][m] - cons_ar[ivx][m] + ptot_c[m]*vx_c
                          - v_bb_c*bbx) / (lambda_ar[m]-vx_c);  // (MUB 51)
      }
      cons_c[ivx][m] = (cons_c[IEN][m] + ptot_c[m]) * vx_c
                       - v_bb_c * bbx;  // (MUB 52)
      cons_c[ivy][m] = (cons_c[IEN][m] + ptot_c[m]) * vy_c
                       - v_bb_c * cons_c[IBY][m];  // (MUB 52)
      cons_c[ivz][m] = (cons_c[IEN][m] + ptot_c[m]) * vz_c
                       - v_bb_c * cons_c[IBZ][m];  // (MUB 52)

      // Calculate fluxes in c region (MUB 11)
      for (int n = 0; n < NWAVE; ++n) {
        if (vx_c >= 0.0) {  // cL region
          flux_c[n][m] = flux_al[n][m] + lambda_al[m] * (cons_c[n][m] - cons_al[n][m]);
        } else {  // cR region
          flux_c[n][m] = flux_ar[n][m] + lambda_ar[m] * (cons_c[n][m] - cons_ar[n][m]);
        }
      }
    }

    for (int m=0; m<std::min(SIMD_WIDTH,iu-i+1); m++) {
      int ipm = i+m;
      // Calculate extremal wavespeeds (MB 55)
      Real lambda_l = std::min(lambdas_m_l(ipm), lambdas_m_r(ipm));
      Real lambda_r = std::max(lambdas_p_l(ipm), lambdas_p_r(ipm));

      // Calculate interface velocity
      Real v_interface = 0.0;
      if (GENERAL_RELATIVITY) {
        v_interface = gi(i01,ipm) / std::sqrt(SQR(gi(i01,ipm)) - gi(I00,ipm)*gi(i11,ipm));
      }

      // Determine region of wavefan
      if (lambda_l >= v_interface) {  // L region
        for (int n = 0; n < NWAVE; ++n) {
          cons_interface[n][m] = cons_l[n][m];
          flux_interface[n][m] = flux_l[n][m];
        }
      } else if (lambda_r <= v_interface) { // R region
        for (int n = 0; n < NWAVE; ++n) {
          cons_interface[n][m] = cons_r[n][m];
          flux_interface[n][m] = flux_r[n][m];
        }
      } else if (switch_to_hlle[m]) {  // HLL region
        for (int n = 0; n < NWAVE; ++n) {
          cons_interface[n][m] = cons_hll[n][m];
          flux_interface[n][m] = flux_hll[n][m];
        }
      } else if (lambda_al[m] >= v_interface-vc_extension) {  // aL region
        for (int n = 0; n < NWAVE; ++n) {
          cons_interface[n][m] = cons_al[n][m];
          flux_interface[n][m] = flux_al[n][m];
        }
      } else if (lambda_ar[m] <= v_interface+vc_extension) {  // aR region
        for (int n = 0; n < NWAVE; ++n) {
          cons_interface[n][m] = cons_ar[n][m];
          flux_interface[n][m] = flux_ar[n][m];
        }
      } else {  // c region
        for (int n = 0; n < NWAVE; ++n) {
          cons_interface[n][m] = cons_c[n][m];
          flux_interface[n][m] = flux_c[n][m];
        }
      }

      // Check for any remaining HLLD failures and switch to HLLE if found
      if (!switch_to_hlle[m]) {
        for (int n = 0; n < NWAVE; ++n) {
          if (!std::isfinite(cons_interface[n][m])
              || !std::isfinite(flux_interface[n][m])) {
            switch_to_hlle[m] = true;
          }
        }
        if (switch_to_hlle[m]) {
          for (int n = 0; n < NWAVE; ++n) {
            cons_interface[n][m] = cons_hll[n][m];
            flux_interface[n][m] = flux_hll[n][m];
          }
        }
      }

      if (GENERAL_RELATIVITY) {
        for (int n = 0; n < NWAVE; ++n) {
          cons(n,ipm) = cons_interface[n][m];
        }
      }

      // Set fluxes
      for (int n = 0; n < NHYDRO; ++n) {
        flux(n,k,j,ipm) = flux_interface[n][m];
      }
      ey(k,j,ipm) = -flux_interface[IBY][m];
      ez(k,j,ipm) = flux_interface[IBZ][m];
    }
  }

  // Transform fluxes to global coordinates if in GR
#if GENERAL_RELATIVITY
  {
    switch (ivx) {
      case IVX:
        pmb->pcoord->FluxToGlobal1(k, j, il, iu, cons, bb_normal, flux, ey, ez);
        break;
      case IVY:
        pmb->pcoord->FluxToGlobal2(k, j, il, iu, cons, bb_normal, flux, ey, ez);
        break;
      case IVZ:
        pmb->pcoord->FluxToGlobal3(k, j, il, iu, cons, bb_normal, flux, ey, ez);
        break;
    }
  }
#endif  // GENERAL_RELATIVITY
  return;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct enthalpy
// Inputs:
//   w_guess: guess for total enthalpy W
//   dd: relativistic density D
//   ee: total energy E
//   m_sq: square magnitude of momentum \vec{m}
//   bb_sq: square magnitude of magnetic field \vec{B}
//   ss_sq: (\vec{m} \cdot \vec{B})^2
//   gamma_prime: reduced adiabatic gas constant Gamma' = Gamma/(Gamma-1)
// Outputs:
//   returned value: calculated minus given value of E
// Notes:
//   follows Mignone & McKinney 2007, MNRAS 378 1118 (MM)
//   implementation follows that of hlld_sr.c in Athena 4.2

Real EResidual(Real w_guess, Real dd, Real ee, Real m_sq, Real bb_sq, Real ss_sq,
               Real gamma_prime) {
  Real v_sq = (m_sq + ss_sq/SQR(w_guess) * (2.0*w_guess + bb_sq))
              / SQR(w_guess + bb_sq);                                      // (cf. MM A3)
  Real gamma_sq = 1.0/(1.0-v_sq);
  Real gamma_lorentz = std::sqrt(gamma_sq);
  Real chi = (1.0 - v_sq) * (w_guess - gamma_lorentz * dd);        // (cf. MM A11)
  Real pgas = chi/gamma_prime;                                     // (MM A17)
  Real ee_calc = w_guess - pgas + 0.5*bb_sq * (1.0+v_sq)
                 - ss_sq / (2.0*SQR(w_guess));                                // (MM A1)
  return ee_calc - ee;
}

//----------------------------------------------------------------------------------------
// Derivative of EResidual()
// Inputs:
//   w_guess: guess for total enthalpy W
//   dd: relativistic density D
//   m_sq: square magnitude of momentum \vec{m}
//   bb_sq: square magnitude of magnetic field \vec{B}
//   ss_sq: (\vec{m} \cdot \vec{B})^2
//   gamma_prime: reduced adiabatic gas constant Gamma' = Gamma/(Gamma-1)
// Outputs:
//   returned value: derivative of calculated value of E
// Notes:
//   follows Mignone & McKinney 2007, MNRAS 378 1118 (MM)
//   implementation follows that of hlld_sr.c in Athena 4.2

Real EResidualPrime(Real w_guess, Real dd, Real m_sq, Real bb_sq, Real ss_sq,
                    Real gamma_prime) {
  Real v_sq = (m_sq + ss_sq/SQR(w_guess) * (2.0*w_guess + bb_sq))
              / SQR(w_guess + bb_sq);                                 // (cf. MM A3)
  Real gamma_sq = 1.0/(1.0-v_sq);
  Real gamma_lorentz = std::sqrt(gamma_sq);
  Real chi = (1.0 - v_sq) * (w_guess - gamma_lorentz * dd);           // (cf. MM A11)
  Real w_cu = SQR(w_guess) * w_guess;
  Real w_b_cu = SQR(w_guess + bb_sq) * (w_guess + bb_sq);
  Real dv_sq_dw = -2.0 / (w_cu*w_b_cu)                                // (MM A16)
                  * (ss_sq* (3.0*w_guess*(w_guess+bb_sq) + SQR(bb_sq)) + m_sq*w_cu);
  Real dchi_dw = 1.0 - v_sq                                           // (cf. MM A14)
                 - gamma_lorentz/2.0 * (dd + 2.0*gamma_lorentz*chi) * dv_sq_dw;
  Real drho_dw = -gamma_lorentz*dd/2.0 * dv_sq_dw;                    // (MM A15)
  Real dpgas_dchi = 1.0/gamma_prime;                                  // (MM A18)
  Real dpgas_drho = 0.0;                                              // (MM A18)
  Real dpgas_dw = dpgas_dchi * dchi_dw + dpgas_drho * drho_dw;
  return 1.0 - dpgas_dw + 0.5*bb_sq * dv_sq_dw + ss_sq/w_cu;
}

} // namespace mhd
