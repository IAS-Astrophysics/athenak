//========================================================================================
// Athena++ (Kokkos version) astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_srmhd.cpp
//! \brief Local Lax-Friedrichs (LLF) Riemann solver for special relativistic MHD.

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

namespace mhd {

//----------------------------------------------------------------------------------------
//! \fn void LLF
//  \brief The LLF Riemann solver for SR MHD

KOKKOS_INLINE_FUNCTION
void LLF_SR(TeamMember_t const &member, const EOS_Data &eos,
     const RegionIndcs &indcs,const DualArray1D<RegionSize> &size,const CoordData &coord,
     const int m, const int k, const int j, const int il, const int iu, const int ivx,
     const ScrArray2D<Real> &wl, const ScrArray2D<Real> &wr,
     const ScrArray2D<Real> &bl, const ScrArray2D<Real> &br, const DvceArray4D<Real> &bx,
     DvceArray5D<Real> flx, DvceArray4D<Real> ey, DvceArray4D<Real> ez) {
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  int iby = ((ivx-IVX) + 1)%3;
  int ibz = ((ivx-IVX) + 2)%3;
  const Real gm1 = (eos.gamma - 1.0);
  const Real gamma_prime = eos.gamma/gm1;

  par_for_inner(member, il, iu, [&](const int i) {
    // Extract left primitives
    Real rho_l = wl(IDN,i);
    Real ux_l = wl(ivx,i);
    Real uy_l = wl(ivy,i);
    Real uz_l = wl(ivz,i);
    Real u_l[4];
    u_l[0] = sqrt(1.0 + SQR(ux_l) + SQR(uy_l) + SQR(uz_l));
    u_l[1] = ux_l;
    u_l[2] = uy_l;
    u_l[3] = uz_l;
    Real bb2_l = bl(iby,i);
    Real bb3_l = bl(ibz,i);

    // Extract right primitives
    Real rho_r = wr(IDN,i);
    Real ux_r = wr(ivx,i);
    Real uy_r = wr(ivy,i);
    Real uz_r = wr(ivz,i);
    Real u_r[4];
    u_r[0] = sqrt(1.0 + SQR(ux_r) + SQR(uy_r) + SQR(uz_r));
    u_r[1] = ux_r;
    u_r[2] = uy_r;
    u_r[3] = uz_r;
    Real bb2_r = br(iby,i);
    Real bb3_r = br(ibz,i);

    Real pgas_l, pgas_r;
    pgas_l = eos.IdealGasPressure(wl(IDN,i), wl(IEN,i));
    pgas_r = eos.IdealGasPressure(wr(IDN,i), wr(IEN,i));

    // Extract normal magnetic field
    Real bb1 = bx(m,k,j,i);

    // Calculate 4-magnetic field in left state
    Real b_l[4];
    b_l[0] = bb1*u_l[1] + bb2_l*u_l[2] + bb3_l*u_l[3];
    b_l[1] = (bb1   + b_l[0] * u_l[1]) / u_l[0];
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
    Real lm_l, lp_l;
    eos.IdealSRMHDFastSpeeds(rho_l, pgas_l, u_l[1], u_l[0], b_sq_l, lp_l, lm_l);

    // Calculate right wavespeeds
    Real lm_r, lp_r;
    eos.IdealSRMHDFastSpeeds(rho_r, pgas_r, u_r[1], u_r[0], b_sq_r, lp_r, lm_r);

    // Calculate extremal wavespeeds
    Real lambda_l = fmin(lm_l, lm_r);  // (MB 55)
    Real lambda_r = fmax(lp_l, lp_r);  // (MB 55)
    Real lambda = fmax(lambda_r, -lambda_l);

    // Calculate conserved quantities in L region (MUB 8)
    MHDCons1D consl;
    Real wgas_l = rho_l + gamma_prime * pgas_l;
    Real wtot_l = wgas_l + b_sq_l;
    Real ptot_l = pgas_l + 0.5*b_sq_l;
    consl.d  = rho_l * u_l[0];
    consl.e  = wtot_l * u_l[0] * u_l[0] - b_l[0] * b_l[0] - ptot_l;
    consl.mx = wtot_l * u_l[1] * u_l[0] - b_l[1] * b_l[0];
    consl.my = wtot_l * u_l[2] * u_l[0] - b_l[2] * b_l[0];
    consl.mz = wtot_l * u_l[3] * u_l[0] - b_l[3] * b_l[0];
    consl.by = b_l[2] * u_l[0] - b_l[0] * u_l[2];
    consl.bz = b_l[3] * u_l[0] - b_l[0] * u_l[3];

    // Calculate fluxes in L region (MUB 15)
    MHDCons1D fl;
    fl.d  = rho_l * u_l[1];
    fl.e  = wtot_l * u_l[0] * u_l[1] - b_l[0] * b_l[1];
    fl.mx = wtot_l * u_l[1] * u_l[1] - b_l[1] * b_l[1] + ptot_l;
    fl.my = wtot_l * u_l[2] * u_l[1] - b_l[2] * b_l[1];
    fl.mz = wtot_l * u_l[3] * u_l[1] - b_l[3] * b_l[1];
    fl.by = b_l[2] * u_l[1] - b_l[1] * u_l[2];
    fl.bz = b_l[3] * u_l[1] - b_l[1] * u_l[3];

    // Calculate conserved quantities in R region (MUB 8)
    MHDCons1D consr;
    Real wgas_r = rho_r + gamma_prime * pgas_r;
    Real wtot_r = wgas_r + b_sq_r;
    Real ptot_r = pgas_r + 0.5*b_sq_r;
    consr.d  = rho_r * u_r[0];
    consr.e  = wtot_r * u_r[0] * u_r[0] - b_r[0] * b_r[0] - ptot_r;
    consr.mx = wtot_r * u_r[1] * u_r[0] - b_r[1] * b_r[0];
    consr.my = wtot_r * u_r[2] * u_r[0] - b_r[2] * b_r[0];
    consr.mz = wtot_r * u_r[3] * u_r[0] - b_r[3] * b_r[0];
    consr.by = b_r[2] * u_r[0] - b_r[0] * u_r[2];
    consr.bz = b_r[3] * u_r[0] - b_r[0] * u_r[3];

    // Calculate fluxes in R region (MUB 15)
    MHDCons1D fr;
    fr.d  = rho_r * u_r[1];
    fr.e  = wtot_r * u_r[0] * u_r[1] - b_r[0] * b_r[1];
    fr.mx = wtot_r * u_r[1] * u_r[1] - b_r[1] * b_r[1] + ptot_r;
    fr.my = wtot_r * u_r[2] * u_r[1] - b_r[2] * b_r[1];
    fr.mz = wtot_r * u_r[3] * u_r[1] - b_r[3] * b_r[1];
    fr.by = b_r[2] * u_r[1] - b_r[1] * u_r[2];
    fr.bz = b_r[3] * u_r[1] - b_r[1] * u_r[3];

    // Store results in 3D array of fluxes
    flx(m,IDN,k,j,i) = 0.5 * (fl.d  + fr.d  - lambda * (consr.d  - consl.d ));
    flx(m,IEN,k,j,i) = 0.5 * (fl.e  + fr.e  - lambda * (consr.e  - consl.e ));
    flx(m,ivx,k,j,i) = 0.5 * (fl.mx + fr.mx - lambda * (consr.mx - consl.mx));
    flx(m,ivy,k,j,i) = 0.5 * (fl.my + fr.my - lambda * (consr.my - consl.my));
    flx(m,ivz,k,j,i) = 0.5 * (fl.mz + fr.mz - lambda * (consr.mz - consl.mz));

    ey(m,k,j,i) = -0.5 * (fl.by + fr.by - lambda * (consr.by - consl.by));
    ez(m,k,j,i) =  0.5 * (fl.bz + fr.bz - lambda * (consr.bz - consl.bz));

    // We evolve tau = E - D
    flx(m,IEN,k,j,i) -= flx(m,IDN,k,j,i);
  });

  return;
}

} // namespace mhd
