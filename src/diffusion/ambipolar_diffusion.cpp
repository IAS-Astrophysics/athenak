//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ambipolar_diffusion.cpp
//  \brief Implements functions for AmbipolarDiffusion class.
//  The ambipolar EMF is: E_amb = eta_ad * [B^2 * J - (J.B) * B]
//  Following the algorithm in Athena++ src/field/field_diffusion/diffusivity.cpp

#include <float.h>
#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "ambipolar_diffusion.hpp"
#include "current_density.hpp"

//----------------------------------------------------------------------------------------
// constructor

AmbipolarDiffusion::AmbipolarDiffusion(MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  eta_ad = pin->GetReal("mhd","eta_ad");
}

//----------------------------------------------------------------------------------------
// destructor

AmbipolarDiffusion::~AmbipolarDiffusion() {
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::AddAmbipolarEMFs()
//! \brief Wrapper function that dispatches to type-specific ambipolar EMF implementation.

void AmbipolarDiffusion::AddAmbipolarEMFs(const DvceFaceFld4D<Real> &b0,
    const DvceArray5D<Real> &bcc0, DvceEdgeFld4D<Real> &efld) {
  AddEMFConstantAmbipolar(b0, bcc0, efld);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::AddAmbipolarFluxes()
//! \brief Wrapper function that dispatches to type-specific ambipolar energy flux.

void AmbipolarDiffusion::AddAmbipolarFluxes(const DvceFaceFld4D<Real> &b0,
    const DvceArray5D<Real> &bcc0, DvceFaceFld5D<Real> &flx) {
  AddFluxConstantAmbipolar(b0, bcc0, flx);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::AddEMFConstantAmbipolar()
//  \brief Adds ambipolar EMF to edge-centered electric field.
//  E_amb = eta_ad * [ B^2 * J  -  (J . B) * B ]
//  where J = curl(B) at edges and B is interpolated to edges from face/cell centers.
//  This is equivalent to eta_A * [J - (J.B)B/B^2] with eta_A = eta_ad * B^2.
//
//  Algorithm follows Athena++ AmbipolarEMF in field_diffusion/diffusivity.cpp.

void AmbipolarDiffusion::AddEMFConstantAmbipolar(const DvceFaceFld4D<Real> &b0,
    const DvceArray5D<Real> &bcc0, DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  //---- 1-D problem:
  // Edge e2 and e3 are at the same location as face x1f
  // B at edges: Bx from face x1f, By from bcc0 average, Bz from bcc0 average
  if (pmy_pack->pmesh->one_d) {
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta = eta_ad;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("amb1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, js, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      par_for_inner(member, is, ie+1, [&](const int i) {
        // In 1D, edges are co-located with x1-faces.
        // Interpolate B to edge: Bx = face value, By/Bz = average of adjacent cells
        Real intBx = b0.x1f(m,ks,js,i);
        Real intBy = 0.5*(bcc0(m,IBY,ks,js,i) + bcc0(m,IBY,ks,js,i-1));
        Real intBz = 0.5*(bcc0(m,IBZ,ks,js,i) + bcc0(m,IBZ,ks,js,i-1));

        Real Bsq = SQR(intBx) + SQR(intBy) + SQR(intBz);
        Real JdotB = j1(i)*intBx + j2(i)*intBy + j3(i)*intBz;

        // E_amb = eta_ad * (B^2 * J - (J.B) * B)
        Real e2_amb = eta * (Bsq*j2(i) - JdotB*intBy);
        Real e3_amb = eta * (Bsq*j3(i) - JdotB*intBz);

        e2(m,ks,  js  ,i) += e2_amb;
        e2(m,ke+1,js  ,i) += e2_amb;
        e3(m,ks  ,js  ,i) += e3_amb;
        e3(m,ks  ,je+1,i) += e3_amb;
      });
    });
    return;
  }

  //---- 2-D problem:
  // e1 edges at (i, j+1/2): Bx from bcc0 avg over j,j-1; By from x2f; Bz from bcc0 avg
  // e2 edges at (i+1/2, j): same as 1D
  // e3 edges at (i+1/2, j+1/2): all from averages
  if (pmy_pack->pmesh->two_d) {
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta = eta_ad;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("amb2", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      par_for_inner(member, is, ie+1, [&](const int i) {
        // --- e1 edge at (i, j+1/2, ks) ---
        // Bx: average bcc0 over j and j-1
        Real intBx_e1 = 0.5*(bcc0(m,IBX,ks,j,i) + bcc0(m,IBX,ks,j-1,i));
        // By: face x2f value at this edge
        Real intBy_e1 = b0.x2f(m,ks,j,i);
        // Bz: average bcc0 over j and j-1
        Real intBz_e1 = 0.5*(bcc0(m,IBZ,ks,j,i) + bcc0(m,IBZ,ks,j-1,i));

        Real Bsq_e1 = SQR(intBx_e1) + SQR(intBy_e1) + SQR(intBz_e1);
        Real JdotB_e1 = j1(i)*intBx_e1 + j2(i)*intBy_e1 + j3(i)*intBz_e1;
        Real e1_amb = eta * (Bsq_e1*j1(i) - JdotB_e1*intBx_e1);
        e1(m,ks,  j,i) += e1_amb;
        e1(m,ke+1,j,i) += e1_amb;

        // --- e2 edge at (i+1/2, j, ks) ---
        Real intBx_e2 = b0.x1f(m,ks,j,i);
        Real intBy_e2 = 0.5*(bcc0(m,IBY,ks,j,i) + bcc0(m,IBY,ks,j,i-1));
        Real intBz_e2 = 0.5*(bcc0(m,IBZ,ks,j,i) + bcc0(m,IBZ,ks,j,i-1));

        Real Bsq_e2 = SQR(intBx_e2) + SQR(intBy_e2) + SQR(intBz_e2);
        Real JdotB_e2 = j1(i)*intBx_e2 + j2(i)*intBy_e2 + j3(i)*intBz_e2;
        Real e2_amb = eta * (Bsq_e2*j2(i) - JdotB_e2*intBy_e2);
        e2(m,ks,  j,i) += e2_amb;
        e2(m,ke+1,j,i) += e2_amb;

        // --- e3 edge at (i+1/2, j+1/2, ks) ---
        Real intBx_e3 = 0.25*(bcc0(m,IBX,ks,j,i) + bcc0(m,IBX,ks,j-1,i)
                             + bcc0(m,IBX,ks,j,i-1) + bcc0(m,IBX,ks,j-1,i-1));
        Real intBy_e3 = 0.25*(bcc0(m,IBY,ks,j,i) + bcc0(m,IBY,ks,j-1,i)
                             + bcc0(m,IBY,ks,j,i-1) + bcc0(m,IBY,ks,j-1,i-1));
        Real intBz_e3 = 0.25*(bcc0(m,IBZ,ks,j,i) + bcc0(m,IBZ,ks,j-1,i)
                             + bcc0(m,IBZ,ks,j,i-1) + bcc0(m,IBZ,ks,j-1,i-1));

        Real Bsq_e3 = SQR(intBx_e3) + SQR(intBy_e3) + SQR(intBz_e3);
        Real JdotB_e3 = j1(i)*intBx_e3 + j2(i)*intBy_e3 + j3(i)*intBz_e3;
        Real e3_amb = eta * (Bsq_e3*j3(i) - JdotB_e3*intBz_e3);
        e3(m,ks,j,i) += e3_amb;
      });
    });
    return;
  }

  //---- 3-D problem:
  // e1 at (i, j+1/2, k+1/2): Bx from 4-cell avg of bcc0; By from 2-face avg of x2f;
  //                           Bz from 2-face avg of x3f
  // e2 at (i+1/2, j, k+1/2): By from 4-cell avg of bcc0; Bx from 2-face avg of x1f;
  //                           Bz from 2-face avg of x3f
  // e3 at (i+1/2, j+1/2, k): Bz from 4-cell avg of bcc0; Bx from 2-face avg of x1f;
  //                           By from 2-face avg of x2f

  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto eta = eta_ad;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  par_for_outer("amb3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

    CurrentDensity(member, m, k, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

    par_for_inner(member, is, ie+1, [&](const int i) {
      // --- e1 at (i, j+1/2, k+1/2) ---
      // Bx: 4-cell average from bcc0 (cells sharing this edge)
      Real intBx_e1 = 0.25*(bcc0(m,IBX,k,j,i) + bcc0(m,IBX,k-1,j,i)
                           + bcc0(m,IBX,k,j-1,i) + bcc0(m,IBX,k-1,j-1,i));
      // By: average of two x2f faces sharing this edge
      Real intBy_e1 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k-1,j,i));
      // Bz: average of two x3f faces sharing this edge
      Real intBz_e1 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j-1,i));

      Real Bsq_e1 = SQR(intBx_e1) + SQR(intBy_e1) + SQR(intBz_e1);
      Real JdotB_e1 = j1(i)*intBx_e1 + j2(i)*intBy_e1 + j3(i)*intBz_e1;
      e1(m,k,j,i) += eta * (Bsq_e1*j1(i) - JdotB_e1*intBx_e1);

      // --- e2 at (i+1/2, j, k+1/2) ---
      // Bx: average of two x1f faces sharing this edge
      Real intBx_e2 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k-1,j,i));
      // By: 4-cell average from bcc0
      Real intBy_e2 = 0.25*(bcc0(m,IBY,k,j,i) + bcc0(m,IBY,k-1,j,i)
                           + bcc0(m,IBY,k,j,i-1) + bcc0(m,IBY,k-1,j,i-1));
      // Bz: average of two x3f faces sharing this edge
      Real intBz_e2 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j,i-1));

      Real Bsq_e2 = SQR(intBx_e2) + SQR(intBy_e2) + SQR(intBz_e2);
      Real JdotB_e2 = j1(i)*intBx_e2 + j2(i)*intBy_e2 + j3(i)*intBz_e2;
      e2(m,k,j,i) += eta * (Bsq_e2*j2(i) - JdotB_e2*intBy_e2);

      // --- e3 at (i+1/2, j+1/2, k) ---
      // Bx: average of two x1f faces sharing this edge
      Real intBx_e3 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j-1,i));
      // By: average of two x2f faces sharing this edge
      Real intBy_e3 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j,i-1));
      // Bz: 4-cell average from bcc0
      Real intBz_e3 = 0.25*(bcc0(m,IBZ,k,j,i) + bcc0(m,IBZ,k,j-1,i)
                           + bcc0(m,IBZ,k,j,i-1) + bcc0(m,IBZ,k,j-1,i-1));

      Real Bsq_e3 = SQR(intBx_e3) + SQR(intBy_e3) + SQR(intBz_e3);
      Real JdotB_e3 = j1(i)*intBx_e3 + j2(i)*intBy_e3 + j3(i)*intBz_e3;
      e3(m,k,j,i) += eta * (Bsq_e3*j3(i) - JdotB_e3*intBz_e3);
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::AddFluxConstantAmbipolar()
//  \brief Adds Poynting flux from ambipolar diffusion to energy flux.
//  The energy flux is S = E_amb x B, where E_amb = eta_ad * [B^2*J - (J.B)*B].
//  Following the structure of Resistivity::AddFluxConstantResist but with
//  the ambipolar EMF replacing eta*J.

void AmbipolarDiffusion::AddFluxConstantAmbipolar(const DvceFaceFld4D<Real> &b,
    const DvceArray5D<Real> &bcc0, DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto eta = eta_ad;

  //------------------------------
  // energy fluxes in x1-direction: S_1 = E2_amb*B3 - E3_amb*B2
  // Compute J and B at x1-face, then compute ambipolar EMF components E2, E3
  auto &flx1 = flx.x1f;
  par_for("amb_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // J at x1 face: compute from finite differences (same as Ohmic energy flux)
    Real j2k   = -(b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1;
    Real j2kp1 = -(b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j,i-1))/size.d_view(m).dx1;

    Real j3j   = (b.x2f(m,k,j  ,i) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1;
    Real j3jp1 = (b.x2f(m,k,j+1,i) - b.x2f(m,k,j+1,i-1))/size.d_view(m).dx1;

    if (multi_d) {
      j3j   -= (b.x1f(m,k,j  ,i) - b.x1f(m,k,j-1,i))/size.d_view(m).dx2;
      j3jp1 -= (b.x1f(m,k,j+1,i) - b.x1f(m,k,j  ,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      j2k   += (b.x1f(m,k  ,j,i) - b.x1f(m,k-1,j,i))/size.d_view(m).dx3;
      j2kp1 += (b.x1f(m,k+1,j,i) - b.x1f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    // Average J to face center
    Real j2_fc = 0.5*(j2k + j2kp1);
    Real j3_fc = 0.5*(j3j + j3jp1);
    // j1 at x1-face is not needed for the Poynting flux in x1 direction,
    // but is needed for the J.B projection
    // j1 = dB3/dx2 - dB2/dx3
    Real j1_fc = 0.0;
    if (multi_d) {
      j1_fc += 0.5*((b.x3f(m,k,j  ,i) - b.x3f(m,k,j-1  ,i))/size.d_view(m).dx2
                   + (b.x3f(m,k,j+1,i) - b.x3f(m,k,j    ,i))/size.d_view(m).dx2);
    }
    if (three_d) {
      j1_fc -= 0.5*((b.x2f(m,k  ,j,i) - b.x2f(m,k-1,j,i))/size.d_view(m).dx3
                   + (b.x2f(m,k+1,j,i) - b.x2f(m,k  ,j,i))/size.d_view(m).dx3);
    }

    // B at face center
    Real Bx_fc = b.x1f(m,k,j,i);
    Real By_fc = 0.5*(bcc0(m,IBY,k,j,i) + bcc0(m,IBY,k,j,i-1));
    Real Bz_fc = 0.5*(bcc0(m,IBZ,k,j,i) + bcc0(m,IBZ,k,j,i-1));

    Real Bsq = SQR(Bx_fc) + SQR(By_fc) + SQR(Bz_fc);
    Real JdotB = j1_fc*Bx_fc + j2_fc*By_fc + j3_fc*Bz_fc;

    // E_amb components perpendicular to x1
    Real E2_amb = eta * (Bsq*j2_fc - JdotB*By_fc);
    Real E3_amb = eta * (Bsq*j3_fc - JdotB*Bz_fc);

    // S_1 = E2*B3 - E3*B2
    flx1(m,IEN,k,j,i) += E2_amb*Bz_fc - E3_amb*By_fc;
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //------------------------------
  // energy fluxes in x2-direction: S_2 = E3_amb*B1 - E1_amb*B3
  auto &flx2 = flx.x2f;
  par_for("amb_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // J at x2 face
    Real j1k   = (b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2;
    Real j1kp1 = (b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j-1,i))/size.d_view(m).dx2;

    Real j3i   = (b.x2f(m,k,j,i  ) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i  ) - b.x1f(m,k,j-1,i  ))/size.d_view(m).dx2;
    Real j3ip1 = (b.x2f(m,k,j,i+1) - b.x2f(m,k,j  ,i  ))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i+1) - b.x1f(m,k,j-1,i+1))/size.d_view(m).dx2;

    if (three_d) {
      j1k   -= (b.x2f(m,k  ,j,i) - b.x2f(m,k-1,j,i))/size.d_view(m).dx3;
      j1kp1 -= (b.x2f(m,k+1,j,i) - b.x2f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    Real j1_fc = 0.5*(j1k + j1kp1);
    Real j3_fc = 0.5*(j3i + j3ip1);
    // j2 at x2-face for J.B
    Real j2_fc = 0.0;
    // j2 = dB1/dx3 - dB3/dx1
    j2_fc -= 0.5*((b.x3f(m,k,j,i  ) - b.x3f(m,k,j,i-1))/size.d_view(m).dx1
                 + (b.x3f(m,k,j,i+1) - b.x3f(m,k,j,i  ))/size.d_view(m).dx1);
    if (three_d) {
      j2_fc += 0.5*((b.x1f(m,k  ,j,i) - b.x1f(m,k-1,j,i))/size.d_view(m).dx3
                   + (b.x1f(m,k+1,j,i) - b.x1f(m,k  ,j,i))/size.d_view(m).dx3);
    }

    // B at face center
    Real Bx_fc = 0.5*(bcc0(m,IBX,k,j,i) + bcc0(m,IBX,k,j-1,i));
    Real By_fc = b.x2f(m,k,j,i);
    Real Bz_fc = 0.5*(bcc0(m,IBZ,k,j,i) + bcc0(m,IBZ,k,j-1,i));

    Real Bsq = SQR(Bx_fc) + SQR(By_fc) + SQR(Bz_fc);
    Real JdotB = j1_fc*Bx_fc + j2_fc*By_fc + j3_fc*Bz_fc;

    Real E1_amb = eta * (Bsq*j1_fc - JdotB*Bx_fc);
    Real E3_amb = eta * (Bsq*j3_fc - JdotB*Bz_fc);

    // S_2 = E3*B1 - E1*B3
    flx2(m,IEN,k,j,i) += E3_amb*Bx_fc - E1_amb*Bz_fc;
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //------------------------------
  // energy fluxes in x3-direction: S_3 = E1_amb*B2 - E2_amb*B1
  auto &flx3 = flx.x3f;
  par_for("amb_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // J at x3 face
    Real j1j   = (b.x3f(m,k,j  ,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j  ,i) - b.x2f(m,k-1,j  ,i))/size.d_view(m).dx3;
    Real j1jp1 = (b.x3f(m,k,j+1,i) - b.x3f(m,k  ,j  ,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j+1,i) - b.x2f(m,k-1,j+1,i))/size.d_view(m).dx3;

    Real j2i   = -(b.x3f(m,k,j,i  ) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i  ) - b.x1f(m,k-1,j,i  ))/size.d_view(m).dx3;
    Real j2ip1 = -(b.x3f(m,k,j,i+1) - b.x3f(m,k  ,j,i  ))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i+1) - b.x1f(m,k-1,j,i+1))/size.d_view(m).dx3;

    Real j1_fc = 0.5*(j1j + j1jp1);
    Real j2_fc = 0.5*(j2i + j2ip1);
    // j3 at x3-face for J.B
    Real j3_fc = 0.0;
    // j3 = dB2/dx1 - dB1/dx2
    j3_fc += 0.5*((b.x2f(m,k,j,i  ) - b.x2f(m,k,j,i-1))/size.d_view(m).dx1
                 + (b.x2f(m,k,j,i+1) - b.x2f(m,k,j,i  ))/size.d_view(m).dx1);
    j3_fc -= 0.5*((b.x1f(m,k,j  ,i) - b.x1f(m,k,j-1,i))/size.d_view(m).dx2
                 + (b.x1f(m,k,j+1,i) - b.x1f(m,k,j  ,i))/size.d_view(m).dx2);

    // B at face center
    Real Bx_fc = 0.5*(bcc0(m,IBX,k,j,i) + bcc0(m,IBX,k-1,j,i));
    Real By_fc = 0.5*(bcc0(m,IBY,k,j,i) + bcc0(m,IBY,k-1,j,i));
    Real Bz_fc = b.x3f(m,k,j,i);

    Real Bsq = SQR(Bx_fc) + SQR(By_fc) + SQR(Bz_fc);
    Real JdotB = j1_fc*Bx_fc + j2_fc*By_fc + j3_fc*Bz_fc;

    Real E1_amb = eta * (Bsq*j1_fc - JdotB*Bx_fc);
    Real E2_amb = eta * (Bsq*j2_fc - JdotB*By_fc);

    // S_3 = E1*B2 - E2*B1
    flx3(m,IEN,k,j,i) += E1_amb*By_fc - E2_amb*Bx_fc;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AmbipolarDiffusion::NewTimeStep()
//! \brief Compute new time step for ambipolar diffusion.
//! dt_amb <= fac * dx^2 / (eta_ad * B_max^2)
//! Unlike Ohmic (constant eta_ohm), the effective diffusivity eta_ad*B^2 varies in space.

void AmbipolarDiffusion::NewTimeStep(const DvceArray5D<Real> &w0,
                                     const EOS_Data &eos_data) {
  dtnew = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mb_size;
  auto &bcc0 = pmy_pack->pmhd->bcc0;

  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto eta = eta_ad;

  // Find max B^2 across all cells and meshblocks
  Real max_bsq = 0.0;
  Kokkos::parallel_reduce("amb_maxbsq",
    Kokkos::RangePolicy<>(DevExeSpace(), 0,
      (nmb1+1)*(ke-ks+1)*(je-js+1)*(ie-is+1)),
  KOKKOS_LAMBDA(const int &idx, Real &lmax) {
    int nx1 = ie - is + 1;
    int nx2 = je - js + 1;
    int nx3 = ke - ks + 1;
    int nkji = nx3*nx2*nx1;
    int nji = nx2*nx1;
    int m = idx / nkji;
    int k = (idx - m*nkji) / nji + ks;
    int j = (idx - m*nkji - (k-ks)*nji) / nx1 + js;
    int i = (idx - m*nkji - (k-ks)*nji - (j-js)*nx1) + is;
    Real bsq = SQR(bcc0(m,IBX,k,j,i)) + SQR(bcc0(m,IBY,k,j,i))
             + SQR(bcc0(m,IBZ,k,j,i));
    lmax = fmax(lmax, bsq);
  }, Kokkos::Max<Real>(max_bsq));

  Real eta_eff = eta * max_bsq;
  if (eta_eff < 1.0e-30) {
    return;  // B~0 everywhere, no constraint
  }

  for (int m=0; m<=(nmb1); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/eta_eff);
    if (pmy_pack->pmesh->multi_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/eta_eff);
    }
    if (pmy_pack->pmesh->three_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/eta_eff);
    }
  }
  return;
}
