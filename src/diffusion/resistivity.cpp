//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.cpp
//  \brief Implements functions for Resistivity class.

#include <float.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string> // string

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "resistivity.hpp"
#include "current_density.hpp"

//----------------------------------------------------------------------------------------
// Inline helpers: single components of the edge-centered current density J = curl(B).
// These reproduce the finite-difference stencils used by CurrentDensity() (see
// current_density.hpp) but return one component at an arbitrary (m,k,j,i) so that the
// ambipolar EMF/flux kernels can recompute J inline at the neighbouring edges required
// for cross-component averaging.
//   J1 at x1-edge (k-1/2, j-1/2, i),  J2 at x2-edge (k-1/2, j, i-1/2),
//   J3 at x3-edge (k, j-1/2, i-1/2).
// multi_d adds the d/dx2 terms (2D and 3D); three_d adds the d/dx3 terms (3D only).

KOKKOS_INLINE_FUNCTION
Real EdgeJ1(const int m, const int k, const int j, const int i,
            const DvceFaceFld4D<Real> &b, const RegionSize &size,
            const bool multi_d, const bool three_d) {
  Real j1 = 0.0;
  if (multi_d) j1 += (b.x3f(m,k,j,i) - b.x3f(m,k,j-1,i))/size.dx2;
  if (three_d) j1 -= (b.x2f(m,k,j,i) - b.x2f(m,k-1,j,i))/size.dx3;
  return j1;
}

KOKKOS_INLINE_FUNCTION
Real EdgeJ2(const int m, const int k, const int j, const int i,
            const DvceFaceFld4D<Real> &b, const RegionSize &size,
            const bool three_d) {
  Real j2 = -(b.x3f(m,k,j,i) - b.x3f(m,k,j,i-1))/size.dx1;
  if (three_d) j2 += (b.x1f(m,k,j,i) - b.x1f(m,k-1,j,i))/size.dx3;
  return j2;
}

KOKKOS_INLINE_FUNCTION
Real EdgeJ3(const int m, const int k, const int j, const int i,
            const DvceFaceFld4D<Real> &b, const RegionSize &size,
            const bool multi_d) {
  Real j3 = (b.x2f(m,k,j,i) - b.x2f(m,k,j,i-1))/size.dx1;
  if (multi_d) j3 -= (b.x1f(m,k,j,i) - b.x1f(m,k,j-1,i))/size.dx2;
  return j3;
}

//----------------------------------------------------------------------------------------
// ctor: also calls Resistivity base class constructor

Resistivity::Resistivity(MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  // Read non-ideal MHD coefficients (if any). A non-zero value enables the term.
  eta_ohm = pin->GetOrAddReal("mhd","eta_ohm",0.0);
  eta_ad  = pin->GetOrAddReal("mhd","eta_ad",0.0);
}

//----------------------------------------------------------------------------------------
// Resistivity destructor

Resistivity::~Resistivity() {
}

//----------------------------------------------------------------------------------------
//! \fn void AddResistiveEMFs()
//! \brief Wrapper function that adds non-ideal electric fields to the corner-centered
//! EMF. Adds the Ohmic contribution if eta_ohm != 0 and the ambipolar contribution if
//! eta_ad != 0. Both use constant coefficients.

void Resistivity::AddResistiveEMFs(const DvceFaceFld4D<Real> &b0,
    DvceEdgeFld4D<Real> &efld) {
  if (eta_ohm != 0.0) {
    AddEMFConstantResist(b0, efld);
  }
  if (eta_ad != 0.0) {
    AddEMFConstantAmbipolar(b0, efld);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddResistiveFluxes()
//! \brief Wrapper function that adds non-ideal energy (Poynting) fluxes to the energy
//! flux. Adds the Ohmic contribution if eta_ohm != 0 and the ambipolar contribution if
//! eta_ad != 0. Both use constant coefficients.

void Resistivity::AddResistiveFluxes(const DvceFaceFld4D<Real> &b0,
    DvceFaceFld5D<Real> &flx) {
  if (eta_ohm != 0.0) {
    AddFluxConstantResist(b0, flx);
  }
  if (eta_ad != 0.0) {
    AddFluxConstantAmbipolar(b0, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddEMFConstantResist()
//  \brief Adds electric field from Ohmic resistivity to corner-centered electric field
//  Using Ohm's Law to compute the electric field:  E + (v x B) = \eta J, then
//    E_{inductive} = - (v x B)  [computed in the MHD Riemann solver]
//    E_{resistive} = \eta J     [computed in this function]

void Resistivity::AddEMFConstantResist(const DvceFaceFld4D<Real> &b0,
    DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  //---- 1-D problem:
  //  copy face-centered E-fields to edges and return.
  //  Note e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]

  if (pmy_pack->pmesh->one_d) {
    // capture class variables for the kernels
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, js, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        e2(m,ks,  js  ,i) += eta_o*j2(i);
        e2(m,ke+1,js  ,i) += eta_o*j2(i);
        e3(m,ks  ,js  ,i) += eta_o*j3(i);
        e3(m,ks  ,je+1,i) += eta_o*j3(i);
      });
    });
    return;
  }

  //---- 2-D problem:
  if (pmy_pack->pmesh->two_d) {
    // capture class variables for the kernels
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm2", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        e1(m,ks,  j,i) += eta_o*j1(i);
        e1(m,ke+1,j,i) += eta_o*j1(i);
        e2(m,ks,  j,i) += eta_o*j2(i);
        e2(m,ke+1,j,i) += eta_o*j2(i);
        e3(m,ks  ,j,i) += eta_o*j3(i);
      });
    });
    return;
  }

  //---- 3-D problem:

  // capture class variables for the kernels
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto eta_o = eta_ohm;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  par_for_outer("ohm3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

    CurrentDensity(member, m, k, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

    // Add E_{resistive} = \eta J to corner-centered electric fields
    par_for_inner(member, is, ie+1, [&](const int i) {
      e1(m,k,j,i) += eta_o*j1(i);
      e2(m,k,j,i) += eta_o*j2(i);
      e3(m,k,j,i) += eta_o*j3(i);
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddResistiveFluxConstantResist()
//  \brief Adds Poynting flux from Ohmic resistivity to energy flux
//  Total energy equation is dE/dt = - Div(F) where F = (E X B) = \eta (J X B)

void Resistivity::AddFluxConstantResist(const DvceFaceFld4D<Real> &b,
                                        DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real qa = 0.25*eta_ohm;

  //------------------------------
  // energy fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("ohm_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

    // flx1 = (E X B)_{1} =  ((\eta J) X B)_{1} = \eta (J2*B3 - J3*B2)
    flx1(m,IEN,k,j,i) += qa*(j2k  *(b.x3f(m,k  ,j  ,i) + b.x3f(m,k  ,j  ,i-1)) +
                             j2kp1*(b.x3f(m,k+1,j  ,i) + b.x3f(m,k+1,j  ,i-1)) -
                             j3j  *(b.x2f(m,k  ,j  ,i) + b.x2f(m,k  ,j  ,i-1)) -
                             j3jp1*(b.x2f(m,k  ,j+1,i) + b.x2f(m,k  ,j+1,i-1)));
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //------------------------------
  // energy fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("ohm_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

    // E2 = \eta (J X B)_{2} = \eta (J3*B1 - J1*B3)
    flx2(m,IEN,k,j,i) += qa*(j3i  *(b.x1f(m,k  ,j,i  ) + b.x1f(m,k  ,j-1,i  )) +
                             j3ip1*(b.x1f(m,k  ,j,i+1) + b.x1f(m,k  ,j-1,i+1)) -
                             j1k  *(b.x3f(m,k  ,j,i  ) + b.x3f(m,k  ,j-1,i  )) -
                             j1kp1*(b.x3f(m,k+1,j,i  ) + b.x3f(m,k+1,j-1,i  )));
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //------------------------------
  // energy fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("ohm_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1j   = (b.x3f(m,k,j  ,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j  ,i) - b.x2f(m,k-1,j  ,i))/size.d_view(m).dx3;
    Real j1jp1 = (b.x3f(m,k,j+1,i) - b.x3f(m,k  ,j  ,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j+1,i) - b.x2f(m,k-1,j+1,i))/size.d_view(m).dx3;

    Real j2i   = -(b.x3f(m,k,j,i  ) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i  ) - b.x1f(m,k-1,j,i  ))/size.d_view(m).dx3;
    Real j2ip1 = -(b.x3f(m,k,j,i+1) - b.x3f(m,k  ,j,i  ))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i+1) - b.x1f(m,k-1,j,i+1))/size.d_view(m).dx3;

    // E2 = \eta (J X B)_{2} = \eta (J1*B2 - J2*B1)
    flx3(m,IEN,k,j,i) += qa*(j1j  *(b.x2f(m,k,j  ,i  ) + b.x2f(m,k-1,j  ,i  )) +
                             j1jp1*(b.x2f(m,k,j+1,i  ) + b.x2f(m,k-1,j+1,i  )) -
                             j2i  *(b.x1f(m,k,j  ,i  ) + b.x1f(m,k-1,j  ,i  )) -
                             j2ip1*(b.x1f(m,k,j  ,i+1) + b.x1f(m,k-1,j  ,i+1)));
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddEMFConstantAmbipolar()
//  \brief Adds ambipolar EMF to the corner-centered electric field.
//    E_amb = eta_ad * [ B^2 * J - (J . B) * B ]
//  This is the expanded form of eta_A*[J - (J.B)B/B^2] with eta_A = eta_ad*B^2; the
//  expanded form avoids any division by B^2 (no floor needed). The algorithm follows
//  Athena++ AmbipolarEMF (field_diffusion/diffusivity.cpp): J = curl(B) at edges, and
//  the cross-J components are averaged to the target edge (4-pt in 3D, 2-pt across a
//  degenerate direction in 2D). Here J is recomputed inline from b0 via EdgeJ{1,2,3}();
//  cell-centered B (bcc0) is obtained from the parent MHD class and used for the
//  edge-diagonal B interpolation, exactly as the (now-retired) standalone module did.

void Resistivity::AddEMFConstantAmbipolar(const DvceFaceFld4D<Real> &b0,
    DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &bcc0 = pmy_pack->pmhd->bcc0;
  auto size = pmy_pack->pmb->mb_size;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  auto eta = eta_ad;

  //---- 1-D problem:
  // All edges co-located with x1-faces. No cross-component averaging needed.
  // J1 = 0 in 1D (no curl contributions from degenerate directions).
  if (pmy_pack->pmesh->one_d) {
    par_for("amb_emf1", DevExeSpace(), 0, nmb1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int i) {
      Real intBx = b0.x1f(m,ks,js,i);
      Real intBy = 0.5*(bcc0(m,IBY,ks,js,i) + bcc0(m,IBY,ks,js,i-1));
      Real intBz = 0.5*(bcc0(m,IBZ,ks,js,i) + bcc0(m,IBZ,ks,js,i-1));

      Real intJ2 = EdgeJ2(m,ks,js,i,b0,size.d_view(m),three_d);
      Real intJ3 = EdgeJ3(m,ks,js,i,b0,size.d_view(m),multi_d);

      Real Bsq = SQR(intBx) + SQR(intBy) + SQR(intBz);
      Real JdotB = intJ2*intBy + intJ3*intBz;

      Real e2_amb = eta * (Bsq*intJ2 - JdotB*intBy);
      Real e3_amb = eta * (Bsq*intJ3 - JdotB*intBz);

      e2(m,ks,  js  ,i) += e2_amb;
      e2(m,ke+1,js  ,i) += e2_amb;
      e3(m,ks  ,js  ,i) += e3_amb;
      e3(m,ks  ,je+1,i) += e3_amb;
    });
    return;
  }

  //---- 2-D problem:
  // Cross-component J averaging needed in x1-x2 plane; k direction is degenerate.
  if (pmy_pack->pmesh->two_d) {
    par_for("amb_emf2", DevExeSpace(), 0, nmb1, js, je+1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      // --- e1 at (cell i, face j) ---
      Real intJ1_e1 = EdgeJ1(m,ks,j,i,b0,size.d_view(m),multi_d,three_d);
      Real intJ2_e1 = 0.25*(EdgeJ2(m,ks,j-1,i  ,b0,size.d_view(m),three_d)
                          + EdgeJ2(m,ks,j-1,i+1,b0,size.d_view(m),three_d)
                          + EdgeJ2(m,ks,j,  i  ,b0,size.d_view(m),three_d)
                          + EdgeJ2(m,ks,j,  i+1,b0,size.d_view(m),three_d));
      Real intJ3_e1 = 0.5*(EdgeJ3(m,ks,j,i  ,b0,size.d_view(m),multi_d)
                         + EdgeJ3(m,ks,j,i+1,b0,size.d_view(m),multi_d));

      Real intBx_e1 = 0.5*(bcc0(m,IBX,ks,j,i) + bcc0(m,IBX,ks,j-1,i));
      Real intBy_e1 = b0.x2f(m,ks,j,i);
      Real intBz_e1 = 0.5*(bcc0(m,IBZ,ks,j,i) + bcc0(m,IBZ,ks,j-1,i));

      Real Bsq_e1 = SQR(intBx_e1) + SQR(intBy_e1) + SQR(intBz_e1);
      Real JdotB_e1 = intJ1_e1*intBx_e1 + intJ2_e1*intBy_e1 + intJ3_e1*intBz_e1;
      Real e1_amb = eta * (Bsq_e1*intJ1_e1 - JdotB_e1*intBx_e1);
      e1(m,ks,  j,i) += e1_amb;
      e1(m,ke+1,j,i) += e1_amb;

      // --- e2 at (face i, cell j) ---
      Real intJ1_e2 = 0.25*(EdgeJ1(m,ks,j,  i-1,b0,size.d_view(m),multi_d,three_d)
                          + EdgeJ1(m,ks,j,  i  ,b0,size.d_view(m),multi_d,three_d)
                          + EdgeJ1(m,ks,j+1,i-1,b0,size.d_view(m),multi_d,three_d)
                          + EdgeJ1(m,ks,j+1,i  ,b0,size.d_view(m),multi_d,three_d));
      Real intJ2_e2 = EdgeJ2(m,ks,j,i,b0,size.d_view(m),three_d);
      Real intJ3_e2 = 0.5*(EdgeJ3(m,ks,j,  i,b0,size.d_view(m),multi_d)
                         + EdgeJ3(m,ks,j+1,i,b0,size.d_view(m),multi_d));

      Real intBx_e2 = b0.x1f(m,ks,j,i);
      Real intBy_e2 = 0.5*(bcc0(m,IBY,ks,j,i) + bcc0(m,IBY,ks,j,i-1));
      Real intBz_e2 = 0.5*(bcc0(m,IBZ,ks,j,i) + bcc0(m,IBZ,ks,j,i-1));

      Real Bsq_e2 = SQR(intBx_e2) + SQR(intBy_e2) + SQR(intBz_e2);
      Real JdotB_e2 = intJ1_e2*intBx_e2 + intJ2_e2*intBy_e2 + intJ3_e2*intBz_e2;
      Real e2_amb = eta * (Bsq_e2*intJ2_e2 - JdotB_e2*intBy_e2);
      e2(m,ks,  j,i) += e2_amb;
      e2(m,ke+1,j,i) += e2_amb;

      // --- e3 at (face i, face j) ---
      Real intJ1_e3 = 0.5*(EdgeJ1(m,ks,j,i-1,b0,size.d_view(m),multi_d,three_d)
                         + EdgeJ1(m,ks,j,i  ,b0,size.d_view(m),multi_d,three_d));
      Real intJ2_e3 = 0.5*(EdgeJ2(m,ks,j-1,i,b0,size.d_view(m),three_d)
                         + EdgeJ2(m,ks,j,  i,b0,size.d_view(m),three_d));
      Real intJ3_e3 = EdgeJ3(m,ks,j,i,b0,size.d_view(m),multi_d);

      Real intBx_e3 = 0.5*(b0.x1f(m,ks,j,i) + b0.x1f(m,ks,j-1,i));
      Real intBy_e3 = 0.5*(b0.x2f(m,ks,j,i) + b0.x2f(m,ks,j,i-1));
      Real intBz_e3 = 0.25*(bcc0(m,IBZ,ks,j,i) + bcc0(m,IBZ,ks,j-1,i)
                           + bcc0(m,IBZ,ks,j,i-1) + bcc0(m,IBZ,ks,j-1,i-1));

      Real Bsq_e3 = SQR(intBx_e3) + SQR(intBy_e3) + SQR(intBz_e3);
      Real JdotB_e3 = intJ1_e3*intBx_e3 + intJ2_e3*intBy_e3 + intJ3_e3*intBz_e3;
      Real e3_amb = eta * (Bsq_e3*intJ3_e3 - JdotB_e3*intBz_e3);
      e3(m,ks,j,i) += e3_amb;
    });
    return;
  }

  //---- 3-D problem:
  // Cross-component J is 4-point averaged to correct edge positions (matching Athena++).
  // B interpolation: diagonal from 4-cell bcc0 avg, off-diagonal from 2-face avg.
  par_for("amb_emf3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je+1, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // --- e1 at x1-edge (cell i, face j, face k) ---
    Real intJ1_e1 = EdgeJ1(m,k,j,i,b0,size.d_view(m),multi_d,three_d);
    Real intJ2_e1 = 0.25*(EdgeJ2(m,k,j-1,i  ,b0,size.d_view(m),three_d)
                        + EdgeJ2(m,k,j-1,i+1,b0,size.d_view(m),three_d)
                        + EdgeJ2(m,k,j,  i  ,b0,size.d_view(m),three_d)
                        + EdgeJ2(m,k,j,  i+1,b0,size.d_view(m),three_d));
    Real intJ3_e1 = 0.25*(EdgeJ3(m,k-1,j,i  ,b0,size.d_view(m),multi_d)
                        + EdgeJ3(m,k-1,j,i+1,b0,size.d_view(m),multi_d)
                        + EdgeJ3(m,k,  j,i  ,b0,size.d_view(m),multi_d)
                        + EdgeJ3(m,k,  j,i+1,b0,size.d_view(m),multi_d));

    Real intBx_e1 = 0.25*(bcc0(m,IBX,k,j,i) + bcc0(m,IBX,k-1,j,i)
                         + bcc0(m,IBX,k,j-1,i) + bcc0(m,IBX,k-1,j-1,i));
    Real intBy_e1 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k-1,j,i));
    Real intBz_e1 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j-1,i));

    Real Bsq_e1 = SQR(intBx_e1) + SQR(intBy_e1) + SQR(intBz_e1);
    Real JdotB_e1 = intJ1_e1*intBx_e1 + intJ2_e1*intBy_e1 + intJ3_e1*intBz_e1;
    e1(m,k,j,i) += eta * (Bsq_e1*intJ1_e1 - JdotB_e1*intBx_e1);

    // --- e2 at x2-edge (face i, cell j, face k) ---
    Real intJ1_e2 = 0.25*(EdgeJ1(m,k,j,  i-1,b0,size.d_view(m),multi_d,three_d)
                        + EdgeJ1(m,k,j,  i  ,b0,size.d_view(m),multi_d,three_d)
                        + EdgeJ1(m,k,j+1,i-1,b0,size.d_view(m),multi_d,three_d)
                        + EdgeJ1(m,k,j+1,i  ,b0,size.d_view(m),multi_d,three_d));
    Real intJ2_e2 = EdgeJ2(m,k,j,i,b0,size.d_view(m),three_d);
    Real intJ3_e2 = 0.25*(EdgeJ3(m,k-1,j,  i,b0,size.d_view(m),multi_d)
                        + EdgeJ3(m,k-1,j+1,i,b0,size.d_view(m),multi_d)
                        + EdgeJ3(m,k,  j,  i,b0,size.d_view(m),multi_d)
                        + EdgeJ3(m,k,  j+1,i,b0,size.d_view(m),multi_d));

    Real intBx_e2 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k-1,j,i));
    Real intBy_e2 = 0.25*(bcc0(m,IBY,k,j,i) + bcc0(m,IBY,k-1,j,i)
                         + bcc0(m,IBY,k,j,i-1) + bcc0(m,IBY,k-1,j,i-1));
    Real intBz_e2 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j,i-1));

    Real Bsq_e2 = SQR(intBx_e2) + SQR(intBy_e2) + SQR(intBz_e2);
    Real JdotB_e2 = intJ1_e2*intBx_e2 + intJ2_e2*intBy_e2 + intJ3_e2*intBz_e2;
    e2(m,k,j,i) += eta * (Bsq_e2*intJ2_e2 - JdotB_e2*intBy_e2);

    // --- e3 at x3-edge (face i, face j, cell k) ---
    Real intJ1_e3 = 0.25*(EdgeJ1(m,k,  j,i-1,b0,size.d_view(m),multi_d,three_d)
                        + EdgeJ1(m,k,  j,i  ,b0,size.d_view(m),multi_d,three_d)
                        + EdgeJ1(m,k+1,j,i-1,b0,size.d_view(m),multi_d,three_d)
                        + EdgeJ1(m,k+1,j,i  ,b0,size.d_view(m),multi_d,three_d));
    Real intJ2_e3 = 0.25*(EdgeJ2(m,k,  j-1,i,b0,size.d_view(m),three_d)
                        + EdgeJ2(m,k,  j,  i,b0,size.d_view(m),three_d)
                        + EdgeJ2(m,k+1,j-1,i,b0,size.d_view(m),three_d)
                        + EdgeJ2(m,k+1,j,  i,b0,size.d_view(m),three_d));
    Real intJ3_e3 = EdgeJ3(m,k,j,i,b0,size.d_view(m),multi_d);

    Real intBx_e3 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j-1,i));
    Real intBy_e3 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j,i-1));
    Real intBz_e3 = 0.25*(bcc0(m,IBZ,k,j,i) + bcc0(m,IBZ,k,j-1,i)
                         + bcc0(m,IBZ,k,j,i-1) + bcc0(m,IBZ,k,j-1,i-1));

    Real Bsq_e3 = SQR(intBx_e3) + SQR(intBy_e3) + SQR(intBz_e3);
    Real JdotB_e3 = intJ1_e3*intBx_e3 + intJ2_e3*intBy_e3 + intJ3_e3*intBz_e3;
    e3(m,k,j,i) += eta * (Bsq_e3*intJ3_e3 - JdotB_e3*intBz_e3);
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddFluxConstantAmbipolar()
//  \brief Adds the ambipolar Poynting flux to the energy flux.
//  Since S_AD = E_AD x B and E_AD = eta_ad*(B^2*J - (J.B)*B), the field-parallel part
//  vanishes in the cross product (B x B = 0), leaving S = eta_ad*B^2*(J x B). The
//  simplified edge EMF e = eta_ad*B^2*J is formed at each edge using the same B
//  interpolation stencils as AddEMFConstantAmbipolar, averaged to the face, then crossed
//  with cell-centered B (bcc0) averaged to the face -- the same "average then multiply"
//  pattern as the Ohmic Poynting flux. J is recomputed inline via EdgeJ{1,2,3}().

void Resistivity::AddFluxConstantAmbipolar(const DvceFaceFld4D<Real> &b0,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &bcc = pmy_pack->pmhd->bcc0;
  auto size = pmy_pack->pmb->mb_size;
  bool multi_d = pmy_pack->pmesh->multi_d;
  bool three_d = pmy_pack->pmesh->three_d;
  auto eta = eta_ad;

  //---- 1-D problem:
  // All edges coincide with x1-faces. B^2 is the same at all edge positions.
  if (pmy_pack->pmesh->one_d) {
    auto &flx1 = flx.x1f;
    par_for("amb_heat1d", DevExeSpace(), 0, nmb1, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int i) {
      Real Bx = b0.x1f(m,ks,js,i);
      Real By = 0.5*(bcc(m,IBY,ks,js,i-1) + bcc(m,IBY,ks,js,i));
      Real Bz = 0.5*(bcc(m,IBZ,ks,js,i-1) + bcc(m,IBZ,ks,js,i));
      Real Bsq = SQR(Bx) + SQR(By) + SQR(Bz);

      Real e2_fc = eta * Bsq * EdgeJ2(m,ks,js,i,b0,size.d_view(m),three_d);
      Real e3_fc = eta * Bsq * EdgeJ3(m,ks,js,i,b0,size.d_view(m),multi_d);

      flx1(m,IEN,ks,js,i) += e2_fc*Bz - e3_fc*By;
    });
    return;
  }

  //---- 2-D problem:
  if (pmy_pack->pmesh->two_d) {
    // x1-flux: e2 at x2-edge (no k-avg), e3 at x3-edges (j) and (j+1)
    auto &flx1 = flx.x1f;
    par_for("amb_heat1_2d", DevExeSpace(), 0, nmb1, js, je, is, ie+1,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      // e2 at x2-edge (ks)
      Real Bx = b0.x1f(m,ks,j,i);
      Real By = 0.5*(bcc(m,IBY,ks,j,i-1) + bcc(m,IBY,ks,j,i));
      Real Bz = 0.5*(bcc(m,IBZ,ks,j,i-1) + bcc(m,IBZ,ks,j,i));
      Real e2_fc = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                 * EdgeJ2(m,ks,j,i,b0,size.d_view(m),three_d);

      // e3 at x3-edge (j)
      Bx = 0.5*(b0.x1f(m,ks,j,i) + b0.x1f(m,ks,j-1,i));
      By = 0.5*(b0.x2f(m,ks,j,i) + b0.x2f(m,ks,j,i-1));
      Bz = 0.25*(bcc(m,IBZ,ks,j,i) + bcc(m,IBZ,ks,j-1,i)
                + bcc(m,IBZ,ks,j,i-1) + bcc(m,IBZ,ks,j-1,i-1));
      Real e3_j = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ3(m,ks,j,i,b0,size.d_view(m),multi_d);

      // e3 at x3-edge (j+1)
      Bx = 0.5*(b0.x1f(m,ks,j+1,i) + b0.x1f(m,ks,j,i));
      By = 0.5*(b0.x2f(m,ks,j+1,i) + b0.x2f(m,ks,j+1,i-1));
      Bz = 0.25*(bcc(m,IBZ,ks,j+1,i) + bcc(m,IBZ,ks,j,i)
                + bcc(m,IBZ,ks,j+1,i-1) + bcc(m,IBZ,ks,j,i-1));
      Real e3_jp1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                  * EdgeJ3(m,ks,j+1,i,b0,size.d_view(m),multi_d);

      Real e3_fc = 0.5*(e3_j + e3_jp1);

      Real b2_fc = 0.5*(bcc(m,IBY,ks,j,i-1) + bcc(m,IBY,ks,j,i));
      Real b3_fc = 0.5*(bcc(m,IBZ,ks,j,i-1) + bcc(m,IBZ,ks,j,i));

      flx1(m,IEN,ks,j,i) += e2_fc*b3_fc - e3_fc*b2_fc;
    });

    // x2-flux: e3 at x3-edges (i) and (i+1), e1 at x1-edge (no k-avg)
    auto &flx2 = flx.x2f;
    par_for("amb_heat2_2d", DevExeSpace(), 0, nmb1, js, je+1, is, ie,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      // e3 at x3-edge (i)
      Real Bx = 0.5*(b0.x1f(m,ks,j,i) + b0.x1f(m,ks,j-1,i));
      Real By = 0.5*(b0.x2f(m,ks,j,i) + b0.x2f(m,ks,j,i-1));
      Real Bz = 0.25*(bcc(m,IBZ,ks,j,i) + bcc(m,IBZ,ks,j-1,i)
                     + bcc(m,IBZ,ks,j,i-1) + bcc(m,IBZ,ks,j-1,i-1));
      Real e3_i = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ3(m,ks,j,i,b0,size.d_view(m),multi_d);

      // e3 at x3-edge (i+1)
      Bx = 0.5*(b0.x1f(m,ks,j,i+1) + b0.x1f(m,ks,j-1,i+1));
      By = 0.5*(b0.x2f(m,ks,j,i+1) + b0.x2f(m,ks,j,i));
      Bz = 0.25*(bcc(m,IBZ,ks,j,i+1) + bcc(m,IBZ,ks,j-1,i+1)
                + bcc(m,IBZ,ks,j,i) + bcc(m,IBZ,ks,j-1,i));
      Real e3_ip1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                  * EdgeJ3(m,ks,j,i+1,b0,size.d_view(m),multi_d);

      Real e3_fc = 0.5*(e3_i + e3_ip1);

      // e1 at x1-edge (ks)
      Bx = 0.5*(bcc(m,IBX,ks,j,i) + bcc(m,IBX,ks,j-1,i));
      By = b0.x2f(m,ks,j,i);
      Bz = 0.5*(bcc(m,IBZ,ks,j,i) + bcc(m,IBZ,ks,j-1,i));
      Real e1_fc = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                 * EdgeJ1(m,ks,j,i,b0,size.d_view(m),multi_d,three_d);

      Real b1_fc = 0.5*(bcc(m,IBX,ks,j-1,i) + bcc(m,IBX,ks,j,i));
      Real b3_fc = 0.5*(bcc(m,IBZ,ks,j-1,i) + bcc(m,IBZ,ks,j,i));

      flx2(m,IEN,ks,j,i) += e3_fc*b1_fc - e1_fc*b3_fc;
    });
    return;
  }

  //---- 3-D problem:
  // x1-flux: e2 at x2-edges (k,k+1), e3 at x3-edges (j,j+1)
  auto &flx1 = flx.x1f;
  par_for("amb_heat1_3d", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // e2 at x2-edge (k)
    Real Bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k-1,j,i));
    Real By = 0.25*(bcc(m,IBY,k,j,i) + bcc(m,IBY,k-1,j,i)
                   + bcc(m,IBY,k,j,i-1) + bcc(m,IBY,k-1,j,i-1));
    Real Bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j,i-1));
    Real e2_k = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
              * EdgeJ2(m,k,j,i,b0,size.d_view(m),three_d);

    // e2 at x2-edge (k+1)
    Bx = 0.5*(b0.x1f(m,k+1,j,i) + b0.x1f(m,k,j,i));
    By = 0.25*(bcc(m,IBY,k+1,j,i) + bcc(m,IBY,k,j,i)
              + bcc(m,IBY,k+1,j,i-1) + bcc(m,IBY,k,j,i-1));
    Bz = 0.5*(b0.x3f(m,k+1,j,i) + b0.x3f(m,k+1,j,i-1));
    Real e2_kp1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ2(m,k+1,j,i,b0,size.d_view(m),three_d);

    Real e2_fc = 0.5*(e2_k + e2_kp1);

    // e3 at x3-edge (j)
    Bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j-1,i));
    By = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j,i-1));
    Bz = 0.25*(bcc(m,IBZ,k,j,i) + bcc(m,IBZ,k,j-1,i)
              + bcc(m,IBZ,k,j,i-1) + bcc(m,IBZ,k,j-1,i-1));
    Real e3_j = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
              * EdgeJ3(m,k,j,i,b0,size.d_view(m),multi_d);

    // e3 at x3-edge (j+1)
    Bx = 0.5*(b0.x1f(m,k,j+1,i) + b0.x1f(m,k,j,i));
    By = 0.5*(b0.x2f(m,k,j+1,i) + b0.x2f(m,k,j+1,i-1));
    Bz = 0.25*(bcc(m,IBZ,k,j+1,i) + bcc(m,IBZ,k,j,i)
              + bcc(m,IBZ,k,j+1,i-1) + bcc(m,IBZ,k,j,i-1));
    Real e3_jp1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ3(m,k,j+1,i,b0,size.d_view(m),multi_d);

    Real e3_fc = 0.5*(e3_j + e3_jp1);

    Real b2_fc = 0.5*(bcc(m,IBY,k,j,i-1) + bcc(m,IBY,k,j,i));
    Real b3_fc = 0.5*(bcc(m,IBZ,k,j,i-1) + bcc(m,IBZ,k,j,i));

    flx1(m,IEN,k,j,i) += e2_fc*b3_fc - e3_fc*b2_fc;
  });

  // x2-flux: e3 at x3-edges (i,i+1), e1 at x1-edges (k,k+1)
  auto &flx2 = flx.x2f;
  par_for("amb_heat2_3d", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // e3 at x3-edge (i)
    Real Bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j-1,i));
    Real By = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j,i-1));
    Real Bz = 0.25*(bcc(m,IBZ,k,j,i) + bcc(m,IBZ,k,j-1,i)
                   + bcc(m,IBZ,k,j,i-1) + bcc(m,IBZ,k,j-1,i-1));
    Real e3_i = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
              * EdgeJ3(m,k,j,i,b0,size.d_view(m),multi_d);

    // e3 at x3-edge (i+1)
    Bx = 0.5*(b0.x1f(m,k,j,i+1) + b0.x1f(m,k,j-1,i+1));
    By = 0.5*(b0.x2f(m,k,j,i+1) + b0.x2f(m,k,j,i));
    Bz = 0.25*(bcc(m,IBZ,k,j,i+1) + bcc(m,IBZ,k,j-1,i+1)
              + bcc(m,IBZ,k,j,i) + bcc(m,IBZ,k,j-1,i));
    Real e3_ip1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ3(m,k,j,i+1,b0,size.d_view(m),multi_d);

    Real e3_fc = 0.5*(e3_i + e3_ip1);

    // e1 at x1-edge (k)
    Bx = 0.25*(bcc(m,IBX,k,j,i) + bcc(m,IBX,k-1,j,i)
              + bcc(m,IBX,k,j-1,i) + bcc(m,IBX,k-1,j-1,i));
    By = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k-1,j,i));
    Bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j-1,i));
    Real e1_k = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
              * EdgeJ1(m,k,j,i,b0,size.d_view(m),multi_d,three_d);

    // e1 at x1-edge (k+1)
    Bx = 0.25*(bcc(m,IBX,k+1,j,i) + bcc(m,IBX,k,j,i)
              + bcc(m,IBX,k+1,j-1,i) + bcc(m,IBX,k,j-1,i));
    By = 0.5*(b0.x2f(m,k+1,j,i) + b0.x2f(m,k,j,i));
    Bz = 0.5*(b0.x3f(m,k+1,j,i) + b0.x3f(m,k+1,j-1,i));
    Real e1_kp1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ1(m,k+1,j,i,b0,size.d_view(m),multi_d,three_d);

    Real e1_fc = 0.5*(e1_k + e1_kp1);

    Real b1_fc = 0.5*(bcc(m,IBX,k,j-1,i) + bcc(m,IBX,k,j,i));
    Real b3_fc = 0.5*(bcc(m,IBZ,k,j-1,i) + bcc(m,IBZ,k,j,i));

    flx2(m,IEN,k,j,i) += e3_fc*b1_fc - e1_fc*b3_fc;
  });

  // x3-flux: e1 at x1-edges (j,j+1), e2 at x2-edges (i,i+1)
  auto &flx3 = flx.x3f;
  par_for("amb_heat3_3d", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // e1 at x1-edge (j)
    Real Bx = 0.25*(bcc(m,IBX,k,j,i) + bcc(m,IBX,k-1,j,i)
                   + bcc(m,IBX,k,j-1,i) + bcc(m,IBX,k-1,j-1,i));
    Real By = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k-1,j,i));
    Real Bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j-1,i));
    Real e1_j = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
              * EdgeJ1(m,k,j,i,b0,size.d_view(m),multi_d,three_d);

    // e1 at x1-edge (j+1)
    Bx = 0.25*(bcc(m,IBX,k,j+1,i) + bcc(m,IBX,k-1,j+1,i)
              + bcc(m,IBX,k,j,i) + bcc(m,IBX,k-1,j,i));
    By = 0.5*(b0.x2f(m,k,j+1,i) + b0.x2f(m,k-1,j+1,i));
    Bz = 0.5*(b0.x3f(m,k,j+1,i) + b0.x3f(m,k,j,i));
    Real e1_jp1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ1(m,k,j+1,i,b0,size.d_view(m),multi_d,three_d);

    Real e1_fc = 0.5*(e1_j + e1_jp1);

    // e2 at x2-edge (i)
    Bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k-1,j,i));
    By = 0.25*(bcc(m,IBY,k,j,i) + bcc(m,IBY,k-1,j,i)
              + bcc(m,IBY,k,j,i-1) + bcc(m,IBY,k-1,j,i-1));
    Bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k,j,i-1));
    Real e2_i = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
              * EdgeJ2(m,k,j,i,b0,size.d_view(m),three_d);

    // e2 at x2-edge (i+1)
    Bx = 0.5*(b0.x1f(m,k,j,i+1) + b0.x1f(m,k-1,j,i+1));
    By = 0.25*(bcc(m,IBY,k,j,i+1) + bcc(m,IBY,k-1,j,i+1)
              + bcc(m,IBY,k,j,i) + bcc(m,IBY,k-1,j,i));
    Bz = 0.5*(b0.x3f(m,k,j,i+1) + b0.x3f(m,k,j,i));
    Real e2_ip1 = eta * (SQR(Bx) + SQR(By) + SQR(Bz))
                * EdgeJ2(m,k,j,i+1,b0,size.d_view(m),three_d);

    Real e2_fc = 0.5*(e2_i + e2_ip1);

    Real b1_fc = 0.5*(bcc(m,IBX,k-1,j,i) + bcc(m,IBX,k,j,i));
    Real b2_fc = 0.5*(bcc(m,IBY,k-1,j,i) + bcc(m,IBY,k,j,i));

    flx3(m,IEN,k,j,i) += e1_fc*b2_fc - e2_fc*b1_fc;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Resistivity::NewTimeStep()
//! \brief Compute new time step for non-ideal MHD (Ohmic + ambipolar).
//! Ohmic:     dt <= fac * dx^2 / eta_ohm                  (constant diffusivity)
//! Ambipolar: dt <= fac * dx^2 / (eta_ad * B_max^2)       (diffusivity eta_ad*B^2 varies)
//! The most restrictive limit over the active terms and all directions is taken.

void Resistivity::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  // non-ideal MHD timestep on MeshBlock(s) in this pack
  dtnew = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mb_size;
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }

  // Effective diffusivity is the sum of the Ohmic coefficient and the maximum ambipolar
  // coefficient (eta_ad * B_max^2). Summing is conservative when both terms are active.
  Real eta_eff = eta_ohm;
  if (eta_ad != 0.0) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb1 = pmy_pack->nmb_thispack - 1;
    auto &bcc0 = pmy_pack->pmhd->bcc0;

    // Find max B^2 across all cells and meshblocks in this pack
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

    eta_eff += eta_ad * max_bsq;
  }

  // If no non-ideal term contributes (e.g. eta_ohm=0 and B~0 everywhere), leave dtnew at
  // the float max so it does not constrain the global timestep.
  if (eta_eff <= 0.0) {
    return;
  }

  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
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
