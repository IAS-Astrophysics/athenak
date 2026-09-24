//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ambipolar.cpp
//  \brief Implements ambipolar diffusion methods of the Resistivity class: the inline
//  edge-current helpers EdgeJ{1,2,3} and the ambipolar EMF and Poynting-flux kernels.

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "resistivity.hpp"

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
//! \fn AddEMFConstantAmbipolar()
//  \brief Adds electric field from ambipolar diffusion to corner-centered electric field
//    E_amb = eta_ad * [ B^2 J - (J . B) B ]     (eta_A = eta_ad * B^2)
//  J = curl(B) and B are evaluated at each edge and combined as above. J is recomputed
//  inline via EdgeJ{1,2,3}(); the cross-J components and B are averaged to the edge
//  (4-pt in 3D, 2-pt across a degenerate direction in 2D), with cell-centered B (bcc0)
//  from the parent MHD class used for the edge-diagonal interpolation. Follows Athena++
//  AmbipolarEMF (field_diffusion/diffusivity.cpp).

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
//  \brief Adds Poynting flux from ambipolar diffusion to energy flux
//  S_AD = (E_amb X B) = eta_ad*B^2*(J X B)   [the (J.B)B term drops since (J.B)B X B = 0]
//  The edge EMF eta_ad*B^2*J is formed at each edge with the same B interpolation as
//  AddEMFConstantAmbipolar, averaged to the face, then crossed with face-averaged
//  cell-centered B (bcc0). J is recomputed inline via EdgeJ{1,2,3}().

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
