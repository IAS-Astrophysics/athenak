//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file excision.cpp
//! \brief sets boolean masks for horizon excision

#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates.hpp"
#include "cell_locations.hpp"
#include "coordinates/adm.hpp"

// inlined spherical Kerr-Schild r evaluated at CKS x1, x2, x3
KOKKOS_INLINE_FUNCTION
Real KSRX(const Real x1, const Real x2, const Real x3, const Real a) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(x3)))/2.0);
}

KOKKOS_INLINE_FUNCTION
Real SmoothStep01(const Real x) {
  Real s = fmin(1.0, fmax(0.0, x));
  return s*s*(3.0 - 2.0*s);
}

KOKKOS_INLINE_FUNCTION
Real KSRXSpin(const Real x1, const Real x2, const Real x3,
              const Real ax, const Real ay, const Real az) {
  Real a2 = SQR(ax) + SQR(ay) + SQR(az);
  if (a2 <= 0.0) {
    return sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  }
  Real a = sqrt(a2);
  Real rad2 = SQR(x1) + SQR(x2) + SQR(x3);
  Real zspin = (x1*ax + x2*ay + x3*az)/a;
  return sqrt((rad2 - a2 + sqrt(SQR(rad2 - a2) + 4.0*a2*SQR(zspin)))/2.0);
}

KOKKOS_INLINE_FUNCTION
void BoostedDisplacement(const Real x1, const Real x2, const Real x3,
                         const Real cx, const Real cy, const Real cz,
                         const Real vx, const Real vy, const Real vz,
                         Real *xbh, Real *ybh, Real *zbh) {
  Real dx = x1 - cx, dy = x2 - cy, dz = x3 - cz;
  Real v2 = SQR(vx) + SQR(vy) + SQR(vz);
  Real q;
  if (v2 < 1.0e-12) {
    q = 0.5 + 0.375*v2 + 0.3125*SQR(v2);
  } else {
    Real gamma = 1.0/sqrt(fmax(1.0 - v2, 1.0e-300));
    q = (gamma - 1.0)/v2;
  }
  Real vd = vx*dx + vy*dy + vz*dz;
  *xbh = dx + q*vx*vd;
  *ybh = dy + q*vy*vd;
  *zbh = dz + q*vz*vd;
}

//----------------------------------------------------------------------------------------
//! \fn void Coordinates::SetExcisionMasks()
//  \brief Sets boolean masks for the excision in CKS

void Coordinates::SetExcisionMasks(DvceArray4D<bool> &excision_floor,
                                   DvceArray4D<bool> &excision_flux) {
  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is; int js = indcs.js; int ks = indcs.ks;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &spin = coord_data.bh_spin;
  auto &excision_radius = coord_data.rexcise;
  auto &weight = excision_weight;
  Real smooth_width = coord_data.smooth_excise_width;

  auto &flux_excise_r = coord_data.flux_excise_r;

  // NOTE(@pdmullen):
  // excision_floor: - if r_ks evaluated at this CC is <= excision_radius, mask the cell.
  // excision_flux:  - if r_ks evaluated at any portion of the two cells connecting
  //                   each face of this cell is <= excision_radius, mask the cell.
  par_for("set_excision", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // NOTE(@pdmullen): In some instances, calls to x? will access coordinate information
    // for which there is *no corresponding logical counterpart*, however, the
    // LeftEdgeX/CellCenterX functions can handle "out-of-range" queries.
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    // We calculate the distance to the corner to make sure that only cells completely
    // inside the horizon are excised.
    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    Real x1v   = CellCenterX(i  -is, indcs.nx1, x1min, x1max);
    Real x1vm1 = CellCenterX(i-1-is, indcs.nx1, x1min, x1max);
    Real x1vp1 = CellCenterX(i+1-is, indcs.nx1, x1min, x1max);
    Real x1f   = LeftEdgeX  (i  -is, indcs.nx1, x1min, x1max);
    Real x1fm1 = LeftEdgeX  (i-1-is, indcs.nx1, x1min, x1max);
    Real x1fp1 = LeftEdgeX  (i+1-is, indcs.nx1, x1min, x1max);
    Real x1fp2 = LeftEdgeX  (i+2-is, indcs.nx1, x1min, x1max);

    Real x2v   = CellCenterX(j  -js, indcs.nx2, x2min, x2max);
    Real x2vm1 = CellCenterX(j-1-js, indcs.nx2, x2min, x2max);
    Real x2vp1 = CellCenterX(j+1-js, indcs.nx2, x2min, x2max);
    Real x2f   = LeftEdgeX  (j  -js, indcs.nx2, x2min, x2max);
    Real x2fm1 = LeftEdgeX  (j-1-js, indcs.nx2, x2min, x2max);
    Real x2fp1 = LeftEdgeX  (j+1-js, indcs.nx2, x2min, x2max);
    Real x2fp2 = LeftEdgeX  (j+2-js, indcs.nx2, x2min, x2max);

    Real x3v   = CellCenterX(k  -ks, indcs.nx3, x3min, x3max);
    Real x3vm1 = CellCenterX(k-1-ks, indcs.nx3, x3min, x3max);
    Real x3vp1 = CellCenterX(k+1-ks, indcs.nx3, x3min, x3max);
    Real x3f   = LeftEdgeX  (k  -ks, indcs.nx3, x3min, x3max);
    Real x3fm1 = LeftEdgeX  (k-1-ks, indcs.nx3, x3min, x3max);
    Real x3fp1 = LeftEdgeX  (k+1-ks, indcs.nx3, x3min, x3max);
    Real x3fp2 = LeftEdgeX  (k+2-ks, indcs.nx3, x3min, x3max);

    excision_floor(m,k,j,i) = false;
    excision_flux(m,k,j,i) = false;
    weight(m,k,j,i) = 0.0;

    // Set excision floor mask
    Real rks = KSRX(x1v, x2v, x3v, spin);
    if (rks <= excision_radius) {
      excision_floor(m,k,j,i) = true;
      weight(m,k,j,i) = SmoothStep01((excision_radius - rks)/smooth_width);
    }

    // Set excision flux mask
    Real x1, x2, x3;

    // Check face at i
    x1 = x1v;
    x1 = (fabs(x1) < fabs(x1vm1)) ? x1 : x1vm1;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fm1)) ? x1 : x1fm1;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x2 = x2v;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x3 = x3v;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    if (KSRX(x1,x2,x3,spin) <= flux_excise_r) excision_flux(m,k,j,i) = true;

    // check face at i+1
    x1 = x1vp1;
    x1 = (fabs(x1) < fabs(x1v))   ? x1 : x1v;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fp2)) ? x1 : x1fp2;
    if (KSRX(x1,x2,x3,spin) <= flux_excise_r) excision_flux(m,k,j,i) = true;

    // Check face at j
    x1 = x1v;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x2 = x2v;
    x2 = (fabs(x2) < fabs(x2vm1)) ? x2 : x2vm1;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fm1)) ? x2 : x2fm1;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x3 = x3v;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    if (KSRX(x1,x2,x3,spin) <= flux_excise_r) excision_flux(m,k,j,i) = true;

    // Check face at j+1
    x2 = x2vp1;
    x2 = (fabs(x2) < fabs(x2v))   ? x2 : x2v;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fp2)) ? x2 : x2fp2;
    if (KSRX(x1,x2,x3,spin) <= flux_excise_r) excision_flux(m,k,j,i) = true;

    // Check face at k
    x1 = x1v;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x2 = x2v;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x3 = x3v;
    x3 = (fabs(x3) < fabs(x3vm1)) ? x3 : x3vm1;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fm1)) ? x3 : x3fm1;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    if (KSRX(x1,x2,x3,spin) <= flux_excise_r) excision_flux(m,k,j,i) = true;

    // Check face at k+1
    x3 = x3vp1;
    x3 = (fabs(x3) < fabs(x3v))   ? x3 : x3v;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fp2)) ? x3 : x3fp2;
    if (KSRX(x1,x2,x3,spin) <= flux_excise_r) excision_flux(m,k,j,i) = true;
  });

  return;
}

void Coordinates::UpdateExcisionMasks() {
  if (coord_data.excision_scheme == ExcisionScheme::lapse) {
    // capture variables for kernel
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    int nmb1 = pmy_pack->nmb_thispack - 1;
    auto &adm = pmy_pack->padm->adm;
    auto &floor = excision_floor;
    auto &flux = excision_flux;
    auto &weight = excision_weight;

    Real &excise_lapse = coord_data.excise_lapse;
    Real lapse_width = coord_data.smooth_excise_lapse_width;

    par_for("set_excision", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      bool excise = (adm.alpha(m,k,j,i) < excise_lapse);
      floor(m,k,j,i) = excise;
      flux(m,k,j,i) = excise;
      weight(m,k,j,i) = excise ? SmoothStep01((excise_lapse - adm.alpha(m,k,j,i))/
                                                lapse_width) : 0.0;
    });
  } else if (coord_data.excision_scheme == ExcisionScheme::puncture) {
    // capture variables for kernel
    auto &size = pmy_pack->pmb->mb_size; 
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    int is = indcs.is; int js = indcs.js; int ks = indcs.ks;
    int nmb1 = pmy_pack->nmb_thispack - 1;
    auto &floor = excision_floor;
    auto &flux = excision_flux;
    auto &weight = excision_weight;

    Real p0_x = coord_data.punc_0[0];
    Real p0_y = coord_data.punc_0[1];
    Real p0_z = coord_data.punc_0[2];
    Real p0_ax = coord_data.punc_0_spin[0];
    Real p0_ay = coord_data.punc_0_spin[1];
    Real p0_az = coord_data.punc_0_spin[2];
    Real p0_vx = coord_data.punc_0_vel[0];
    Real p0_vy = coord_data.punc_0_vel[1];
    Real p0_vz = coord_data.punc_0_vel[2];
    
    Real p1_x = coord_data.punc_1[0];
    Real p1_y = coord_data.punc_1[1];
    Real p1_z = coord_data.punc_1[2];
    Real p1_ax = coord_data.punc_1_spin[0];
    Real p1_ay = coord_data.punc_1_spin[1];
    Real p1_az = coord_data.punc_1_spin[2];
    Real p1_vx = coord_data.punc_1_vel[0];
    Real p1_vy = coord_data.punc_1_vel[1];
    Real p1_vz = coord_data.punc_1_vel[2];

    Real &punc_0_r = coord_data.punc_0_rad;
    Real &punc_1_r = coord_data.punc_1_rad;
    Real smooth_width = coord_data.smooth_excise_width;

    par_for("set_excision", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x1v   = CellCenterX(i  -is, indcs.nx1, x1min, x1max);
      Real x2v   = CellCenterX(j  -js, indcs.nx2, x2min, x2max);
      Real x3v   = CellCenterX(k  -ks, indcs.nx3, x3min, x3max);

      Real x0, y0, z0, x1, y1, z1;
      BoostedDisplacement(x1v, x2v, x3v, p0_x, p0_y, p0_z, p0_vx, p0_vy, p0_vz,
                          &x0, &y0, &z0);
      BoostedDisplacement(x1v, x2v, x3v, p1_x, p1_y, p1_z, p1_vx, p1_vy, p1_vz,
                          &x1, &y1, &z1);
      Real r0 = KSRXSpin(x0, y0, z0, p0_ax, p0_ay, p0_az);
      Real r1 = KSRXSpin(x1, y1, z1, p1_ax, p1_ay, p1_az);

      bool excise = ( r0 <= punc_0_r || r1 <= punc_1_r );
      floor(m,k,j,i) = excise;
      flux(m,k,j,i) = excise;
      Real w0 = (punc_0_r > 0.0) ? SmoothStep01((punc_0_r - r0)/smooth_width) : 0.0;
      Real w1 = (punc_1_r > 0.0) ? SmoothStep01((punc_1_r - r1)/smooth_width) : 0.0;
      weight(m,k,j,i) = fmax(w0, w1);
    });
  }
}
