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

// inlined spherical Kerr-Schild r evaluated at CKS x1, x2, x3
KOKKOS_INLINE_FUNCTION
Real KSRX(const Real x1, const Real x2, const Real x3, const Real a) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(x3)))/2.0);
}

//----------------------------------------------------------------------------------------
//! \fn void Coordinates::SetExcisionMasks()
//  \brief Sets boolean masks for the excision radius in CKS

void Coordinates::SetExcisionMasks() {
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

  auto &cc_mask_ = cc_mask;
  auto &fc_mask_ = fc_mask;
  // NOTE(@pdmullen):
  // cc_mask: if r_ks evaluated at *this cell-center* is <= 1, mask the cell.
  // fc_mask: if r_ks evaluated at *this face-center* is <=1, or if *any other
  // portion of grid cells sharing this face* is <=1, mask the cell (added complexity
  // here as two neighboring cells share a face)
  par_for("set_masks", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Extract grid cell positions
    Real x1, x2, x3;
    // NOTE(@pdmullen):
    // In some instances, calls to x? will access coordinate information for which
    // there is *no corresponding logical counterpart*, however, the
    // LeftEdgeX/CellCenterX functions can handle "out-of-range" queries.
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

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

    // Set cc_mask
    x1 = x1v;
    x2 = x2v;
    x3 = x3v;
    cc_mask_(m,k,j,i) = (KSRX(x1,x2,x3,spin) <= 1.0) ? true : false;

    // Set fc_mask.x1f
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
    fc_mask_.x1f(m,k,j,i) = (KSRX(x1,x2,x3,spin) <= 1.0) ? true : false;
    if (i==(n1-1)) {
      x1 = x1vp1;
      x1 = (fabs(x1) < fabs(x1v))   ? x1 : x1v;
      x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
      x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
      x1 = (fabs(x1) < fabs(x1fp2)) ? x1 : x1fp2;
      fc_mask_.x1f(m,k,j,i+1) = (KSRX(x1,x2,x3,spin) <= 1.0) ? true : false;
    }

    // Set fc_mask.x2f
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
    fc_mask_.x2f(m,k,j,i) = (KSRX(x1,x2,x3,spin) <= 1.0) ? true : false;
    if (j==(n2-1)) {
      x2 = x2vp1;
      x2 = (fabs(x2) < fabs(x2v))   ? x2 : x2v;
      x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
      x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
      x2 = (fabs(x2) < fabs(x2fp2)) ? x2 : x2fp2;
      fc_mask_.x2f(m,k,j+1,i) = (KSRX(x1,x2,x3,spin) <= 1.0) ? true : false;
    }

    // Set fc_mask.x3f
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
    fc_mask_.x3f(m,k,j,i) = (KSRX(x1,x2,x3,spin) <= 1.0) ? true : false;
    if (k==(n3-1)) {
      x3 = x3vp1;
      x3 = (fabs(x3) < fabs(x3v))   ? x3 : x3v;
      x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
      x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
      x3 = (fabs(x3) < fabs(x3fp2)) ? x3 : x3fp2;
      fc_mask_.x3f(m,k+1,j,i) = (KSRX(x1,x2,x3,spin) <= 1.0) ? true : false;
    }
  });

  return;
}
