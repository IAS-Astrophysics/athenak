//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_update.cpp
//  \brief Performs update of MHD conserved variables (u0) for each stage of explicit
//  SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and partial time
//  step appropriate to stage.
//  Both the flux divergence and physical source terms are included in the update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::EntropyReset
//  \brief Reset the total entropy for entropy fix

void MHD::EntropyReset() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  auto &coord = pmy_pack->pcoord->coord_data;
  auto &flat = coord.is_minkowski;
  auto &spin = coord.bh_spin;
  auto &u0_ = u0;
  auto &w0_ = w0;
  int entropyIdx = nmhd+nscalars-1;
  Real gm1 = peos->eos_data.gamma-1;

  auto &customize_fofc_ = customize_fofc;
  auto &fofc_ = fofc;

  par_for("entropy_reset",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // compute metric and inverse
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
    Real alpha = sqrt(-1.0/gupper[0][0]);

    // compute total entropy
    Real &wdn = w0_(m,IDN,k,j,i);
    Real &wvx = w0_(m,IVX,k,j,i);
    Real &wvy = w0_(m,IVY,k,j,i);
    Real &wvz = w0_(m,IVZ,k,j,i);
    Real &wen = w0_(m,IEN,k,j,i);
    Real q = glower[1][1]*wvx*wvx + 2.0*glower[1][2]*wvx*wvy + 2.0*glower[1][3]*wvx*wvz
           + glower[2][2]*wvy*wvy + 2.0*glower[2][3]*wvy*wvz
           + glower[3][3]*wvz*wvz;
    Real gamma = sqrt(1.0 + q);
    Real u0 = gamma/alpha;

    // assign total entropy to the first scalar
    u0_(m,entropyIdx,k,j,i) = gm1*wen / pow(wdn,gm1) * u0;

    if (customize_fofc_) u0_(m,entropyIdx,k,j,i) = fofc_(m,entropyIdx,k,j,i);

    // what to do with coarse_u0 ???
  });

  return;
} // end void MHD::EntropyReset

} // namespace mhd
