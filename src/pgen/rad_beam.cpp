//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_beam.cpp
//  \brief Beam test for radiation

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation beam test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);
  auto &size = pmbp->pmb->mb_size;
  auto &flat = pmbp->pcoord->coord_data.is_minkowski;
  auto &spin = pmbp->pcoord->coord_data.bh_spin;

  // set initial condition intensity and beam source mask
  Real p1 = pin->GetReal("problem", "pos_1");
  Real p2 = pin->GetReal("problem", "pos_2");
  Real p3 = pin->GetReal("problem", "pos_3");
  Real d1 = pin->GetReal("problem", "dir_1");
  Real d2 = pin->GetReal("problem", "dir_2");
  Real d3 = pin->GetReal("problem", "dir_3");
  Real width_ = pin->GetReal("problem", "width");
  Real spread_ = pin->GetReal("problem", "spread");

  auto &nh_c_ = pmbp->prad->nh_c;
  auto &tet_c_ = pmbp->prad->tet_c;
  auto &beam_mask = pmbp->prad->beam_mask;
  par_for("rad_beam",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);

    // Calculate proper distance to beam origin and minimum angle between directions
    Real dx1 = x1v - p1;
    Real dx2 = x2v - p2;
    Real dx3 = x3v - p3;
    Real dx_sq = glower[1][1]*dx1*dx1 +2.0*glower[1][2]*dx1*dx2 + 2.0*glower[1][3]*dx1*dx3
               + glower[2][2]*dx2*dx2 +2.0*glower[2][3]*dx2*dx3
               + glower[3][3]*dx3*dx3;
    Real mu_min = cos(spread_/2.0*M_PI/180.0);

    // Calculate contravariant time component of direction
    Real temp_a = glower[0][0];
    Real temp_b = 2.0*(glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3);
    Real temp_c = glower[1][1]*d1*d1 + 2.0*glower[1][2]*d1*d2 + 2.0*glower[1][3]*d1*d3
                + glower[2][2]*d2*d2 + 2.0*glower[2][3]*d2*d3
                + glower[3][3]*d3*d3;
    Real d0 = ((-temp_b - sqrt(SQR(temp_b) - 4.0*temp_a*temp_c))/(2.0*temp_a));

    // lower indices
    Real dc0 = glower[0][0]*d0 + glower[0][1]*d1 + glower[0][2]*d2 + glower[0][3]*d3;
    Real dc1 = glower[0][1]*d0 + glower[1][1]*d1 + glower[1][2]*d2 + glower[1][3]*d3;
    Real dc2 = glower[0][2]*d0 + glower[1][2]*d1 + glower[2][2]*d2 + glower[2][3]*d3;
    Real dc3 = glower[0][3]*d0 + glower[1][3]*d1 + glower[2][3]*d2 + glower[3][3]*d3;

    // Calculate covariant direction in tetrad frame
    Real dtc0 = (tet_c_(m,0,0,k,j,i)*dc0 + tet_c_(m,0,1,k,j,i)*dc1 +
                 tet_c_(m,0,2,k,j,i)*dc2 + tet_c_(m,0,3,k,j,i)*dc3);
    Real dtc1 = (tet_c_(m,1,0,k,j,i)*dc0 + tet_c_(m,1,1,k,j,i)*dc1 +
                 tet_c_(m,1,2,k,j,i)*dc2 + tet_c_(m,1,3,k,j,i)*dc3)/(-dtc0);
    Real dtc2 = (tet_c_(m,2,0,k,j,i)*dc0 + tet_c_(m,2,1,k,j,i)*dc1 +
                 tet_c_(m,2,2,k,j,i)*dc2 + tet_c_(m,2,3,k,j,i)*dc3)/(-dtc0);
    Real dtc3 = (tet_c_(m,3,0,k,j,i)*dc0 + tet_c_(m,3,1,k,j,i)*dc1 +
                 tet_c_(m,3,2,k,j,i)*dc2 + tet_c_(m,3,3,k,j,i)*dc3)/(-dtc0);

    // Go through angles
    for (int n=0; n<=nang1; ++n) {
      Real mu = (nh_c_.d_view(n,1) * dtc1
               + nh_c_.d_view(n,2) * dtc2
               + nh_c_.d_view(n,3) * dtc3);
      if ((dx_sq < SQR(width_/2.0)) && (mu > mu_min)) {
        beam_mask(m,n,k,j,i) = true;
      } else {
        beam_mask(m,n,k,j,i) = false;
      }
    }
  });

  return;
}
