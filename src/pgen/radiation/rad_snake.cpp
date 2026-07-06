//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_snake.cpp
//  \brief Snake beam test for radiation

// C++ headers
#include <algorithm>  // min, max
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/srcterms.hpp"
#include "pgen.hpp"

KOKKOS_INLINE_FUNCTION
void ComputeSnakeMetricAndTetrad(Real x, Real y, Real z,
                                 const Real mag, const Real kym, const bool snake_tet,
                                 Real g[][4], Real gi[][4],
                                 Real e[][4], Real ecov[][4], Real omega[][4][4]);

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Sets initial conditions for GR radiation snake beam test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);
  auto &size = pmbp->pmb->mb_size;
  int &nmb = pmbp->nmb_thispack;

  // angular mesh data
  auto &num_neighbors_ = pmbp->prad->prgeo->num_neighbors;

  // snake params
  Real mag = pin->GetReal("problem", "snake_mag");
  Real kym = pin->GetReal("problem", "snake_kym");
  bool snake_tet = pin->GetOrAddBoolean("problem", "snake_tet", false);

  // override tetrad
  auto &nh_c_ = pmbp->prad->nh_c;
  auto &nh_f_ = pmbp->prad->nh_f;
  auto &tet_c_ = pmbp->prad->tet_c;
  auto &tetcov_c_ = pmbp->prad->tetcov_c;
  auto &unit_flux_ = pmbp->prad->prgeo->unit_flux;
  auto &na_ = pmbp->prad->na;
  par_for("tet_c",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
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

    // calculate snake metric and tetrad
    Real glower[4][4] = {0.0}; Real gupper[4][4] = {0.0};
    Real e[4][4] = {0.0}; Real e_cov[4][4] = {0.0}; Real omega[4][4][4] = {0.0};
    ComputeSnakeMetricAndTetrad(x1v,x2v,x3v,mag,kym,snake_tet,
                                glower,gupper,e,e_cov,omega);

    // set tetrad (and covariant tetrad)
    for (int d1=0; d1<4; ++d1) {
      for (int d2=0; d2<4; ++d2) {
        tet_c_   (m,d1,d2,k,j,i) = e[d1][d2];
        tetcov_c_(m,d1,d2,k,j,i) = e_cov[d1][d2];
      }
    }

    // set n^a coordinate components
    for (int n=0; n<=nang1; ++n) {
      for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
        Real zetaf = acos(nh_f_.d_view(n,nb,3));
        Real na1 = 0.0; Real na2 = 0.0;
        for (int q=0; q<4; ++q) {
          for (int p=0; p<4; ++p) {
            na1 += (1.0/sin(zetaf)*nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p)
                    * (nh_f_.d_view(n,nb,0)*omega[3][q][p]
                    -  nh_f_.d_view(n,nb,3)*omega[0][q][p]));
            na2 += (1.0/SQR(sin(zetaf))*nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p)
                    * (nh_f_.d_view(n,nb,2)*omega[1][q][p]
                    -  nh_f_.d_view(n,nb,1)*omega[2][q][p]));
          }
        }
        na_(m,n,k,j,i,nb) = na1*unit_flux_.d_view(n,nb,0)
                          + SQR(sin(zetaf))*na2*unit_flux_.d_view(n,nb,1);
      }
    }
  });

  // set tetrad components (subset) at x1f
  auto tet_d1_x1f_ = pmbp->prad->tet_d1_x1f;
  par_for("tet_x1f",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,n1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1f = LeftEdgeX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // calculate snake metric and tetrad
    Real glower[4][4] = {0.0}; Real gupper[4][4] = {0.0};
    Real e[4][4] = {0.0}; Real e_cov[4][4] = {0.0}; Real omega[4][4][4] = {0.0};
    ComputeSnakeMetricAndTetrad(x1f,x2v,x3v,mag,kym,snake_tet,
                                glower,gupper,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d1_x1f_   (m,d,k,j,i) = e[d][1]; }
  });

  // set tetrad components (subset) at x2f
  auto tet_d2_x2f_ = pmbp->prad->tet_d2_x2f;
  par_for("tet_x2f",DevExeSpace(),0,(nmb-1),0,(n3-1),0,n2,0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2f = LeftEdgeX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // calculate snake metric and tetrad
    Real glower[4][4] = {0.0}; Real gupper[4][4] = {0.0};
    Real e[4][4] = {0.0}; Real e_cov[4][4] = {0.0}; Real omega[4][4][4] = {0.0};
    ComputeSnakeMetricAndTetrad(x1v,x2f,x3v,mag,kym,snake_tet,
                                glower,gupper,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d2_x2f_(m,d,k,j,i) = e[d][2]; }
  });

  // set tetrad components (subset) at x3f
  auto tet_d3_x3f_ = pmbp->prad->tet_d3_x3f;
  par_for("tet_x3f",DevExeSpace(),0,(nmb-1),0,n3,0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3f = LeftEdgeX(k-ks, indcs.nx3, x3min, x3max);

    // calculate snake metric and tetrad
    Real glower[4][4] = {0.0}; Real gupper[4][4] = {0.0};
    Real e[4][4] = {0.0}; Real e_cov[4][4] = {0.0}; Real omega[4][4][4] = {0.0};
    ComputeSnakeMetricAndTetrad(x1v,x2v,x3f,mag,kym,snake_tet,
                                glower,gupper,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d3_x3f_   (m,d,k,j,i) = e[d][3]; }
  });

  // set initial condition intensity and beam source mask
  Real p1 = pin->GetReal("problem", "pos_1");
  Real p2 = pin->GetReal("problem", "pos_2");
  Real p3 = pin->GetReal("problem", "pos_3");
  Real width_ = pin->GetReal("problem", "width");
  Real spread_ = pin->GetReal("problem", "spread");
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

    // calculate snake metric and tetrad
    Real glower[4][4] = {0.0}; Real gupper[4][4] = {0.0};
    Real e[4][4] = {0.0}; Real e_cov[4][4] = {0.0}; Real omega[4][4][4] = {0.0};
    ComputeSnakeMetricAndTetrad(x1v,x2v,x3v,mag,kym,snake_tet,
                                glower,gupper,e,e_cov,omega);

    // Calculate proper distance to beam origin and minimum angle between directions
    Real dx1 = x1v - p1;
    Real dx2 = x2v - p2;
    Real dx3 = x3v - p3;
    Real dx_sq = glower[1][1]*dx1*dx1+2.0*glower[1][2]*dx1*dx2+2.0*glower[1][3]*dx1*dx3
               + glower[2][2]*dx2*dx2+2.0*glower[2][3]*dx2*dx3
               + glower[3][3]*dx3*dx3;
    Real mu_min = cos(spread_/2.0*M_PI/180.0);

    // Calculate contravariant time component of direction
    Real dmag = sqrt(1.0 + SQR(mag*kym*M_PI*cos(kym*M_PI*x2v)));
    Real d1 = (-mag*kym*M_PI*cos(kym*M_PI*x2v))/dmag;
    Real d2 = 1.0/dmag;
    Real d3 = 0.0/dmag;
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

KOKKOS_INLINE_FUNCTION
void ComputeSnakeMetricAndTetrad(Real x, Real y, Real z,
                                 const Real mag, const Real kym, const bool snake_tet,
                                 Real g[][4], Real gi[][4],
                                 Real e[][4], Real ecov[][4], Real omega[][4][4]) {
  // terms appearing in snake metric/tetrads
  Real cos_term = -mag*kym*M_PI*cos(kym*M_PI*y);
  Real sin_term =  mag*SQR(kym*M_PI)*sin(kym*M_PI*y);

  // covariant metric
  g[0][0] = -1.0;
  g[1][1] = 1.0;
  g[1][2] = -cos_term;
  g[2][1] = -cos_term;
  g[2][2] = 1.0 + SQR(cos_term);
  g[3][3] = 1.0;

  // contravariant metric
  gi[0][0] = -1.0;
  gi[1][1] = 1.0 + SQR(cos_term);
  gi[1][2] = cos_term;
  gi[2][1] = cos_term;
  gi[2][2] = 1.0;
  gi[3][3] = 1.0;

  // derivatives of covariant metric
  Real dg[4][4][4] = {0.0};
  dg[2][1][2] = -sin_term;
  dg[2][2][1] = -sin_term;
  dg[2][2][2] = 2.0*cos_term*sin_term;

  // tetrad (aligned with Cartesian)
  e[0][0] = 1.0;
  e[1][1] = 1.0;
  e[2][1] = cos_term;
  e[2][2] = 1.0;
  e[3][3] = 1.0;

  // derivatives of tetrad
  Real de[4][4][4] = {0.0};
  de[2][2][1] = sin_term;

  if (snake_tet) {
    e[2][1] = 0.0;  // reset to zero
    // snake tetrad (aligned with snake)
    e[0][0] = 1.0;
    e[1][1] = sqrt(1.0+SQR(cos_term));
    e[1][2] = cos_term/sqrt(1.0+SQR(cos_term));
    e[2][2] = 1.0/sqrt(1.0+SQR(cos_term));
    e[3][3] = 1.0;

    de[2][2][1] = 0.0;  // reset to zero
    // derivatives of snake tetrad
    de[2][1][1] = cos_term*sin_term/sqrt(1.0+SQR(cos_term));
    de[2][1][2] = sin_term/pow(sqrt(1.0+SQR(cos_term)), 3.0);
    de[2][2][2] = -cos_term*sin_term/pow(sqrt(1.0+SQR(cos_term)), 3.0);
  }

  // Set Minkowski metric
  Real eta[4][4] = {0.0};
  eta[0][0] = -1.0;
  eta[1][1] = 1.0;
  eta[2][2] = 1.0;
  eta[3][3] = 1.0;

  // Calculate covariant tetrad
  Real ei[4][4] = {0.0};
  Real gamma[4][4][4] = {0.0};
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      ecov[i][j]=0.0;
      ei[i][j]=0.0;
      for (int k=0; k<4; ++k) {
        gamma[i][j][k]=0.0;
        ecov[i][j] += g[j][k]*e[i][k];
        for (int l=0; l<4; ++l) {
          ei[i][j] += eta[i][k]*g[j][l]*e[k][l];
          gamma[i][j][k] += 0.5*gi[i][l]*(dg[j][l][k] + dg[k][l][j] - dg[l][j][k]);
        }
      }
    }
  }

  // Calculate Ricci rotation coefficients
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      for (int k=0; k<4; ++k) {
        omega[i][j][k]=0.0;
        for (int l=0; l<4; ++l) {
          for (int m=0; m<4; ++m) {
            omega[i][j][k] += ei[i][l]*e[k][m]*de[m][j][l];
            for (int n=0; n<4; ++n) {
              omega[i][j][k] += ei[i][l]*e[k][m]*gamma[l][m][n]*e[j][n];
            }
          }
        }
      }
    }
  }

  return;
}
