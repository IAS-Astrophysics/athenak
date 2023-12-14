//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tetrad.cpp
//  \brief sets orthonormal tetrad

#include <math.h>
#include <float.h>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation.hpp"
#include "radiation_tetrad.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SetOrthonormalTetrad()
//! \brief Set orthonormal tetrad data

void Radiation::SetOrthonormalTetrad() {
  auto &size = pmy_pack->pmb->mb_size;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &nmb = pmy_pack->nmb_thispack;

  int nang1 = prgeo->nangles - 1;
  auto &num_neighbors_ = prgeo->num_neighbors;
  auto nh_c_ = nh_c;

  auto &coord = pmy_pack->pcoord->coord_data;
  bool &flat = coord.is_minkowski;
  Real &spin = coord.bh_spin;

  // define tetrad frame
  for (int n=0; n<=nang1; ++n) {
    nh_c.h_view(n,0) = 1.0;
    nh_c.h_view(n,1) = prgeo->cart_pos.h_view(n,0);
    nh_c.h_view(n,2) = prgeo->cart_pos.h_view(n,1);
    nh_c.h_view(n,3) = prgeo->cart_pos.h_view(n,2);
    if (angular_fluxes) {
      for (int nb=0; nb<num_neighbors_.h_view(n); ++nb) {
        nh_f.h_view(n,nb,0) = 1.0;
        nh_f.h_view(n,nb,1) = prgeo->cart_pos_mid.h_view(n,nb,0);
        nh_f.h_view(n,nb,2) = prgeo->cart_pos_mid.h_view(n,nb,1);
        nh_f.h_view(n,nb,3) = prgeo->cart_pos_mid.h_view(n,nb,2);
      }
      if (num_neighbors_.h_view(n)==5) {
        nh_f.h_view(n,5,0) = (FLT_MAX);
        nh_f.h_view(n,5,1) = (FLT_MAX);
        nh_f.h_view(n,5,2) = (FLT_MAX);
        nh_f.h_view(n,5,3) = (FLT_MAX);
      }
    }
  }
  nh_c.template modify<HostMemSpace>();
  nh_c.template sync<DevExeSpace>();
  nh_f.template modify<HostMemSpace>();
  nh_f.template sync<DevExeSpace>();

  // set tetrad components
  auto tet_c_ = tet_c;
  auto tetcov_c_ = tetcov_c;
  par_for("tet_c/tetcov_c",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
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
    for (int d1=0; d1<4; ++d1) {
      for (int d2=0; d2<4; ++d2) {
        tet_c_   (m,d1,d2,k,j,i) = e[d1][d2];
        tetcov_c_(m,d1,d2,k,j,i) = e_cov[d1][d2];
      }
    }
  });

  // set tetrad components (subset) at x1f
  auto tet_d1_x1f_ = tet_d1_x1f;
  par_for("tet_d1_x1f",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,n1,
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

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1f,x2v,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1f,x2v,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1f,x2v,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d1_x1f_(m,d,k,j,i) = e[d][1]; }
  });

  // set tetrad components (subset) at x2f
  auto tet_d2_x2f_ = tet_d2_x2f;
  par_for("tet_d2_x2f",DevExeSpace(),0,(nmb-1),0,(n3-1),0,n2,0,(n1-1),
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

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2f,x3v,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2f,x3v,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2f,x3v,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d2_x2f_(m,d,k,j,i) = e[d][2]; }
  });

  // set tetrad components (subset) at x3f
  auto tet_d3_x3f_ = tet_d3_x3f;
  par_for("tet_d3_x3f",DevExeSpace(),0,(nmb-1),0,n3,0,(n2-1),0,(n1-1),
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

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v,x2v,x3f,flat,spin,glower,gupper);
    Real dgx[4][4], dgy[4][4], dgz[4][4];
    ComputeMetricDerivatives(x1v,x2v,x3f,flat,spin,dgx,dgy,dgz);
    Real e[4][4], e_cov[4][4], omega[4][4][4];
    ComputeTetrad(x1v,x2v,x3f,flat,spin,glower,gupper,dgx,dgy,dgz,e,e_cov,omega);
    for (int d=0; d<4; ++d) { tet_d3_x3f_(m,d,k,j,i) = e[d][3]; }
  });

  // Calculate n^angle
  if (angular_fluxes) {
    auto uflux = prgeo->unit_flux;
    auto nh_f_ = nh_f;
    auto na_ = na;
    par_for("na",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
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
      for (int n=0; n<=nang1; ++n) {
        for (int nb=0; nb<num_neighbors_.d_view(n); ++nb) {
          Real iszetaf = 1.0/sqrt(1.0 - SQR(nh_f_.d_view(n,nb,3)));
          Real na1 = 0.0; Real na2 = 0.0;
          for (int q=0; q<4; ++q) {
            for (int p=0; p<4; ++p) {
              Real nhfqp = nh_f_.d_view(n,nb,q)*nh_f_.d_view(n,nb,p);
              na1 += (nhfqp*(nh_f_.d_view(n,nb,0)*omega[3][q][p] -
                             nh_f_.d_view(n,nb,3)*omega[0][q][p]));
              na2 += (nhfqp*(nh_f_.d_view(n,nb,2)*omega[1][q][p] -
                             nh_f_.d_view(n,nb,1)*omega[2][q][p]));
            }
          }
          na_(m,n,k,j,i,nb) = iszetaf*na1*uflux.d_view(n,nb,0)+na2*uflux.d_view(n,nb,1);
        }
      }
    });
  }

  // set transformation between normal and tetrad frame
  if (is_hydro_enabled || is_mhd_enabled) {
    auto norm_to_tet_ = norm_to_tet;
    par_for("norm_to_tet",DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
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

      // Minkowski metric
      Real eta[4][4] = {0.0};
      eta[0][0] = -1.0;
      eta[1][1] = 1.0;
      eta[2][2] = 1.0;
      eta[3][3] = 1.0;

      // Calculate normal-to-coordinate transformation
      Real norm_to_coord[4][4] = {0.0};
      Real alpha = 1.0/sqrt(-gupper[0][0]);
      norm_to_coord[0][0] = 1.0/alpha;
      norm_to_coord[1][0] = -alpha*gupper[0][1];
      norm_to_coord[2][0] = -alpha*gupper[0][2];
      norm_to_coord[3][0] = -alpha*gupper[0][3];
      norm_to_coord[1][1] = 1.0;
      norm_to_coord[2][2] = 1.0;
      norm_to_coord[3][3] = 1.0;

      for (int d1=0; d1<4; ++d1) {
        for (int d2=0; d2<4; ++d2) {
          norm_to_tet_(m,d1,d2,k,j,i) = 0.0;
          for (int p=0; p<4; ++p) {
            for (int q=0; q<4; ++q) {
              norm_to_tet_(m,d1,d2,k,j,i) += eta[d1][p]*e_cov[p][q]*norm_to_coord[q][d2];
            }
          }
        }
      }
    });
  }

  return;
}

} // namespace radiation
