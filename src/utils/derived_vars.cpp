//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file derived_variables.cpp
//! \brief Calculates derived variables used for outputs, mesh refinement criteria, etc.
//! Variables are only calculated over active zones (ghost zones excluded).

#include <iostream>
#include <sstream>
#include <string>   // std::string, to_string()

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_tetrad.hpp"
#include "particles/particles.hpp"
#include "utils/current.hpp"

//----------------------------------------------------------------------------------------
//! \fn  ComputeDerivedVariable()
//! \brief Returns derived variable(s) specified by "name" in dvars(m,n,k,j,i) array
//! starting at n=index

void ComputeDerivedVariable(std::string name, int index, MeshBlockPack* pmbp,
                            DvceArray5D<Real> dvars) {
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmbp->pmesh->mb_indcs;
//  int &ng = indcs.ng;
//  int n1 = indcs.nx1 + 2*ng;
//  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
//  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;

  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  // radiation coordinate frame energy density R^0^0
  if (name.compare("rad_coord_e") == 0) {
    // Coordinates
    auto &coord = pmbp->pcoord->coord_data;
    bool &flat = coord.is_minkowski;
    Real &spin = coord.bh_spin;

    // Radiation
    int nang1 = pmbp->prad->prgeo->nangles - 1;
    auto nh_c_ = pmbp->prad->nh_c;
    auto tet_c_ = pmbp->prad->tet_c;
    auto tetcov_c_ = pmbp->prad->tetcov_c;
    auto domega = pmbp->prad->prgeo->solid_angles;
    auto i0_ = pmbp->prad->i0;

    par_for("moments",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
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

      // Extract components of metric
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);

      // coordinate component n^0
      Real n0 = tet_c_(m,0,0,k,j,i);

      // set coordinate frame component
      dvars(m,index,k,j,i) = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real nmun1 = 0.0; Real nmun2 = 0.0; Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          nmun1 += tet_c_   (m,d,0,k,j,i)*nh_c_.d_view(n,d);
          nmun2 += tet_c_   (m,d,0,k,j,i)*nh_c_.d_view(n,d);
          n_0   += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);
        }
        dvars(m,index,k,j,i) += (nmun1*nmun2*(i0_(m,n,k,j,i)/(n0*n_0))*domega.d_view(n));
      }
    });
  }
  return;
}
