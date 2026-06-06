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
#include "dyn_radiation/dyn_radiation.hpp"
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
    // Radiation
    int nang1 = -1;
    bool use_adm_radiation = false;
    DualArray2D<Real> nh_c_;
    DvceArray6D<Real> tet_c_;
    DvceArray6D<Real> tetcov_c_;
    DualArray1D<Real> domega;
    DvceArray5D<Real> i0_;
    DvceArray4D<Real> sqrt_detg_c_;
    if (pmbp->prad != nullptr) {
      nang1 = pmbp->prad->prgeo->nangles - 1;
      nh_c_ = pmbp->prad->nh_c;
      tet_c_ = pmbp->prad->tet_c;
      tetcov_c_ = pmbp->prad->tetcov_c;
      domega = pmbp->prad->prgeo->solid_angles;
      i0_ = pmbp->prad->i0;
    } else {
      nang1 = pmbp->pdynrad->prgeo->nangles - 1;
      use_adm_radiation = pmbp->pdynrad->use_adm_geometry;
      nh_c_ = pmbp->pdynrad->nh_c;
      tet_c_ = pmbp->pdynrad->tet_c;
      tetcov_c_ = pmbp->pdynrad->tetcov_c;
      domega = pmbp->pdynrad->prgeo->solid_angles;
      i0_ = pmbp->pdynrad->i0;
      sqrt_detg_c_ = pmbp->pdynrad->sqrt_detg_c;
    }

    par_for("moments",DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // coordinate component n^0
      Real n0 = tet_c_(m,0,0,k,j,i);
      Real intensity_norm = use_adm_radiation ? sqrt_detg_c_(m,k,j,i) : n0;

      // set coordinate frame component
      dvars(m,index,k,j,i) = 0.0;
      for (int n=0; n<=nang1; ++n) {
        Real nmun1 = 0.0; Real nmun2 = 0.0; Real n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          nmun1 += tet_c_   (m,d,0,k,j,i)*nh_c_.d_view(n,d);
          nmun2 += tet_c_   (m,d,0,k,j,i)*nh_c_.d_view(n,d);
          n_0   += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(n,d);
        }
        Real intensity = i0_(m,n,k,j,i)/intensity_norm;
        if (!(use_adm_radiation)) { intensity /= n_0; }
        dvars(m,index,k,j,i) += nmun1*nmun2*intensity*domega.d_view(n);
      }
    });
  }
  return;
}
