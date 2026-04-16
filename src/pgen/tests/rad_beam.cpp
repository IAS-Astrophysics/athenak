//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_beam.cpp
//  \brief Beam test for radiation.  Also checks orthonormality of tetrad

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
#include "pgen/pgen.hpp"

// Prototypes for user-defined BCs
void ZeroIntensity(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::RadiationBeam(ParameterInput *pin)
//! \brief Checks tetrad is orthonormal.  Beam is introduced as rad_srcterm, so nothing
//! need be done here

void ProblemGenerator::RadiationBeam(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // User boundary function
  user_bcs_func = ZeroIntensity;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);
  auto &size = pmbp->pmb->mb_size;
  auto &flat = pmbp->pcoord->coord_data.is_minkowski;
  auto &spin = pmbp->pcoord->coord_data.bh_spin;
  auto &use_excise = pmbp->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmbp->pcoord->excision_floor;

  auto &tet_c_ = pmbp->prad->tet_c;
  par_for("check_tetrad",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        excised = true;
      }
    }

    if (!(excised)) {
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

      // Compute eta_alpha beta = g_mu nu e^mu_alpha e^nu_beta
      Real test_eta[4][4] = {0.0};
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          test_eta[alpha][beta] = 0.0;
          for (int mu=0; mu<4; ++mu) {
            for (int nu=0; nu<4; ++nu) {
              test_eta[alpha][beta] += (glower[mu][nu]*
                                        tet_c_(m,alpha,mu,k,j,i)*tet_c_(m,beta,nu,k,j,i));
            }
          }
        }
      }

      // Check for orthonormality
      for (int alpha=0; alpha<4; ++alpha) {
        for (int beta=0; beta<4; ++beta) {
          Real comp = 1.0;
          if   (alpha != beta) comp =  0.0;
          else if (alpha == 0) comp = -1.0;
          if (fabs(test_eta[alpha][beta] - comp) > 1.0e-13) {
            Kokkos::abort("Tetrad is not orthonormal!\n");
          }
        }
      }
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ZeroIntensity
//! \brief Sets boundary condition on surfaces of computational domain

void ZeroIntensity(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;

  // Determine if radiation is enabled
  bool is_radiation_enabled_ = (pm->pmb_pack->prad != nullptr) ? true : false;
  DvceArray5D<Real> i0_; int nang1;
  if (is_radiation_enabled_) {
    i0_ = pm->pmb_pack->prad->i0;
    nang1 = pm->pmb_pack->prad->prgeo->nangles - 1;
  }
  int nmb = pm->pmb_pack->nmb_thispack;

  // X1-Boundary
  if (is_radiation_enabled_) {
    // Set X1-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x1", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int n, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,is-i-1) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          i0_(m,n,k,j,ie+i+1) = 0.0;
        }
      }
    });
  }

  // X2-Boundary
  if (is_radiation_enabled_) {
    // Set X2-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x2", DevExeSpace(),0,(nmb-1),0,nang1,0,(n3-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,js-j-1,i) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          i0_(m,n,k,je+j+1,i) = 0.0;
        }
      }
    });
  }

  // x3-Boundary
  if (is_radiation_enabled_) {
    // Set x3-BCs on i0 if Meshblock face is at the edge of computational domain
    par_for("noinflow_rad_x3", DevExeSpace(),0,(nmb-1),0,nang1,0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int n, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ks-k-1,j,i) = 0.0;
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          i0_(m,n,ke+k+1,j,i) = 0.0;
        }
      }
    });
  }

  return;
}

