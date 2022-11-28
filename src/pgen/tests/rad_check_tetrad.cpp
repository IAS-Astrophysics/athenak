//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_check_tetrad.cpp
//  \brief Unit test to ensure tetrad is orthonormal

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation/radiation.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//  \brief Checks orthonormality of tetrad

void ProblemGenerator::CheckOrthonormalTetrad(ParameterInput *pin, const bool restart) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
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
