//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file radiation_m1_bcs.cpp
//  \brief boundary conditions for grey M1

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "radiation_m1/radiation_m1.hpp"

//----------------------------------------------------------------------------------------
//! \!fn void BoundaryValues::RadiationBCs()
//! \brief Apply physical boundary conditions for radiation at faces of MB which
//! are at the edge of the computational domain

void MeshBoundaryValues::RadiationM1BCs(MeshBlockPack *ppack,
                                        DualArray2D<Real> i_in,
                                        DvceArray5D<Real> i0) {
  // loop over all MeshBlocks in this MeshBlockPack
  auto &pm = ppack->pmesh;
  auto &indcs = ppack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  auto &nspecies_ = ppack->pmesh->pmb_pack->pradm1->nspecies;
  auto &nvars_ = ppack->pmesh->pmb_pack->pradm1->nvars;
  auto &mb_bcs = ppack->pmb->mb_bcs;
  auto &params_ = ppack->pmesh->pmb_pack->pradm1->params;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int nmb = ppack->nmb_thispack;

  Real rad_E_floor = 1e-14; //@TODO: get actual floor value
  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    int &is = indcs.is;
    int &ie = indcs.ie;
    par_for(
      "radiationm1bc_x1", DevExeSpace(), 0, (nmb - 1), 0, (nspecies_ - 1), 0,
      (n3 - 1), 0, (n2 - 1), KOKKOS_LAMBDA(int m, int nuidx, int k, int j) {
          // apply physical boundaries to inner_x1
        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::reflect:
            for (int i = 0; i < ng; ++i) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, is - i - 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, is + i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, is - i - 1) = -i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, is + i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, is - i - 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, is + i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, is - i - 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, is + i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, is - i - 1) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, is + i);
              }
            }
            break;
          case BoundaryFlag::outflow:
            for (int i = 0; i < ng; ++i) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, is - i - 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, is);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, is - i - 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, is);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, is - i - 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, is);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, is - i - 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, is);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, is - i - 1) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, is);
              }
            }
            break;
          case BoundaryFlag::vacuum:
            for (int i = 0; i < ng; ++i) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, is - i - 1) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, is - i - 1) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, is - i - 1) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, is - i - 1) = params_.rad_E_floor;
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, is - i - 1) = params_.rad_N_floor;
              }
            }
            break;
          default:
            break;
            
        }

          // apply physical boundaries to outer_x1
        switch (mb_bcs.d_view(m, BoundaryFace::outer_x1)) {
          case BoundaryFlag::reflect:
            for (int i = 0; i < ng; ++i) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, ie + i + 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, ie - i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, ie + i + 1) = -i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, ie - i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, ie + i + 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, ie - i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, ie + i + 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, ie - i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, ie + i + 1) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, ie - i);
              }
            }
            break;
          case BoundaryFlag::outflow:
            for (int i = 0; i < ng; ++i) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, ie + i + 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, ie);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, ie + i + 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, ie);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, ie + i + 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, ie);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, ie + i + 1) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, ie);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, ie + i + 1) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, ie);
              }
            }
            break;
          case BoundaryFlag::vacuum:
            for (int i = 0; i < ng; ++i) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, ie + i + 1) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, ie + i + 1) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, ie + i + 1) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, ie + i + 1) = params_.rad_E_floor;
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, ie + i + 1) = params_.rad_N_floor;
              }
            }
            break;
          default:
            break;
        }
      });
  }
  if (pm->one_d)
    return;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    int &js = indcs.js;
    int &je = indcs.je;
    par_for(
      "radiationbc_x2", DevExeSpace(), 0, (nmb - 1), 0, (nspecies_ - 1), 0,
      (n3 - 1), 0, (n1 - 1), KOKKOS_LAMBDA(int m, int nuidx, int k, int i) {
          // apply physical boundaries to inner_x2
      switch (mb_bcs.d_view(m, BoundaryFace::inner_x2)) {
          case BoundaryFlag::reflect:
            for (int j = 0; j < ng; ++j) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, js - j - 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, js + j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, js - j - 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, js + j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, js - j - 1, i) = -i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, js + j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, js - j - 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, js + j, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, js - j - 1, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, js + j, i);
              }
            }
            break;
          case BoundaryFlag::outflow:
            for (int j = 0; j < ng; ++j) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, js - j - 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, js, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, js - j - 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, js, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, js - j - 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, js, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, js - j - 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, js, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, js - j - 1, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, js, i);
              }
            }
          case BoundaryFlag::vacuum:
            for (int j = 0; j < ng; ++j) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, js - j - 1, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, js - j - 1, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, js - j - 1, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, js - j - 1, i) = params_.rad_E_floor;
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, js - j - 1, i) = params_.rad_N_floor;
              }
            }
            break;
          default:
            break;
        }

          // apply physical boundaries to outer_x2
        switch (mb_bcs.d_view(m, BoundaryFace::outer_x2)) {
          case BoundaryFlag::reflect:
            for (int j = 0; j < ng; ++j) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, je + j + 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, je - j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, je + j + 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, je - j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, je + j + 1, i) = -i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, je - j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, je + j + 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, je - j, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, je + j + 1, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, je - j, i);
              }
            }
            break;
          case BoundaryFlag::outflow:
            for (int j = 0; j < ng; ++j) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, je + j + 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, je, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, je + j + 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, je, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, je + j + 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, je, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, je + j + 1, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, je, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, je + j + 1, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, je, i);
              }
            }
            break;
          case BoundaryFlag::vacuum:
            for (int j = 0; j < ng; ++j) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), k, je + j + 1, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, je + j + 1, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, je + j + 1, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, je + j + 1, i) = params_.rad_E_floor;
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), k, je + j + 1, i) = params_.rad_N_floor;
              }
            }
            break;
          default:
            break;
        }
      });
  }
  if (pm->two_d)
    return;

  // only apply BCs if not periodic
  if (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic)
    return;
  int &ks = indcs.ks;
  int &ke = indcs.ke;
  par_for(
    "radiationbc_x3", DevExeSpace(), 0, (nmb - 1), 0, (nspecies_ - 1), 0, (n2 - 1),
    0, (n1 - 1), KOKKOS_LAMBDA(int m, int nuidx, int j, int i) {
        // apply physical boundaries to inner_x3
      switch (mb_bcs.d_view(m, BoundaryFace::inner_x3)) {
        case BoundaryFlag::reflect:
          for (int k = 0; k < ng; ++k) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ks - k - 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ks + k, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ks - k - 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ks + k, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ks - k - 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ks + k, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ks - k - 1, j, i) = -i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ks + k, j, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ks - k - 1, j, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ks + k, j, i);
              }
            }
          break;
        case BoundaryFlag::outflow:
          for (int k = 0; k < ng; ++k) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ks - k - 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ks, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ks - k - 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ks, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ks - k - 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ks, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ks - k - 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ks, j, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ks - k - 1, j, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ks, j, i);
              }
            }
          break;
        case BoundaryFlag::vacuum:
          for (int k = 0; k < ng; ++k) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ks - k - 1, j, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ks - k - 1, j, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ks - k - 1, j, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ks - k - 1, j, i) = params_.rad_E_floor;
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ks - k - 1, j, i) = params_.rad_N_floor;
              }
            }
          break;
        default:
          break;
      }

        // apply physical boundaries to outer_x3
      switch (mb_bcs.d_view(m, BoundaryFace::outer_x3)) {
        case BoundaryFlag::reflect:
          for (int k = 0; k < ng; ++k) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ke + k + 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ke - k, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ke + k + 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ke - k, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ke + k + 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ke - k, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ke + k + 1, j, i) = -i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ke - k, j, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ke + k + 1, j, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ke - k, j, i);
              }
            }
          break;
        case BoundaryFlag::outflow:
          for (int k = 0; k < ng; ++k) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ke + k + 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ke, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ke + k + 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ke, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ke + k + 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ke, j, i);
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ke + k + 1, j, i) = i0(m,
               radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ke, j, i);
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ke + k + 1, j, i) = i0(m,
                 radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ke, j, i);
              }
            }
          break;
        case BoundaryFlag::vacuum:
          for (int k = 0; k < ng; ++k) {
              i0(m, radiationm1::CombinedIdx(nuidx, M1_E_IDX, nvars_), ke + k + 1, j, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FX_IDX, nvars_), ke + k + 1, j, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FY_IDX, nvars_), ke + k + 1, j, i) = params_.rad_E_floor;
              i0(m, radiationm1::CombinedIdx(nuidx, M1_FZ_IDX, nvars_), ke + k + 1, j, i) = params_.rad_E_floor;
              if (nspecies_ > 1) {
                i0(m, radiationm1::CombinedIdx(nuidx, M1_N_IDX, nvars_), ke + k + 1, j, i) = params_.rad_N_floor;
              }
            }
          break;
        default:
          break;
      }
    });

  return;
}
