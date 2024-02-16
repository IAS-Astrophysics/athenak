//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <iostream>
#include <limits>
#include <sstream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "z4c/z4c_puncture_tracker.hpp"

namespace z4c {

// set some parameters
Z4c_AMR::Z4c_AMR(Z4c *z4c, ParameterInput *pin): pz4c(z4c), pin(pin) {
  // available methods: "Linf_box_in_box", "L2_sphere_in_sphere", and
  // "fd_truncation_error"
  ref_method     = pin->GetOrAddString("z4c_amr", "method", "Linf_box_in_box");
  x1max          = pin->GetReal("mesh", "x1max");
  x1min          = pin->GetReal("mesh", "x1min");
  half_initial_d = pin->GetOrAddReal("problem", "par_b", 1.);
}

// 1: refines, -1: de-refines, 0: does nothing
void Z4c_AMR::Refine(MeshBlockPack *pmy_pack) {
  // use box in box method
  if (ref_method == "Linf_box_in_box") {
    LinfBoxInBox(pmy_pack);
  // use L-2 norm as a criteria for refinement
  } else if (ref_method == "L2_sphere_in_sphere") {
    L2SphereInSphere(pmy_pack);
  } else {
    std::stringstream msg;
    msg << "No such option for z4c/refinement" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// Mimicking box in box refinement with Linf
void Z4c_AMR::LinfBoxInBox(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  Real L            = (x1max - x1min) / 2. - half_initial_d;

  for (int m = 0; m < nmb; ++m) {
    // level
    int level = pmbp->pmesh->lloc_eachmb->level - pmbp->pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    Real xv[24];

    // Needed to calculate coordinates of vertices of a block with same center
    // but edge of 1/8th of the original size
    Real x1sum_sup = (5 * x1max + 3 * x1min) / 8.;
    Real x1sum_inf = (3 * x1max + 5 * x1min) / 8.;
    Real x2sum_sup = (5 * x2max + 3 * x2min) / 8.;
    Real x2sum_inf = (3 * x2max + 5 * x2min) / 8.;
    Real x3sum_sup = (5 * x3max + 3 * x3min) / 8.;
    Real x3sum_inf = (3 * x3max + 5 * x3min) / 8.;

    xv[0] = x1sum_sup;
    xv[1] = x2sum_sup;
    xv[2] = x3sum_sup;

    xv[3] = x1sum_sup;
    xv[4] = x2sum_sup;
    xv[5] = x3sum_inf;

    xv[6] = x1sum_sup;
    xv[7] = x2sum_inf;
    xv[8] = x3sum_sup;

    xv[9]  = x1sum_sup;
    xv[10] = x2sum_inf;
    xv[11] = x3sum_inf;

    xv[12] = x1sum_inf;
    xv[13] = x2sum_sup;
    xv[14] = x3sum_sup;

    xv[15] = x1sum_inf;
    xv[16] = x2sum_sup;
    xv[17] = x3sum_inf;

    xv[18] = x1sum_inf;
    xv[19] = x2sum_inf;
    xv[20] = x3sum_sup;

    xv[21] = x1sum_inf;
    xv[22] = x2sum_inf;
    xv[23] = x3sum_inf;

    // Min distance between the two punctures
    Real d = std::numeric_limits<Real>::max();
    for (auto ptracker : pmbp->pz4c_ptracker) {
      // Abs difference
      Real diff;
      // Max norm_inf
      Real dmin_punct = std::numeric_limits<Real>::max();
      for (int i_vert = 0; i_vert < 8; ++i_vert) {
        // Norm_inf
        Real norm_inf = -1;
        for (int i_diff = 0; i_diff < 3; ++i_diff) {
          diff = std::abs(ptracker->GetPos(i_diff) - xv[i_vert * 3 + i_diff]);
          if (diff > norm_inf) {
            norm_inf = diff;
          }
        }
        // Calculate minimum of the distances of the 8 vertices above
        if (dmin_punct > norm_inf) {
          dmin_punct = norm_inf;
        }
      }
      // Calculate minimum of the closest between the n punctures
      if (d > dmin_punct) {
        d = dmin_punct;
      }
    }
    Real ratio = L / d;
    if (ratio < 1) {
      refine_flag.d_view(m + mbs) = -1;
      continue;
    }

    // Calculate level that the block should be in, given a box-in-box
    // theoretical structure of the grid
    Real th_level = std::floor(std::log2(ratio));
    if (th_level > level) {
      refine_flag.d_view(m + mbs) = 1;
    } else if (th_level < level) {
      refine_flag.d_view(m + mbs) = -1;
    }
  } // for (int m=0; m < nmb; ++m)
}

// L-2 norm for refinement kind of like sphere in sphere
void Z4c_AMR::L2SphereInSphere(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  Real L            = (x1max - x1min) / 2. - half_initial_d;

  for (int m = 0; m < nmb; ++m) {
    // level
    int level =
      pmbp->pmesh->lloc_eachmb[m + mbs].level - pmbp->pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;
    Real xv[24];

    // Needed to calculate coordinates of vertices of a block with same center
    // but edge of 1/8th of the original size
    Real x1sum_sup = (5 * x1max + 3 * x1min) / 8.;
    Real x1sum_inf = (3 * x1max + 5 * x1min) / 8.;
    Real x2sum_sup = (5 * x2max + 3 * x2min) / 8.;
    Real x2sum_inf = (3 * x2max + 5 * x2min) / 8.;
    Real x3sum_sup = (5 * x3max + 3 * x3min) / 8.;
    Real x3sum_inf = (3 * x3max + 5 * x3min) / 8.;

    xv[0] = x1sum_sup;
    xv[1] = x2sum_sup;
    xv[2] = x3sum_sup;

    xv[3] = x1sum_sup;
    xv[4] = x2sum_sup;
    xv[5] = x3sum_inf;

    xv[6] = x1sum_sup;
    xv[7] = x2sum_inf;
    xv[8] = x3sum_sup;

    xv[9]  = x1sum_sup;
    xv[10] = x2sum_inf;
    xv[11] = x3sum_inf;

    xv[12] = x1sum_inf;
    xv[13] = x2sum_sup;
    xv[14] = x3sum_sup;

    xv[15] = x1sum_inf;
    xv[16] = x2sum_sup;
    xv[17] = x3sum_inf;

    xv[18] = x1sum_inf;
    xv[19] = x2sum_inf;
    xv[20] = x3sum_sup;

    xv[21] = x1sum_inf;
    xv[22] = x2sum_inf;
    xv[23] = x3sum_inf;

    // Min distance between the two punctures
    Real d = std::numeric_limits<Real>::max();
    for (auto ptracker : pmbp->pz4c_ptracker) {
      // square difference
      Real diff;

      Real dmin_punct = std::numeric_limits<Real>::max();
      for (int i_vert = 0; i_vert < 8; ++i_vert) {
        // Norm_L-2
        Real norm_L2 = 0;
        for (int i_diff = 0; i_diff < 3; ++i_diff) {
          diff = (ptracker->GetPos(i_diff) - xv[i_vert * 3 + i_diff])
            * (ptracker->GetPos(i_diff) - xv[i_vert * 3 + i_diff]);
          norm_L2 += diff;
        }
        // Compute the L-2 norm
        norm_L2 = std::sqrt(norm_L2);

        // Calculate minimum of the distances of the 8 vertices above
        if (dmin_punct > norm_L2) {
          dmin_punct = norm_L2;
        }
      }
      // Calculate minimum of the closest between the n punctures
      if (d > dmin_punct) {
        d = dmin_punct;
      }
    }
    Real ratio = L / d;

    if (ratio < 1) {
      refine_flag.d_view(m + mbs) = -1;
      continue;
    }

    // Calculate level that the block should be in, given a box-in-box
    // theoretical structure of the grid
    Real th_level = std::floor(std::log2(ratio));
    if (th_level >= level) {
      refine_flag.d_view(m + mbs) = 1;
    } else {
      refine_flag.d_view(m + mbs) = -1;
    }
  } // for (int m=0; m < nmb; ++m)
}
} // namespace z4c
