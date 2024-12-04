//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "z4c/z4c_amr.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"

#define SQ(X) ((X)*(X))

namespace z4c {

// set some parameters
Z4c_AMR::Z4c_AMR(ParameterInput *pin) {
  std::string ref_method = pin->GetOrAddString("z4c_amr", "method", "trivial");
  if (ref_method == "trivial") {
    method = Trivial;
  } else if (ref_method == "tracker") {
    method = Tracker;
  } else if (ref_method == "chi") {
    method = Chi;
    chi_thresh = pin->GetOrAddReal("z4c_amr", "chi_min", 0.2);
  } else if (ref_method == "dchi") {
    method = dChi;
    dchi_thresh = pin->GetOrAddReal("z4c_amr", "dchi_max", 0.1);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl;
    std::cout << "Unknown refinement strategy: " << ref_method << std::endl;
    std::exit(EXIT_FAILURE);
  }

  for (int nr = 0; nr < 16; ++nr) {
    std::string name = "radius_" + std::to_string(nr) + "_rad";
    if (pin->DoesParameterExist("z4c_amr", name)) {
      radius.push_back(pin->GetReal("z4c_amr", name));
      reflevel.push_back(pin->GetOrAddInteger(
          "z4c_amr", "radius_" + std::to_string(nr) + "_reflevel", -1));
    } else {
      break;
    }
  }
}

// 1: refines, -1: de-refines, 0: does nothing
void Z4c_AMR::Refine(MeshBlockPack *pmy_pack) {
  if (method == Tracker) {
    RefineTracker(pmy_pack);
  } else if (method == Chi) {
    RefineChiMin(pmy_pack);
  } else if (method == dChi) {
    RefineDchiMax(pmy_pack);
  }
  RefineRadii(pmy_pack);
}

// refine region within a certain distance from each compact object
void Z4c_AMR::RefineTracker(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  std::vector<int> flag;
  flag.reserve(pmbp->pz4c->ptracker.size());

  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    flag.clear();
    for (auto & pt : pmbp->pz4c->ptracker) {
      Real d2[8] = {
        SQ(x1min - pt->GetPos(0)) + SQ(x2min - pt->GetPos(1)) + SQ(x3min - pt->GetPos(2)),
        SQ(x1max - pt->GetPos(0)) + SQ(x2min - pt->GetPos(1)) + SQ(x3min - pt->GetPos(2)),
        SQ(x1min - pt->GetPos(0)) + SQ(x2max - pt->GetPos(1)) + SQ(x3min - pt->GetPos(2)),
        SQ(x1max - pt->GetPos(0)) + SQ(x2max - pt->GetPos(1)) + SQ(x3min - pt->GetPos(2)),
        SQ(x1min - pt->GetPos(0)) + SQ(x2min - pt->GetPos(1)) + SQ(x3max - pt->GetPos(2)),
        SQ(x1max - pt->GetPos(0)) + SQ(x2min - pt->GetPos(1)) + SQ(x3max - pt->GetPos(2)),
        SQ(x1min - pt->GetPos(0)) + SQ(x2max - pt->GetPos(1)) + SQ(x3max - pt->GetPos(2)),
        SQ(x1max - pt->GetPos(0)) + SQ(x2max - pt->GetPos(1)) + SQ(x3max - pt->GetPos(2)),
      };
      Real dmin2 = *std::min_element(&d2[0], &d2[8]);
      bool iscontained =
        (pt->GetPos(0) >= x1min && pt->GetPos(0) <= x1max) &&
        (pt->GetPos(1) >= x2min && pt->GetPos(1) <= x2max) &&
        (pt->GetPos(2) >= x3min && pt->GetPos(2) <= x3max);

      if (dmin2 < SQ(pt->GetRadius()) || iscontained) {
        if (pt->GetReflevel() < 0 || level < pt->GetReflevel()) {
          flag.push_back(1);
        } else if (level == pt->GetReflevel()) {
          flag.push_back(0);
        } else {
          flag.push_back(-1);
        }
      } else {
        flag.push_back(-1);
      }
    }
    refine_flag.h_view(m + mbs) = *std::max_element(flag.begin(), flag.end());
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// refine based on min{chi}
void Z4c_AMR::RefineChiMin(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->pz4c->u0;
  int I_Z4C_CHI  = pmbp->pz4c->I_Z4C_CHI;
  // note: we need this to prevent capture by this in the lambda expr.
  auto chi_thresh = this->chi_thresh;

  par_for_outer(
    "Z4c_AMR::ChiMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dmin;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &dmin) {
          int k = (idx) / nji;
          int j = (idx - k * nji) / nx1;
          int i = (idx - k * nji - j * nx1) + is;
          j += js;
          k += ks;
          dmin = fmin(u0(m, I_Z4C_CHI, k, j, i), dmin);
        },
        Kokkos::Min<Real>(team_dmin));

      if (team_dmin < chi_thresh) {
        refine_flag.d_view(m + mbs) = 1;
      }
      if (team_dmin > 1.25 * chi_thresh) {
        refine_flag.d_view(m + mbs) = -1;
      }
    });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

// refine based on max{dchi}
void Z4c_AMR::RefineDchiMax(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs       = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji  = nx2 * nx1;
  auto &u0       = pmbp->pz4c->u0;
  int I_Z4C_CHI  = pmbp->pz4c->I_Z4C_CHI;
  // note: we need this to prevent capture by this in the lambda expr.
  auto dchi_thresh = this->dchi_thresh;

  par_for_outer(
    "Z4c_AMR::ChiMin", DevExeSpace(), 0, 0, 0, (nmb - 1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dmax;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &dmax) {
          int k = (idx) / nji;
          int j = (idx - k * nji) / nx1;
          int i = (idx - k * nji - j * nx1) + is;
          j += js;
          k += ks;
          Real d2 = SQR(u0(m,I_Z4C_CHI,k,j,i+1) - u0(m,I_Z4C_CHI,k,j,i-1));
          d2 += SQR(u0(m,I_Z4C_CHI,k,j+1,i) - u0(m,I_Z4C_CHI,k,j-1,i));
          d2 += SQR(u0(m,I_Z4C_CHI,k+1,j,i) - u0(m,I_Z4C_CHI,k-1,j,i));
          dmax = fmax((sqrt(d2)), dmax);
        },
        Kokkos::Max<Real>(team_dmax));

      if (team_dmax > dchi_thresh) {
        refine_flag.d_view(m + mbs) = 1;
      }
      if (team_dmax < 0.5 * dchi_thresh) {
        refine_flag.d_view(m + mbs) = -1;
      }
    });

  // sync host and device
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
}

// Enforce some minimum resolution within a certain spherical region
void Z4c_AMR::RefineRadii(MeshBlockPack *pmbp) {
  Mesh *pmesh       = pmbp->pmesh;
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &size        = pmbp->pmb->mb_size;
  int nmb           = pmbp->nmb_thispack;
  int mbs           = pmesh->gids_eachrank[global_variable::my_rank];

  for (int m = 0; m < nmb; ++m) {
    // current refinement level
    int level = pmesh->lloc_eachmb[m + mbs].level - pmesh->root_level;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    Real r2[8] = {
      SQ(x1min) + SQ(x2min) + SQ(x3min),
      SQ(x1max) + SQ(x2min) + SQ(x3min),
      SQ(x1min) + SQ(x2max) + SQ(x3min),
      SQ(x1max) + SQ(x2max) + SQ(x3min),
      SQ(x1min) + SQ(x2min) + SQ(x3max),
      SQ(x1max) + SQ(x2min) + SQ(x3max),
      SQ(x1min) + SQ(x2max) + SQ(x3max),
      SQ(x1max) + SQ(x2max) + SQ(x3max),
    };
    Real rmin2 = *std::min_element(&r2[0], &r2[8]);

    for (int ir = 0; ir < radius.size(); ++ir) {
      if (rmin2 < SQ(radius[ir])) {
        if (level < reflevel[ir]) {
          refine_flag.h_view(m + mbs) = 1;
        } else if (level == reflevel[ir] && refine_flag.h_view(m + mbs) == -1) {
          refine_flag.h_view(m + mbs) = 0;
        }
      }
    }
  }

  // sync host and device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

} // namespace z4c
