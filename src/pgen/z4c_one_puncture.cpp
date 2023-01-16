//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain
//

#include <algorithm>
#include <cmath>
#include <sstream>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void PunctureLocationRef(MeshBlockPack* pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  auto &ppos = pmbp->pz4c->ppos;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  int nx3 = indcs.nx3;
  int nx2 = indcs.nx2;
  int nx1 = indcs.nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  Real dist_thresh = 5;

  par_for_outer("MaxDisToPuncture",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real team_dmin=100000.0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& dmin) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1);

        Real x1v = CellCenterX(i, nx1, x1min, x1max);
        Real x2v = CellCenterX(j, nx2, x2min, x2max);
        Real x3v = CellCenterX(k, nx3, x3min, x3max);

        Real d2punc = sqrt((x1v-ppos[0])*(x1v-ppos[0])+
                           (x2v-ppos[1])*(x2v-ppos[1])+
                           (x3v-ppos[2])*(x3v-ppos[2]));
        dmin = fmin(d2punc, dmin);
      },Kokkos::Min<Real>(team_dmin));
      if (team_dmin < dist_thresh) {refine_flag.d_view(m) = 1;}
      if (team_dmin > dist_thresh) {refine_flag.d_view(m) = -1;}
    });
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  user_ref_func = PunctureLocationRef;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  pmbp->pz4c->ADMOnePuncture(pmbp, pin);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  std::cout<<"OnePuncture initialized."<<std::endl;


  return;
}
