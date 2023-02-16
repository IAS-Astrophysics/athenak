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
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void DChiThreshold(MeshBlockPack* pmbp) {
  auto &refine_flag_ = pmbp->pmesh->pmr->refine_flag;
  auto &ppos = pmbp->pz4c->ppos;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  int nx3 = indcs.nx3;
  int nx2 = indcs.nx2;
  int nx1 = indcs.nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;

  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int CHI = pmbp->pz4c->I_Z4c_chi;
  auto &u0 = pmbp->pz4c->u0;

  Real dchi_threshold = 0.2;

  par_for_outer("MaxDisToPuncture",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {

      Real team_ddmax;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& ddmax) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        Real d2 = SQR(u0(m,CHI,k,j,i+1) - u0(m,CHI,k,j,i-1))
                  + SQR(u0(m,CHI,k,j+1,i) - u0(m,CHI,k,j-1,i))
                  + SQR(u0(m,CHI,k+1,j,i) - u0(m,CHI,k-1,j,i));
        ddmax = fmax((sqrt(d2)/u0(m,CHI,k,j,i)), ddmax);
      },Kokkos::Max<Real>(team_ddmax));

      if (team_ddmax > dchi_threshold) {refine_flag_.d_view(m+mbs) = 1;}
      if (team_ddmax < 0.25*dchi_threshold) {refine_flag_.d_view(m+mbs) = -1;}
    });
}

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  user_ref_func = DChiThreshold;

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
