//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/cartesian_ks.hpp"


void ADMKerrSchild(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Kerr-Schild test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMKerrSchild(pmbp, pin);
  // pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  std::cout<<"Kerr-Schild initialized."<<std::endl;

  return;
}

// inlined spherical Kerr-Schild r evaluated at CKS x1, x2, x3
KOKKOS_INLINE_FUNCTION
Real KSRX(const Real x1, const Real x2, const Real x3, const Real a) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  return sqrt((SQR(rad)-SQR(a)+sqrt(SQR(SQR(rad)-SQR(a))+4.0*SQR(a)*SQR(x3)))/2.0);
}


//----------------------------------------------------------------------------------------
//! \fn void ADMKerrSchild(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to Kerr-Schild solution
void ADMKerrSchild(MeshBlockPack *pmbp, ParameterInput *pin) {
  // read bh spin
  Real a = pin->GetOrAddReal("problem", "bh_spin", 0.);
  Real excise_ratio = pin->GetOrAddReal("problem", "excise_ratio", 0.5);

  // set transition radius
  // transit from Kerr-Schild to Minkolsky from 2rH/3 to rH/1
  Real rH = 1 + sqrt(1 - SQR(a));
  Real rexci = excise_ratio*rH;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
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

    Real r = KSRX(x1v,x2v,x3v,a);

    ComputeADMDecomposition(x1v, x2v, x3v, false, a,
      &adm.alpha(m,k,j,i),
      &adm.beta_u(m,0,k,j,i), &adm.beta_u(m,1,k,j,i), &adm.beta_u(m,2,k,j,i),
      &adm.psi4(m,k,j,i),
      &adm.g_dd(m,0,0,k,j,i), &adm.g_dd(m,0,1,k,j,i), &adm.g_dd(m,0,2,k,j,i),
      &adm.g_dd(m,1,1,k,j,i), &adm.g_dd(m,1,2,k,j,i), &adm.g_dd(m,2,2,k,j,i),
      &adm.vK_dd(m,0,0,k,j,i), &adm.vK_dd(m,0,1,k,j,i), &adm.vK_dd(m,0,2,k,j,i),
      &adm.vK_dd(m,1,1,k,j,i), &adm.vK_dd(m,1,2,k,j,i), &adm.vK_dd(m,2,2,k,j,i));
    if (r<rexci) {
      // weight function to attenuate to zero
      Real w = exp(1/(SQR(r)-SQR(rexci))+1/SQR(rexci));
      Real invw = 1-w;
      adm.alpha(m,k,j,i) *= invw;
      adm.alpha(m,k,j,i) += w * 1;
      for (int a = 0; a <= 2; ++a) {
        adm.beta_u(m,a,k,j,i) *= invw;
        for (int b = a; b<=2; ++b) {
          adm.g_dd(m,a,b,k,j,i) *= invw;
          adm.vK_dd(m,a,b,k,j,i) *= invw;
          if (a == b) {
            adm.g_dd(m,a,b,k,j,i) += w * 1;
          }
        }
      }
    }
  });
}

// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
