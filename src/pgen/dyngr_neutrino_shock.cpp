//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dyngr_neutrino_shock.cpp
//  \brief pgen for neutrino-mediated shock calculations

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"

// Forward declaration
template <class EOSPolicy, class ErrorPolicy>
void NeutrinoDominatedShock(Mesh *pmesh, ParameterInput* pin);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for a shock reflection
void ProblemGenerator::UserProblem(ParameterInput* pin, const bool restart) {
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_dynamical_relativistic) {
    std::cout
        << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl
        << "GRMHD shock reflection problem can only be run when <adm> block is present"
        << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (restart) {
    return;
  }

  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto* ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor>*>(pmbp->pdyngr);
  if (ptest_nqt != nullptr) {
    return NeutrinoDominatedShock<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor>(pmy_mesh_, pin);
  }

  auto* ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor>*>(
      pmbp->pdyngr);
  if (ptest_nlog != nullptr) {
    return NeutrinoDominatedShock<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor>(pmy_mesh_, pin);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

// Actual initial data solver
template <class EOSPolicy, class ErrorPolicy>
void NeutrinoDominatedShock(Mesh *pmesh, ParameterInput* pin) {
  // Get the EOS and set units to CGS
  MeshBlockPack* pmbp = pmesh->pmb_pack;
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
      static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr)
          ->eos.ps.GetEOSMutable();
  Real mb = eos.GetBaryonMass();

  MHDPrim1D wpt;
  wpt.d = pin->GetReal("problem", "rho");
  Real nb = wpt.d/mb;
  wpt.vx = pin->GetOrAddReal("problem", "vx", 0.0);
  wpt.vy = pin->GetOrAddReal("problem", "vy", 0.0);
  wpt.vz = pin->GetOrAddReal("problem", "vz", 0.0);
  Real temp = pin->GetReal("problem", "temp");
  wpt.bx = pin->GetOrAddReal("problem", "bx", 0.0);
  wpt.by = pin->GetOrAddReal("problem", "by", 0.0);
  wpt.bz = pin->GetOrAddReal("problem", "bz", 0.0);
  Real yq = pin->GetReal("problem", "yq");

  // compute Lorentz factor
  Real w_lorentz = 1.0 / sqrt(1.0 - (SQR(wpt.vx) + SQR(wpt.vy) + SQR(wpt.vz)));
  wpt.vx *= w_lorentz;
  wpt.vy *= w_lorentz;
  wpt.vz *= w_lorentz;

  // capture variables for the kernel
  auto& indcs = pmesh->mb_indcs;
  int& is = indcs.is;
  int& ie = indcs.ie;
  int& js = indcs.js;
  int& je = indcs.je;
  int& ks = indcs.ks;
  int& ke = indcs.ke;
  
  // capture grid arrays
  auto &w0_ = pmbp->pmhd->w0;
  auto &uradm1_ = pmbp->pradm1->u0;
  auto &nspecies_ = pmbp->pradm1->nspecies;
  if (nspecies_ == 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl;
    std::cout << "Neutrino radiation transport is required for this pgen!\n";
    abort();
  }

  auto &m1_params_ = pmbp->pradm1->params;
  auto &m1_nvars_ = pmbp->pradm1->nvars;

  // setup grid variables
  par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real Y[2] = {yq, 0.0};

    w0_(m,IDN,k,j,i) = wpt.d;
    w0_(m,IVX,k,j,i) = wpt.vx;
    w0_(m,IVY,k,j,i) = wpt.vy;
    w0_(m,IVZ,k,j,i) = wpt.vz;
    w0_(m,IPR,k,j,i) = eos.GetPressure(nb, temp, &Y[0]);
    w0_(m,IYF,k,j,i) = Y[0];
    w0_(m,IYF+1,k,j,i) = Y[1];

    for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
      int IDX = radiationm1::CombinedIdx(nuidx,M1_E_IDX,m1_nvars_);
      uradm1_(m,IDX,k,j,i) = m1_params_.rad_E_floor;
      IDX = radiationm1::CombinedIdx(nuidx,M1_N_IDX,m1_nvars_);
      uradm1_(m,IDX,k,j,i) = m1_params_.rad_N_floor;
    }
  });
    
  pmbp->padm->SetADMVariables(pmbp);
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);

  return;
}