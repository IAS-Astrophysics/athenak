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

#include <Kokkos_Random.hpp>

// Forward declaration
template <class EOSPolicy, class ErrorPolicy>
void NeutrinoDominatedShock(Mesh *pmesh, ParameterInput* pin);

// Prototype for user-defined history
void NeutrinoShockHistory(HistoryData *pdata, Mesh *pm);

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

  user_hist_func = &NeutrinoShockHistory;

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
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);

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
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real Y[2] = {yq, 0.0};

    // apply white-noise density perturbation
    auto rand_gen = rand_pool64.get_state();
    Real rval = 1.0 + pert_amp*(rand_gen.frand() - 0.5);
    w0_(m,IDN,k,j,i) = wpt.d * rval;
    rand_pool64.free_state(rand_gen);

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

// History function: maximum |B^i| (each Cartesian component) over the domain
void NeutrinoShockHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 4;
  pdata->label[0] = "bx-max";
  pdata->label[1] = "by-max";
  pdata->label[2] = "bz-max";
  pdata->label[3] = "rhoy5-max";

  // capture class variables for kernel
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &bcc0_ = pm->pmb_pack->pmhd->bcc0;
  auto &adm = pm->pmb_pack->padm->adm;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real bx_max = 0.0;
  Real by_max = 0.0;
  Real bz_max = 0.0;
  Real rhoy5_max = 0.0;
  Kokkos::parallel_reduce("NeutrinoShockHistorySums",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mb_bx_max, Real &mb_by_max, Real &mb_bz_max,
                Real &mb_rhoy5_max) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    // bcc0 stores the densitized B-field; divide by sqrt(det g) to get B^i
    Real gamma = sqrt(
        adm::SpatialDet(adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                        adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                        adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i)));
    Real bx = fabs(bcc0_(m,IBX,k,j,i)/gamma);
    Real by = fabs(bcc0_(m,IBY,k,j,i)/gamma);
    Real bz = fabs(bcc0_(m,IBZ,k,j,i)/gamma);
    Real rhoy5 = fabs(w0_(m,IDN,k,j,i) * w0_(m,IYF+1,k,j,i));

    mb_bx_max = fmax(bx, mb_bx_max);
    mb_by_max = fmax(by, mb_by_max);
    mb_bz_max = fmax(bz, mb_bz_max);
    mb_rhoy5_max = fmax(rhoy5, mb_rhoy5_max);
  }, Kokkos::Max<Real>(bx_max), Kokkos::Max<Real>(by_max), Kokkos::Max<Real>(bz_max),
     Kokkos::Max<Real>(rhoy5_max));

  // Currently AthenaK only supports MPI_SUM operations between ranks, but we need
  // MPI_MAX here. This is a cheap hack to make it work as intended.
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &bx_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &by_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &bz_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &rhoy5_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&bx_max, &bx_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&by_max, &by_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bz_max, &bz_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rhoy5_max, &rhoy5_max, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    bx_max = 0.0;
    by_max = 0.0;
    bz_max = 0.0;
    rhoy5_max = 0.0;
  }
#endif

  pdata->hdata[0] = bx_max;
  pdata->hdata[1] = by_max;
  pdata->hdata[2] = bz_max;
  pdata->hdata[3] = rhoy5_max;
}