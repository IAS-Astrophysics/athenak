//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coupling.cpp
//! \brief Apply radiative heating/cooling to the fluid energy equation each RK stage:
//!   u0(IEN) += beta_dt * Q_rad   (Davis 2012 Eq. 26 with Q from Eq. 27).

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "nr_radiation/nr_radiation.hpp"

namespace nr_radiation {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus SC::AddQrad
//! \brief Inserted after hydro/mhd RKUpdate, before HydroSrcTerms / MHDSrcTerms.

TaskStatus SC::AddQrad(Driver *pdrive, int stage) {
  if (!affect_fluid) return TaskStatus::complete;

  Real beta_dt = (pdrive->beta[stage-1]) * (pmy_pack->pmesh->dt);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto qrad_ = qrad;
  DvceArray5D<Real> u0;   // exactly one fluid exists when this task is registered
  if (pmy_pack->phydro != nullptr) {
    u0 = pmy_pack->phydro->u0;
  } else {
    u0 = pmy_pack->pmhd->u0;
  }
  par_for("sc_addqrad", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IEN,k,j,i) += beta_dt * qrad_(m,k,j,i);
  });
  return TaskStatus::complete;
}

}  // namespace nr_radiation
