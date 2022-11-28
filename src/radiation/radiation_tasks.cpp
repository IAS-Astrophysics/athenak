//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tasks.cpp
//  \brief implementation of functions that control Hydro tasks in the four task lists:
//  stagestart_tl, stagerun_tl, stageend_tl, operatorsplit_tl (currently not used)

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::AssembleRadiationTasks
//  \brief Adds radiation tasks to stage start/run/end task lists
//  Called by MeshBlockPack::AddPhysicsModules() function after Radiation constrctr
//
//  Stage start tasks are those that must be cmpleted over all MeshBlocks before EACH
//  stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//
//  Stage run tasks are those performed in EACH stage
//
//  Stage end tasks are those that can only be cmpleted after all the stage run tasks are
//  finished over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking
//  sends, etc.

void Radiation::AssembleRadiationTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);
  hydro::Hydro *phyd = pmy_pack->phydro;
  mhd::MHD *pmhd = pmy_pack->pmhd;
  if (pmhd != nullptr && !(fixed_fluid)) {  // radiation MHD
    // start task list
    id.rad_irecv = start.AddTask(&Radiation::InitRecv, this, none);
    id.mhd_irecv = start.AddTask(&mhd::MHD::InitRecv, pmhd, none);

    // run task list
    id.copycons  = run.AddTask(&Radiation::CopyCons, this, none);

    id.rad_flux  = run.AddTask(&Radiation::CalculateFluxes, this, id.copycons);
    id.mhd_flux  = run.AddTask(&mhd::MHD::Fluxes, pmhd, id.rad_flux);

    id.rad_sendf = run.AddTask(&Radiation::SendFlux, this, id.mhd_flux);
    id.rad_recvf = run.AddTask(&Radiation::RecvFlux, this, id.rad_sendf);
    id.mhd_sendf = run.AddTask(&mhd::MHD::SendFlux, pmhd, id.rad_recvf);
    id.mhd_recvf = run.AddTask(&mhd::MHD::RecvFlux, pmhd, id.mhd_sendf);

    id.rad_expl  = run.AddTask(&Radiation::ExpRKUpdate, this, id.mhd_recvf);
    id.mhd_expl  = run.AddTask(&mhd::MHD::ExpRKUpdate, pmhd, id.rad_expl);

    id.mhd_efld  = run.AddTask(&mhd::MHD::CornerE, pmhd, id.mhd_expl);
    id.mhd_sende = run.AddTask(&mhd::MHD::SendE, pmhd, id.mhd_efld);
    id.mhd_recve = run.AddTask(&mhd::MHD::RecvE, pmhd, id.mhd_sende);

    id.mhd_ct    = run.AddTask(&mhd::MHD::CT, pmhd, id.mhd_recve);

    id.rad_src   = run.AddTask(&Radiation::AddRadiationSourceTerm, this, id.mhd_ct);

    id.rad_resti = run.AddTask(&Radiation::RestrictI, this, id.rad_src);
    id.mhd_restu = run.AddTask(&mhd::MHD::RestrictU, pmhd, id.rad_resti);
    id.mhd_restb = run.AddTask(&mhd::MHD::RestrictB, pmhd, id.mhd_restu);

    id.rad_sendi = run.AddTask(&Radiation::SendI, this, id.mhd_restb);
    id.rad_recvi = run.AddTask(&Radiation::RecvI, this, id.rad_sendi);

    id.mhd_sendu = run.AddTask(&mhd::MHD::SendU, pmhd, id.rad_recvi);
    id.mhd_recvu = run.AddTask(&mhd::MHD::RecvU, pmhd, id.mhd_sendu);

    id.mhd_sendb = run.AddTask(&mhd::MHD::SendB, pmhd, id.mhd_recvu);
    id.mhd_recvb = run.AddTask(&mhd::MHD::RecvB, pmhd, id.mhd_sendb);

    id.rad_bcs   = run.AddTask(&Radiation::ApplyPhysicalBCs, this, id.rad_recvi);
    id.mhd_bcs   = run.AddTask(&mhd::MHD::ApplyPhysicalBCs, pmhd, id.mhd_recvb);

    id.mhd_c2p   = run.AddTask(&mhd::MHD::ConToPrim, pmhd, id.mhd_bcs);

    // end task list
    id.rad_clear = end.AddTask(&Radiation::ClearSend, this, none);
    id.mhd_clear = end.AddTask(&mhd::MHD::ClearSend, pmhd, none);
  } else if (phyd != nullptr && !(fixed_fluid)) {  // radiation hydrodynamics
    // start task list
    id.rad_irecv = start.AddTask(&Radiation::InitRecv, this, none);
    id.hyd_irecv = start.AddTask(&hydro::Hydro::InitRecv, phyd, none);

    // run task list
    id.copycons = run.AddTask(&Radiation::CopyCons, this, none);

    id.rad_flux  = run.AddTask(&Radiation::CalculateFluxes, this, id.copycons);
    id.hyd_flux  = run.AddTask(&hydro::Hydro::Fluxes, phyd, id.rad_flux);

    id.rad_sendf = run.AddTask(&Radiation::SendFlux, this, id.hyd_flux);
    id.rad_recvf = run.AddTask(&Radiation::RecvFlux, this, id.rad_sendf);
    id.hyd_sendf = run.AddTask(&hydro::Hydro::SendFlux, phyd, id.rad_recvf);
    id.hyd_recvf = run.AddTask(&hydro::Hydro::RecvFlux, phyd, id.hyd_sendf);

    id.rad_expl  = run.AddTask(&Radiation::ExpRKUpdate, this, id.hyd_recvf);
    id.hyd_expl  = run.AddTask(&hydro::Hydro::ExpRKUpdate, phyd, id.rad_expl);

    id.rad_src   = run.AddTask(&Radiation::AddRadiationSourceTerm, this, id.hyd_expl);

    id.rad_resti = run.AddTask(&Radiation::RestrictI, this, id.rad_src);
    id.hyd_restu = run.AddTask(&hydro::Hydro::RestrictU, phyd, id.rad_src);

    id.rad_sendi = run.AddTask(&Radiation::SendI, this, id.rad_resti);
    id.hyd_sendu = run.AddTask(&hydro::Hydro::SendU, phyd, id.hyd_restu);

    id.rad_recvi = run.AddTask(&Radiation::RecvI, this, id.rad_sendi);
    id.hyd_recvu = run.AddTask(&hydro::Hydro::RecvU, phyd, id.hyd_sendu);

    id.rad_bcs   = run.AddTask(&Radiation::ApplyPhysicalBCs, this, id.rad_recvi);
    id.hyd_bcs   = run.AddTask(&hydro::Hydro::ApplyPhysicalBCs, phyd, id.hyd_recvu);

    id.hyd_c2p   = run.AddTask(&hydro::Hydro::ConToPrim, phyd, id.hyd_bcs);

    // end task list
    id.rad_clear = end.AddTask(&Radiation::ClearSend, this, none);
    id.hyd_clear = end.AddTask(&hydro::Hydro::ClearSend, phyd, none);
  } else {  // radiation
    // start task list
    id.rad_irecv = start.AddTask(&Radiation::InitRecv, this, none);

    // run task list
    id.copycons  = run.AddTask(&Radiation::CopyCons, this, none);
    id.rad_flux  = run.AddTask(&Radiation::CalculateFluxes, this, id.copycons);
    id.rad_sendf = run.AddTask(&Radiation::SendFlux, this, id.rad_flux);
    id.rad_recvf = run.AddTask(&Radiation::RecvFlux, this, id.rad_sendf);
    id.rad_expl  = run.AddTask(&Radiation::ExpRKUpdate, this, id.rad_recvf);
    id.rad_src   = run.AddTask(&Radiation::AddRadiationSourceTerm, this, id.rad_expl);
    id.rad_resti = run.AddTask(&Radiation::RestrictI, this, id.rad_src);
    id.rad_sendi = run.AddTask(&Radiation::SendI, this, id.rad_resti);
    id.rad_recvi = run.AddTask(&Radiation::RecvI, this, id.rad_sendi);
    id.rad_bcs   = run.AddTask(&Radiation::ApplyPhysicalBCs, this, id.rad_recvi);

    // end task list
    id.rad_clear = end.AddTask(&Radiation::ClearSend, this, none);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Radiation variables.

TaskStatus Radiation::InitRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_i->InitRecv(prgeo->nangles);
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_i->InitFluxRecv(prgeo->nangles);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::ClearRecv
//  \brief Waits for all MPI receives to complete before allowing execution to continue

TaskStatus Radiation::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_i->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_i->ClearFluxRecv();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::ClearSend
//  \brief Waits for all MPI sends to complete before allowing execution to continue

TaskStatus Radiation::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_i->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_i->ClearFluxSend();
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CopyCons
//  \brief  copy u0 --> u1 in first stage

TaskStatus Radiation::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), i1, i0);

    hydro::Hydro *phyd = pmy_pack->phydro;
    mhd::MHD *pmhd = pmy_pack->pmhd;
    if (pmhd != nullptr) {
      Kokkos::deep_copy(DevExeSpace(), pmhd->u1, pmhd->u0);
      Kokkos::deep_copy(DevExeSpace(), pmhd->b1.x1f, pmhd->b0.x1f);
      Kokkos::deep_copy(DevExeSpace(), pmhd->b1.x2f, pmhd->b0.x2f);
      Kokkos::deep_copy(DevExeSpace(), pmhd->b1.x3f, pmhd->b0.x3f);
    } else if (phyd != nullptr) {
      Kokkos::deep_copy(DevExeSpace(), phyd->u1, phyd->u0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SendI
//  \brief sends cell-centered conserved variables

TaskStatus Radiation::SendI(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_i->PackAndSendCC(i0, coarse_i0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::RecvU
//  \brief receives cell-centered conserved variables

TaskStatus Radiation::RecvI(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_i->RecvAndUnpackCC(i0, coarse_i0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::SendFlux
//  \brief

TaskStatus Radiation::SendFlux(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_i->PackAndSendFluxCC(iflx);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::RecvFlux
//  \brief

TaskStatus Radiation::RecvFlux(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  TaskStatus tstat = pbval_i->RecvAndUnpackFluxCC(iflx);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::RestrictI
//  \brief

TaskStatus Radiation::RestrictI(Driver *pdrive, int stage) {
  // Only execute this function with SMR/SMR
  if (!(pmy_pack->pmesh->multilevel)) return TaskStatus::complete;

  pmy_pack->pmesh->RestrictCC(i0, coarse_i0);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::ApplyPhysicalBCs
//  \brief

TaskStatus Radiation::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // only apply BCs if domain is not strictly periodic
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // physical BCs
    pbval_i->RadiationBCs((pmy_pack), (pbval_i->i_in), i0);

    // user BCs
    if (pmy_pack->pmesh->pgen->user_bcs) {
      (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
    }
  }
  return TaskStatus::complete;
}

} // namespace radiation
