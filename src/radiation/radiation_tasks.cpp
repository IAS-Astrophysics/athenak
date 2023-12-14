//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_tasks.cpp
//! \brief functions that control Radiation tasks in the four task lists stored in the
//! MeshBlockPack: start_tl, run_tl, end_tl, operator_split_tl (currently not used)

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
//! \brief Adds radiation tasks to stage start/run/end task lists used by time integrators
//! Called by MeshBlockPack::AddPhysics() function directly after Radiation constructor
//! Many of the functions in the task list are implemented in this file because they are
//! simple, or they are wrappers that call one or more other functions.

void Radiation::AssembleRadiationTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);
  hydro::Hydro *phyd = pmy_pack->phydro;
  mhd::MHD *pmhd = pmy_pack->pmhd;

  // construct task list depending on enabled physics modules and radiation parameters
  if (pmhd != nullptr && !(fixed_fluid)) {  // radiation magnetohydrodynamics
    // assemble start task list
    id.rad_irecv = start.AddTask(&Radiation::InitRecv, this, none);
    id.mhd_irecv = start.AddTask(&mhd::MHD::InitRecv, pmhd, none);

    // assemble run task list
    id.copycons  = run.AddTask(&Radiation::CopyCons, this, none);
    id.rad_flux  = run.AddTask(&Radiation::CalculateFluxes, this, id.copycons);
    id.rad_sendf = run.AddTask(&Radiation::SendFlux, this, id.rad_flux);
    id.rad_recvf = run.AddTask(&Radiation::RecvFlux, this, id.rad_sendf);
    id.rad_expl  = run.AddTask(&Radiation::ExpRKUpdate, this, id.rad_recvf);
    id.mhd_flux  = run.AddTask(&mhd::MHD::Fluxes, pmhd, id.rad_expl);
    id.mhd_sendf = run.AddTask(&mhd::MHD::SendFlux, pmhd, id.mhd_flux);
    id.mhd_recvf = run.AddTask(&mhd::MHD::RecvFlux, pmhd, id.mhd_sendf);
    id.mhd_expl  = run.AddTask(&mhd::MHD::ExpRKUpdate, pmhd, id.mhd_recvf);
    id.mhd_efld  = run.AddTask(&mhd::MHD::CornerE, pmhd, id.mhd_expl);
    id.mhd_sende = run.AddTask(&mhd::MHD::SendE, pmhd, id.mhd_efld);
    id.mhd_recve = run.AddTask(&mhd::MHD::RecvE, pmhd, id.mhd_sende);
    id.mhd_ct    = run.AddTask(&mhd::MHD::CT, pmhd, id.mhd_recve);
    id.rad_src   = run.AddTask(&Radiation::AddRadiationSourceTerm, this, id.mhd_ct);
    id.rad_resti = run.AddTask(&Radiation::RestrictI, this, id.rad_src);
    id.rad_sendi = run.AddTask(&Radiation::SendI, this, id.rad_resti);
    id.rad_recvi = run.AddTask(&Radiation::RecvI, this, id.rad_sendi);
    id.mhd_restu = run.AddTask(&mhd::MHD::RestrictU, pmhd, id.rad_recvi);
    id.mhd_sendu = run.AddTask(&mhd::MHD::SendU, pmhd, id.mhd_restu);
    id.mhd_recvu = run.AddTask(&mhd::MHD::RecvU, pmhd, id.mhd_sendu);
    id.mhd_restb = run.AddTask(&mhd::MHD::RestrictB, pmhd, id.mhd_recvu);
    id.mhd_sendb = run.AddTask(&mhd::MHD::SendB, pmhd, id.mhd_restb);
    id.mhd_recvb = run.AddTask(&mhd::MHD::RecvB, pmhd, id.mhd_sendb);
    id.bcs       = run.AddTask(&Radiation::ApplyPhysicalBCs, this, id.mhd_recvb);
    id.rad_prol  = run.AddTask(&Radiation::Prolongate, this, id.bcs);
    id.mhd_prol  = run.AddTask(&mhd::MHD::Prolongate, pmhd, id.rad_prol);
    id.mhd_c2p   = run.AddTask(&mhd::MHD::ConToPrim, pmhd, id.mhd_prol);

    // assemble end task list
    id.rad_csend = end.AddTask(&Radiation::ClearSend, this, none);
    id.mhd_csend = end.AddTask(&mhd::MHD::ClearSend, pmhd, none);
    // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
    // task list anyways to catch potential bugs in MPI communication logic
    id.rad_crecv = end.AddTask(&Radiation::ClearRecv, this, id.rad_csend);
    id.mhd_crecv = end.AddTask(&mhd::MHD::ClearRecv, pmhd, id.mhd_csend);

  } else if (phyd != nullptr && !(fixed_fluid)) {  // radiation hydrodynamics
    // assemble start task list
    id.rad_irecv = start.AddTask(&Radiation::InitRecv, this, none);
    id.hyd_irecv = start.AddTask(&hydro::Hydro::InitRecv, phyd, none);

    // assemble run task list
    id.copycons = run.AddTask(&Radiation::CopyCons, this, none);
    id.rad_flux  = run.AddTask(&Radiation::CalculateFluxes, this, id.copycons);
    id.rad_sendf = run.AddTask(&Radiation::SendFlux, this, id.rad_flux);
    id.rad_recvf = run.AddTask(&Radiation::RecvFlux, this, id.rad_sendf);
    id.rad_expl  = run.AddTask(&Radiation::ExpRKUpdate, this, id.rad_recvf);
    id.hyd_flux  = run.AddTask(&hydro::Hydro::Fluxes, phyd, id.rad_expl);
    id.hyd_sendf = run.AddTask(&hydro::Hydro::SendFlux, phyd, id.hyd_flux);
    id.hyd_recvf = run.AddTask(&hydro::Hydro::RecvFlux, phyd, id.hyd_sendf);
    id.hyd_expl  = run.AddTask(&hydro::Hydro::ExpRKUpdate, phyd, id.hyd_recvf);
    id.rad_src   = run.AddTask(&Radiation::AddRadiationSourceTerm, this, id.hyd_expl);
    id.rad_resti = run.AddTask(&Radiation::RestrictI, this, id.rad_src);
    id.rad_sendi = run.AddTask(&Radiation::SendI, this, id.rad_resti);
    id.rad_recvi = run.AddTask(&Radiation::RecvI, this, id.rad_sendi);
    id.hyd_restu = run.AddTask(&hydro::Hydro::RestrictU, phyd, id.rad_recvi);
    id.hyd_sendu = run.AddTask(&hydro::Hydro::SendU, phyd, id.hyd_restu);
    id.hyd_recvu = run.AddTask(&hydro::Hydro::RecvU, phyd, id.hyd_sendu);
    id.bcs       = run.AddTask(&Radiation::ApplyPhysicalBCs, this, id.hyd_recvu);
    id.rad_prol  = run.AddTask(&Radiation::Prolongate, this, id.bcs);
    id.hyd_prol  = run.AddTask(&hydro::Hydro::Prolongate, phyd, id.rad_prol);
    id.hyd_c2p   = run.AddTask(&hydro::Hydro::ConToPrim, phyd, id.hyd_prol);

    // assemble end task list
    id.rad_csend = end.AddTask(&Radiation::ClearSend, this, none);
    id.hyd_csend = end.AddTask(&hydro::Hydro::ClearSend, phyd, none);
    // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
    // task list anyways to catch potential bugs in MPI communication logic
    id.rad_crecv = end.AddTask(&Radiation::ClearRecv, this, id.rad_csend);
    id.hyd_crecv = end.AddTask(&hydro::Hydro::ClearRecv, phyd, id.hyd_csend);

  } else {  // radiation transport
    // assemble start task list
    id.rad_irecv = start.AddTask(&Radiation::InitRecv, this, none);

    // assemble run task list
    id.copycons = run.AddTask(&Radiation::CopyCons, this, none);
    id.rad_flux  = run.AddTask(&Radiation::CalculateFluxes, this, id.copycons);
    id.rad_sendf = run.AddTask(&Radiation::SendFlux, this, id.rad_flux);
    id.rad_recvf = run.AddTask(&Radiation::RecvFlux, this, id.rad_sendf);
    id.rad_expl  = run.AddTask(&Radiation::ExpRKUpdate, this, id.rad_recvf);
    id.rad_src   = run.AddTask(&Radiation::AddRadiationSourceTerm, this, id.rad_expl);
    id.rad_resti = run.AddTask(&Radiation::RestrictI, this, id.rad_src);
    id.rad_sendi = run.AddTask(&Radiation::SendI, this, id.rad_resti);
    id.rad_recvi = run.AddTask(&Radiation::RecvI, this, id.rad_sendi);
    id.bcs       = run.AddTask(&Radiation::ApplyPhysicalBCs, this, id.rad_recvi);
    id.rad_prol  = run.AddTask(&Radiation::Prolongate, this, id.bcs);

    // assemble end task list
    id.rad_csend = end.AddTask(&Radiation::ClearSend, this, none);
    // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
    // task list anyways to catch potential bugs in MPI communication logic
    id.rad_crecv = end.AddTask(&Radiation::ClearRecv, this, id.rad_csend);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Radiation variables.

TaskStatus Radiation::InitRecv(Driver *pdrive, int stage) {
  // post receives for I
  TaskStatus tstat = pbval_i->InitRecv(prgeo->nangles);
  if (tstat != TaskStatus::complete) return tstat;

  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR, post receives for fluxes of I
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_i->InitFluxRecv(prgeo->nangles);
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CopyCons
//  \brief  copy u0 --> u1 in first stage

TaskStatus Radiation::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    // radiation
    Kokkos::deep_copy(DevExeSpace(), i1, i0);

    // hydro and MHD (if enabled)
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
//! \fn TaskStatus Radiation::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Radiation::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel)  {
    tstat = pbval_i->PackAndSendFluxCC(iflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Radiation::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_i->RecvAndUnpackFluxCC(iflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::RestrictI
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus Radiation::RestrictI(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(i0, coarse_i0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::SendI
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus Radiation::SendI(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_i->PackAndSendCC(i0, coarse_i0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::RecvI
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus Radiation::RecvI(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_i->RecvAndUnpackCC(i0, coarse_i0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and user BCs

TaskStatus Radiation::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs on radiation
  pbval_i->RadiationBCs((pmy_pack), (pbval_i->i_in), i0);

  // physical BCs on (M)HD
  hydro::Hydro *phyd = pmy_pack->phydro;
  mhd::MHD *pmhd = pmy_pack->pmhd;
  if (pmhd != nullptr) {
    pmhd->pbval_u->HydroBCs((pmy_pack), (pmhd->pbval_u->u_in), pmhd->u0);
    pmhd->pbval_b->BFieldBCs((pmy_pack), (pmhd->pbval_b->b_in), pmhd->b0);
  } else if (phyd != nullptr) {
    phyd->pbval_u->HydroBCs((pmy_pack), (phyd->pbval_u->u_in), phyd->u0);
  }

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Radiation::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse bundaries with SMR/AMR

TaskStatus Radiation::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    // prolongate specific intensity
    pbval_i->FillCoarseInBndryCC(i0, coarse_i0);
    pbval_i->ProlongateCC(i0, coarse_i0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.  Called
//! in end_tl, when all steps in run_tl over all MeshBlocks have completed.

TaskStatus Radiation::ClearSend(Driver *pdrive, int stage) {
  // check sends of I complete
  TaskStatus tstat = pbval_i->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  // do not check flux send for ICs (stage < 0)
  if (stage >= 0) {
    // with SMR/AMR check sends of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_i->ClearFluxSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Radiation::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed.
//! Needed in Driver::Initialize to set ghost zones in ICs.

TaskStatus Radiation::ClearRecv(Driver *pdrive, int stage) {
  // check receives of U complete
  TaskStatus tstat = pbval_i->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  // do not check flux receives when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR check receives of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_i->ClearFluxRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

} // namespace radiation
