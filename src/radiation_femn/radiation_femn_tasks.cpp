//========================================================================================
// GR radiation code for AthenaK with FEM_N & FP_N
// Copyright (C) 2023 Maitraya Bhattacharyya <mbb6217@psu.edu> and David Radice <dur566@psu.edu>
// AthenaXX copyright(C) James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_femn_tasks.cpp
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
#include "radiation_femn/radiation_femn.hpp"

namespace radiationfemn {
//----------------------------------------------------------------------------------------
//! \fn  void RadiationFEMN::AssembleRadiationTasks
//! \brief Adds radiation tasks to stage start/run/end task lists used by time integrators
//! Called by MeshBlockPack::AddPhysics() function directly after Radiation constructor
//! Many of the functions in the task list are implemented in this file because they are
//! simple, or they are wrappers that call one or more other functions.

void RadiationFEMN::AssembleRadiationFEMNTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);

  // assemble start task list
  id.rad_irecv = start.AddTask(&RadiationFEMN::InitRecv, this, none);

  // assemble run task list
  id.copycons = run.AddTask(&RadiationFEMN::CopyCons, this, none);
  id.rad_flux = run.AddTask(&RadiationFEMN::CalculateFluxes, this, id.copycons);
  id.rad_sendf = run.AddTask(&RadiationFEMN::SendFlux, this, id.rad_flux);
  id.rad_recvf = run.AddTask(&RadiationFEMN::RecvFlux, this, id.rad_sendf);
  id.rad_expl = run.AddTask(&RadiationFEMN::ExpRKUpdate, this, id.rad_recvf);
  id.rad_src = run.AddTask(&RadiationFEMN::AddRadiationSourceTerm, this, id.rad_expl);

  if (limiter_dg == "minmod2") {
    id.rad_limdg = run.AddTask(&RadiationFEMN::ApplyLimiterDG, this, id.rad_expl);
  } else {
    id.rad_limdg = id.rad_expl;
  }

  if (!fpn && limiter_fem == "clp") {
    id.rad_limfem = run.AddTask(&RadiationFEMN::ApplyLimiterFEM, this, id.rad_limdg);
    id.rad_filterfpn = run.AddTask(&RadiationFEMN::ApplyFilterLanczos, this, id.rad_limfem);
  } else if (fpn) {
    id.rad_limfem = run.AddTask(&RadiationFEMN::ApplyFilterLanczos, this, id.rad_limdg);
    id.rad_filterfpn = run.AddTask(&RadiationFEMN::ApplyFilterLanczos, this, id.rad_limfem);
  } else {
    id.rad_filterfpn = id.rad_limdg;
  }

  id.rad_resti = run.AddTask(&RadiationFEMN::RestrictI, this, id.rad_filterfpn);
  id.rad_sendi = run.AddTask(&RadiationFEMN::SendI, this, id.rad_resti);
  id.rad_recvi = run.AddTask(&RadiationFEMN::RecvI, this, id.rad_sendi);
  id.bcs = run.AddTask(&RadiationFEMN::ApplyPhysicalBCs, this, id.rad_recvi);

  // assemble end task list
  id.rad_csend = end.AddTask(&RadiationFEMN::ClearSend, this, none);
  // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.rad_crecv = end.AddTask(&RadiationFEMN::ClearRecv, this, id.rad_csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void RadiationFEMN::InitRecv
//  \brief function to post non-blocking receives (with MPI), and initialize all boundary
//  receive status flags to waiting (with or without MPI) for Radiation variables.

TaskStatus RadiationFEMN::InitRecv(Driver *pdrive, int stage) {
  // post receives for I
  TaskStatus tstat = pbval_f->InitRecv(num_points_total);
  if (tstat != TaskStatus::complete) return tstat;

  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR, post receives for fluxes of I
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_f->InitFluxRecv(num_points_total);
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void RadiationFEMN::CopyCons
//  \brief  copy f0 --> f1, L_mu_muhat0 --> L_mu_muhat1 in first stage

TaskStatus RadiationFEMN::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), f1, f0);
    Kokkos::deep_copy(DevExeSpace(), L_mu_muhat1, L_mu_muhat0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus RadiationFEMN::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_f->PackAndSendFluxCC(iflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus RadiationFEMN::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_f->RecvAndUnpackFluxCC(iflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::RestrictI
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus RadiationFEMN::RestrictI(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(f0, coarse_f0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::SendI
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus RadiationFEMN::SendI(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_f->PackAndSendCC(f0, coarse_f0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::RecvI
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus RadiationFEMN::RecvI(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_f->RecvAndUnpackCC(f0, coarse_f0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and user BCs

TaskStatus RadiationFEMN::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs on radiation
  // @TODO: Fix radiation BCs now that arrays have energy dependence
  //pbval_f->RadiationBCs((pmy_pack), (pbval_f->i_in), f0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.  Called
//! in end_tl, when all steps in run_tl over all MeshBlocks have completed.

TaskStatus RadiationFEMN::ClearSend(Driver *pdrive, int stage) {
  // check sends of I complete
  TaskStatus tstat = pbval_f->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  // do not check flux send for ICs (stage < 0)
  if (stage >= 0) {
    // with SMR/AMR check sends of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_f->ClearFluxSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus RadiationFEMN::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed.
//! Needed in Driver::Initialize to set ghost zones in ICs.

TaskStatus RadiationFEMN::ClearRecv(Driver *pdrive, int stage) {
  // check receives of U complete
  TaskStatus tstat = pbval_f->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  // do not check flux receives when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR check receives of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_f->ClearFluxRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

} // namespace radiation
