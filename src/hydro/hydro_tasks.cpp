//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_tasks.cpp
//! \brief functions that control Hydro tasks in the four task lists stored in the
//! MeshBlockPack: start_tl, run_tl, end_tl, operator_split_tl

#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AssembleHydroTasks
//! \brief Adds hydro tasks to stage start/run/end task lists used by time integrators
//! Called by MeshBlockPack::AddPhysics() function directly after Hydro constructor
//! Many of the functions in the task list are implemented in this file because they are
//! simple, or they are wrappers that call one or more other functions.
//!
//! start_tl tasks are those that must be cmpleted over all MeshBlocks before EACH
//! stage can be run (such as posting MPI receives, setting BoundaryCommStatus flags, etc)
//!
//! run_tl tasks are those performed in EACH stage
//!
//! end_tl tasks are those that can only be cmpleted after all the stage run tasks are
//! finished over all MeshBlocks for EACH stage, such as clearing all MPI non-blocking
//! sends, etc.

void Hydro::AssembleHydroTasks(TaskList &start, TaskList &run, TaskList &end) {
  TaskID none(0);

  // assemble start task list
  id.irecv = start.AddTask(&Hydro::InitRecv, this, none);

  // assemble run task list
  id.copyu = run.AddTask(&Hydro::CopyCons, this, none);
  id.flux  = run.AddTask(&Hydro::Fluxes,this,id.copyu);
  id.sendf = run.AddTask(&Hydro::SendFlux, this, id.flux);
  id.recvf = run.AddTask(&Hydro::RecvFlux, this, id.sendf);
  id.expl  = run.AddTask(&Hydro::ExpRKUpdate, this, id.recvf);
  id.restu = run.AddTask(&Hydro::RestrictU, this, id.expl);
  id.sendu = run.AddTask(&Hydro::SendU, this, id.restu);
  id.recvu = run.AddTask(&Hydro::RecvU, this, id.sendu);
  id.bcs   = run.AddTask(&Hydro::ApplyPhysicalBCs, this, id.recvu);
  id.prol  = run.AddTask(&Hydro::Prolongate, this, id.bcs);
  id.c2p   = run.AddTask(&Hydro::ConToPrim, this, id.prol);
  id.newdt = run.AddTask(&Hydro::NewTimeStep, this, id.c2p);

  // assemble end task list
  id.csend = end.AddTask(&Hydro::ClearSend, this, none);
  // although RecvFlux/U functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.crecv = end.AddTask(&Hydro::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI), and
//! initialize all boundary receive status flags to waiting (with or without MPI).

TaskStatus Hydro::InitRecv(Driver *pdrive, int stage) {
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nhydro+nscalars);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->InitFluxRecv(nhydro+nscalars);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::CopyCons
//! \brief Simple task list function that copies u0 --> u1 in first stage.  Extended to
//!  handle RK register logic at given stage

TaskStatus Hydro::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  } else {
    if (pdrive->integrator == "rk4") {
      // parallel loop to update u1 with u0 at later stages, only for rk4
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int is = indcs.is, ie = indcs.ie;
      int js = indcs.js, je = indcs.je;
      int ks = indcs.ks, ke = indcs.ke;
      int nmb1 = pmy_pack->nmb_thispack - 1;
      int nvar = nhydro + nscalars;
      auto &u0 = pmy_pack->phydro->u0;
      auto &u1 = pmy_pack->phydro->u1;
      Real &delta = pdrive->delta[stage-1];
      par_for("rk4_copy_cons", DevExeSpace(),0, nmb1, 0, nvar-1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        u1(m,n,k,j,i) += delta*u0(m,n,k,j,i);
      });
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Hydro::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute fluxes
//! of conserved variables

TaskStatus Hydro::Fluxes(Driver *pdrive, int stage) {
  // select which calculate_flux function to call based on rsolver_method
  if (rsolver_method == Hydro_RSolver::advect) {
    CalculateFluxes<Hydro_RSolver::advect>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf) {
    CalculateFluxes<Hydro_RSolver::llf>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle) {
    CalculateFluxes<Hydro_RSolver::hlle>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hllc) {
    CalculateFluxes<Hydro_RSolver::hllc>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::roe) {
    CalculateFluxes<Hydro_RSolver::roe>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf_sr) {
    CalculateFluxes<Hydro_RSolver::llf_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle_sr) {
    CalculateFluxes<Hydro_RSolver::hlle_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hllc_sr) {
    CalculateFluxes<Hydro_RSolver::hllc_sr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::llf_gr) {
    CalculateFluxes<Hydro_RSolver::llf_gr>(pdrive, stage);
  } else if (rsolver_method == Hydro_RSolver::hlle_gr) {
    CalculateFluxes<Hydro_RSolver::hlle_gr>(pdrive, stage);
  }

  // Add viscous, heat-flux, etc fluxes
  if (pvisc != nullptr) {
    pvisc->IsotropicViscousFlux(w0, pvisc->nu, peos->eos_data, uflx);
  }
  if (pcond != nullptr) {
    pcond->AddHeatFlux(w0, peos->eos_data, uflx);
  }

  // call FOFC if necessary
  if (use_fofc) {
    FOFC(pdrive, stage);
  } else if (pmy_pack->pcoord->is_general_relativistic) {
    if (pmy_pack->pcoord->coord_data.bh_excise) {
      FOFC(pdrive, stage);
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Hydro::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryVaLUES function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus Hydro::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus Hydro::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus Hydro::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus Hydro::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ApplyPhysicalBCs
//! \brief Wrapper task list function to call funtions that set physical and user BCs,

TaskStatus Hydro::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse boundaries with SMR/AMR

TaskStatus Hydro::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    if (pmy_pack->pmesh->pmr->prolong_prims) {
      pbval_u->ConsToPrimCoarseBndry(coarse_u0, coarse_w0);
      pbval_u->ProlongateCC(w0, coarse_w0);
      pbval_u->PrimToConsFineBndry(w0, u0);
    } else {
      pbval_u->ProlongateCC(u0, coarse_u0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ConToPrim
//! \brief Wrapper task list function to call ConsToPrim over entire mesh (including gz)

TaskStatus Hydro::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed.  Called
//! in end_tl, when all steps in run_tl over all MeshBlocks have completed.

TaskStatus Hydro::ClearSend(Driver *pdrive, int stage) {
  // check sends of U complete
  TaskStatus tstat = pbval_u->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR check sends of restricted fluxes of U complete
  // do not check flux send for ICs (stage < 0)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxSend();
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList Hydro::ClearRecv
//! \brief iWrapper task list function that checks all MPI receives have completed.
//! Needed in Driver::Initialize to set ghost zones in ICs.

TaskStatus Hydro::ClearRecv(Driver *pdrive, int stage) {
  // check receives of U complete
  TaskStatus tstat = pbval_u->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR check receives of restricted fluxes of U complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (pmy_pack->pmesh->multilevel && (stage >= 0)) {
    tstat = pbval_u->ClearFluxRecv();
  }
  return tstat;
}

} // namespace hydro
